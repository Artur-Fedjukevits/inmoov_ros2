#!/usr/bin/env python3
"""Estimate mass, center of mass and full inertia tensor of every InMoov link
from its visual meshes; writes inertia_estimate.json / inertia_estimate.csv
next to this script.

Method (all quantities in meters / kg, in the URDF link frame):
  * each visual sub-mesh is replaced by its convex hull (the source meshes
    are not watertight, so their own volume is unreliable);
  * link mass = k * V_hull + mass of the servos mounted in that link
    (SERVOS table below);
  * mass is assumed uniform over the hulls of a link (servo mass is "smeared"
    over the link volume -- a simplification);
  * k (effective density) is fitted so the robot without its stand sums to
    ROBOT_MASS;
  * stand_link gets the catalog mass of its IKEA parts.

Reads description/inmoov_i2.urdf.xacro (visual origins/meshes), meshes/ and
model_scale from description/properties.xacro, so run it after
xacrify_scale.py whenever meshes or weights change, then re-apply:

  python3 claude_compute_inertia.py
  python3 apply_inertia.py ../description/inmoov_i2.urdf.xacro inertia_estimate.json

Needs numpy and trimesh (pip install trimesh) -- only for this recomputation,
not for building or displaying the model.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import csv
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
XACRO = os.path.join(PKG, 'description', 'inmoov_i2.urdf.xacro')
PROPERTIES = os.path.join(PKG, 'description', 'properties.xacro')
MESHDIR = os.path.join(PKG, 'meshes')
OUT_JSON = os.path.join(HERE, 'inertia_estimate.json')
OUT_CSV = os.path.join(HERE, 'inertia_estimate.csv')

TOTAL_WEIGHED = 23.0            # kg, robot + stand, measured on scales
STAND_MASS = 1.34 + 5.54        # kg: IKEA OLOV (1.34) + MALSKÄR (5.54, catalog weight incl. packaging)
ROBOT_MASS = TOTAL_WEIGHED - STAND_MASS
EMPTY_MASS, EMPTY_I = 0.001, 1e-9   # kinematic-only links without geometry
PLA_DENSITY = 1240.0            # kg/m^3, solid PLA (for the fill ratio report)

# Servos per link, kg (typical InMoov build; HS-805BB 0.152, MG996R 0.055, SG90/MG90 ~0.010)
SERVOS = {
    'torso_bottom_link': 2*0.152,                  # stomach (2x HS-805BB)
    'torso_y_link': 4*0.152 + 2*0.055,             # 2x omoplate + 2x shoulder, 2x neck
    'left_shoulder_x_link': 0.152, 'right_shoulder_x_link': 0.152,     # rotate
    'left_shoulder_z_link': 0.152, 'right_shoulder_z_link': 0.152,     # bicep
    'left_elbow_x_link': 6*0.055, 'right_elbow_x_link': 6*0.055,       # 5 fingers + wrist (in the forearm)
    'neck_link': 0.055,                            # rothead
    'NewHead_origin': 0.055 + 10*0.010,            # jaw + small i2 face servos
}


def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])


def read_model_scale():
    m = re.search(r'name="model_scale"\s+value="([^"]+)"', open(PROPERTIES, encoding='utf-8').read())
    return float(m.group(1))


def load_urdf(S):
    # Minimal xacro expansion: only ${model_scale*...} expressions are used in the file.
    s = open(XACRO, encoding='utf-8').read()
    s = re.sub(r'\$\{model_scale\*model_scale\*([-0-9.eE]+)\}', lambda m: repr(S*S*float(m.group(1))), s)
    s = re.sub(r'\$\{model_scale\*([-0-9.eE]+)\}', lambda m: repr(S*float(m.group(1))), s)
    s = s.replace('${model_scale}', str(S))
    s = re.sub(r'<xacro:include[^>]*/>', '', s)
    return ET.fromstring(s)


def main():
    S = read_model_scale()
    root = load_urdf(S)

    links = {}
    for l in root.findall('link'):
        name = l.get('name')
        V = 0.0
        M1 = np.zeros(3)
        J = np.zeros((3, 3))
        for v in l.findall('visual'):
            g = v.find('geometry/mesh')
            o = v.find('origin')
            m = trimesh.load(os.path.join(MESHDIR, g.get('filename').split('/')[-1]), force='mesh')
            m.apply_scale([float(x) for x in g.get('scale', '1 1 1').split()])
            T = np.eye(4)
            if o is not None:
                T[:3, 3] = [float(x) for x in o.get('xyz', '0 0 0').split()]
                T[:3, :3] = rpy2R(*[float(x) for x in o.get('rpy', '0 0 0').split()])
            m.apply_transform(T)
            h = m.convex_hull
            vol, c = h.volume, h.center_mass
            Ic = h.moment_inertia            # density 1 -> mass = volume, tensor about c
            # Tensor about the link frame origin (parallel axis theorem); re-centred on the link COM below.
            J += Ic + vol*(np.dot(c, c)*np.eye(3) - np.outer(c, c))
            V += vol
            M1 += vol*c
        links[name] = dict(V=V, M1=M1, J=J)

    robot_V = sum(d['V'] for n, d in links.items() if n != 'stand_link')
    servo_total = sum(SERVOS.values())
    n_empty = sum(1 for d in links.values() if d['V'] == 0)
    k = (ROBOT_MASS - servo_total - EMPTY_MASS*n_empty) / robot_V

    out = {}
    for name, d in links.items():
        if d['V'] == 0:
            out[name] = dict(mass=EMPTY_MASS, com=[0, 0, 0], I=[[EMPTY_I, 0, 0], [0, EMPTY_I, 0], [0, 0, EMPTY_I]],
                             V_cm3=0, servo=0, note='no geometry')
            continue
        c = d['M1']/d['V']
        Jc = d['J'] - d['V']*(np.dot(c, c)*np.eye(3) - np.outer(c, c))   # about the COM, per unit density
        if name == 'stand_link':
            mass, note = STAND_MASS, 'IKEA catalog'
        else:
            mass, note = k*d['V'] + SERVOS.get(name, 0.0), 'estimate'
        I = Jc*(mass/d['V'])
        w = np.linalg.eigvalsh(I)
        assert w.min() > 0 and w[0] + w[1] >= w[2]*0.999, (name, w)
        out[name] = dict(mass=mass, com=c.tolist(), I=I.tolist(), V_cm3=d['V']*1e6,
                         servo=SERVOS.get(name, 0.0), note=note)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(k_kg_per_m3=k, fill_vs_PLA=k/PLA_DENSITY, robot_mass=ROBOT_MASS, stand_mass=STAND_MASS,
                       model_scale=S, links=out), f, indent=1, ensure_ascii=False)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['link', 'mass_kg', 'servo_kg', 'hull_cm3', 'com_x', 'com_y', 'com_z',
                     'ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz', 'note'])
        for n, r in sorted(out.items(), key=lambda x: -x[1]['mass']):
            I = r['I']
            wr.writerow([n, round(r['mass'], 4), r['servo'], round(r['V_cm3'], 1), *[round(x, 5) for x in r['com']],
                         *['%.3e' % x for x in (I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2])], r['note']])

    print(f'k = {k:.1f} kg/m^3 (= {k/PLA_DENSITY:.2f} of solid PLA), robot {ROBOT_MASS:.2f} kg, servos {servo_total:.2f} kg')
    for n, r in sorted(out.items(), key=lambda x: -x[1]['mass'])[:20]:
        print(f"{n:34s} {r['mass']:7.3f} kg  hull {r['V_cm3']:8.1f} cm^3  servo {r['servo']:.3f}")
    print('total', round(sum(r['mass'] for r in out.values()), 3))


if __name__ == '__main__':
    main()
