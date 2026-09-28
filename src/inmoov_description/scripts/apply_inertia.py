#!/usr/bin/env python3
"""Replace the <inertial> blocks in inmoov_i2.urdf.xacro with the estimate
from inertia_estimate.json (produced by claude_compute_inertia.py).

Values are written directly in SI units (kg, m, kg*m^2), without
${model_scale}, with 6 significant digits: LinkForge rounds to 6 decimal
places, which would zero out the tensors of fingertips/eyelids
(~1e-7 kg*m^2) or break the triangle inequality, and Gazebo rejects that.

xacrify_scale.py calls apply_inertia() itself, so after a fresh LinkForge
export a single xacrify_scale.py run is enough. Run this script on its own
only to re-apply a recomputed JSON to an existing xacro:

  python3 apply_inertia.py ../description/inmoov_i2.urdf.xacro inertia_estimate.json

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import json
import re
import sys

LINK_RE = re.compile(r'<link name="([^"]+)">(.*?)</link>', re.S)
INERTIAL_RE = re.compile(r'<inertial>.*?</inertial>', re.S)


def _g(v):
    return f'{v:.6g}'


def load_estimate(json_path):
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)['links']


def apply_inertia(text, data):
    """Return (new_text, number_of_links_replaced, total_mass_kg)."""
    done = set()

    def repl_link(m):
        name, body = m.group(1), m.group(2)
        if name not in data:
            return m.group(0)
        r = data[name]
        c, I = r['com'], r['I']
        new = (f'<inertial>\n      <origin xyz="{_g(c[0])} {_g(c[1])} {_g(c[2])}" rpy="0 0 0" />\n'
               f'      <mass value="{_g(r["mass"])}" />\n'
               f'      <inertia ixx="{_g(I[0][0])}" ixy="{_g(I[0][1])}" ixz="{_g(I[0][2])}" '
               f'iyy="{_g(I[1][1])}" iyz="{_g(I[1][2])}" izz="{_g(I[2][2])}" />\n    </inertial>')
        body2, n = INERTIAL_RE.subn(lambda _: new, body, count=1)
        if n == 0:
            body2 = '\n    ' + new + body
        done.add(name)
        return f'<link name="{name}">{body2}</link>'

    text = LINK_RE.sub(repl_link, text)
    missing = set(data) - done
    if missing:
        raise ValueError(f'links from the estimate not found in the xacro: {sorted(missing)}')
    unestimated = {n for n, _ in LINK_RE.findall(text)} - set(data)
    if unestimated:
        print(f'warning: links without an estimate keep their exported inertia: {sorted(unestimated)}')
    return text, len(done), sum(d['mass'] for d in data.values())


def main():
    xacro_path, json_path = sys.argv[1], sys.argv[2]
    text = open(xacro_path, encoding='utf-8').read()
    text, n, total = apply_inertia(text, load_estimate(json_path))
    open(xacro_path, 'w', encoding='utf-8').write(text)
    print(f'inertial replaced on {n} links, total mass {total:.2f} kg')


if __name__ == '__main__':
    main()
