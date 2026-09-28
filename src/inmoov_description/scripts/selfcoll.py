#!/usr/bin/env python3
"""Self-collision matrix for the InMoov URDF (an analogue of the MoveIt Setup
Assistant's "Self-Collisions" step), used to build description/inmoov_i2.srdf.

For every pair of links with collision geometry it classifies:
  adjacent  -- neighbours in the kinematic tree (links without collision
               geometry are collapsed, so "Adjacent (via X)" skips over them);
  default   -- already in contact in the zero pose (with penetration depth);
  always    -- colliding in >= 95% of random poses;
  never     -- never colliding in N random poses;
  check     -- sometimes colliding (real collisions, keep them enabled).
Random poses are uniform within the joint limits; half of them push ~30% of
the joints to a limit to cover extreme poses. Mimic joints follow their
source joint (multiplier/offset).

The result is written to selfcoll_result.json in the current directory; the
SRDF's <disable_collisions> entries are then chosen from it by hand.

Usage (needs numpy, trimesh and python-fcl: pip install trimesh python-fcl):
  xacro ../description/inmoov_i2.urdf.xacro > /tmp/inmoov.urdf
  python3 selfcoll.py /tmp/inmoov.urdf 20000

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import sys, os, json, itertools, xml.etree.ElementTree as ET
import numpy as np, fcl, trimesh

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URDF = sys.argv[1] if len(sys.argv) > 1 else 'inmoov.urdf'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
rng = np.random.default_rng(0)
root = ET.parse(URDF).getroot()

def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

def origin(el):
    T = np.eye(4)
    o = el.find('origin') if el is not None else None
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get('xyz', '0 0 0').split()]
        T[:3, :3] = rpy2R(*[float(v) for v in o.get('rpy', '0 0 0').split()])
    return T

def mesh_path(uri):
    # package://inmoov_description/... -> this package's directory
    prefix = 'package://inmoov_description/'
    return os.path.join(PKG, uri[len(prefix):]) if uri.startswith(prefix) else uri

def axis_angle(a, q):
    a = a / np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(4); R[:3, :3] = np.eye(3) + np.sin(q)*K + (1-np.cos(q))*K@K
    return R

# ---------- kinematic tree ----------
joints = {}
for j in root.findall('joint'):
    lim = j.find('limit'); m = j.find('mimic')
    joints[j.get('name')] = dict(
        type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
        T=origin(j), axis=np.array([float(v) for v in (j.find('axis').get('xyz') if j.find('axis') is not None else '1 0 0').split()]),
        lo=float(lim.get('lower', 0)) if lim is not None else 0.0, hi=float(lim.get('upper', 0)) if lim is not None else 0.0,
        mimic=(m.get('joint'), float(m.get('multiplier', 1)), float(m.get('offset', 0))) if m is not None else None)
child2joint = {v['child']: k for k, v in joints.items()}
links = [l.get('name') for l in root.findall('link')]
base = [l for l in links if l not in child2joint][0]
order = []  # topological joint order
def walk(link):
    for k, v in joints.items():
        if v['parent'] == link: order.append(k); walk(v['child'])
walk(base)
active = [k for k in order if joints[k]['type'] in ('revolute', 'continuous', 'prismatic') and joints[k]['mimic'] is None]

def fk(q):
    TW = {base: np.eye(4)}
    for k in order:
        j = joints[k]
        if j['type'] == 'fixed': val = 0.0
        elif j['mimic']: src, mul, off = j['mimic']; val = q[src]*mul + off
        else: val = q[k]
        q[k] = val
        TW[j['child']] = TW[j['parent']] @ j['T'] @ (axis_angle(j['axis'], val) if j['type'] != 'fixed' else np.eye(4))
    return TW

# ---------- collision geometry ----------
spheres = []
geoms = {}  # link -> list of (fcl.CollisionObject, T_link_geom)
for l in root.findall('link'):
    for c in l.findall('collision'):
        g = c.find('geometry')[0]; Tl = origin(c)
        if g.tag == 'box': geo = fcl.Box(*[float(v) for v in g.get('size').split()])
        elif g.tag == 'cylinder': geo = fcl.Cylinder(float(g.get('radius')), float(g.get('length')))
        elif g.tag == 'sphere': geo = fcl.Sphere(float(g.get('radius')))
        elif g.tag == 'mesh':
            m = trimesh.load(mesh_path(g.get('filename')), force='mesh')
            sc = [float(v) for v in g.get('scale', '1 1 1').split()]
            m.apply_scale(sc)
            geo = fcl.BVHModel(); geo.beginModel(len(m.vertices), len(m.faces))
            geo.addSubModel(m.vertices, m.faces); geo.endModel()
        if g.tag == 'box': ctr, rad = np.zeros(3), np.linalg.norm([float(v) for v in g.get('size').split()])/2
        elif g.tag == 'cylinder': ctr, rad = np.zeros(3), np.hypot(float(g.get('radius')), float(g.get('length'))/2)
        elif g.tag == 'sphere': ctr, rad = np.zeros(3), float(g.get('radius'))
        else:
            ctr = (m.vertices.min(0)+m.vertices.max(0))/2; rad = np.linalg.norm(m.vertices-ctr, axis=1).max()
        geoms.setdefault(l.get('name'), []).append((fcl.CollisionObject(geo), Tl))
        spheres.append((l.get('name'), len(geoms[l.get('name')])-1, Tl[:3,:3]@ctr+Tl[:3,3], rad))
clinks = [l for l in links if l in geoms]
obj2link = {id(o): ln for ln, lst in geoms.items() for o, _ in lst}

def place(TW):
    for ln, lst in geoms.items():
        for o, Tl in lst:
            T = TW[ln] @ Tl; o.setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))

SL = [sp[0] for sp in spheres]; SR = np.array([sp[3] for sp in spheres])
IU = np.triu_indices(len(spheres), 1)
cand_mask = np.array([SL[i] != SL[j] for i, j in zip(*IU)])
def colliding_pairs(TW):
    place(TW)
    C = np.array([TW[ln][:3, :3] @ c + TW[ln][:3, 3] for ln, _, c, _ in spheres])
    d = np.linalg.norm(C[IU[0]] - C[IU[1]], axis=1)
    hit = (d < SR[IU[0]] + SR[IU[1]]) & cand_mask
    found = set()
    for i, j in zip(IU[0][hit], IU[1][hit]):
        a, b = SL[i], SL[j]; p = tuple(sorted((a, b)))
        if p in found: continue
        if fcl.collide(geoms[a][spheres[i][1]][0], geoms[b][spheres[j][1]][0], fcl.CollisionRequest(), fcl.CollisionResult()):
            found.add(p)
    return found

# ---------- adjacency (collapse links without collision geometry) ----------
def coll_ancestor(link):
    while link in child2joint:
        link = joints[child2joint[link]]['parent']
        if link in geoms: return link
    return None
adjacent = {}
for l in clinks:
    a = coll_ancestor(l)
    if a:
        path = []; x = l
        while x != a: path.append(x); x = joints[child2joint[x]]['parent']
        adjacent[tuple(sorted((l, a)))] = 'Adjacent' if len(path) == 1 else 'Adjacent (via ' + ', '.join(path[1:]) + ')'

# ---------- zero pose ----------
q0 = {k: 0.0 for k in active}
default = colliding_pairs(fk(dict(q0)))
depth = {}
TW0 = fk(dict(q0)); place(TW0)
for a, b in default:
    best = 0.0
    for o1, _ in geoms[a]:
        for o2, _ in geoms[b]:
            rq = fcl.CollisionRequest(num_max_contacts=50, enable_contact=True); rs = fcl.CollisionResult()
            if fcl.collide(o1, o2, rq, rs):
                best = max([best] + [c.penetration_depth for c in rs.contacts])
    depth[(a, b)] = best

# ---------- random sampling ----------
lo = np.array([joints[k]['lo'] for k in active]); hi = np.array([joints[k]['hi'] for k in active])
count = {}
for i in range(N):
    u = rng.random(len(active))
    x = lo + u*(hi - lo)
    if i % 2:  # half of samples: ~30% of joints pushed to a limit (extreme poses)
        mask = rng.random(len(active)) < 0.3
        x[mask] = np.where(rng.random(mask.sum()) < 0.5, lo[mask], hi[mask])
    for p in colliding_pairs(fk(dict(zip(active, x)))):
        count[p] = count.get(p, 0) + 1

allpairs = [tuple(sorted(p)) for p in itertools.combinations(clinks, 2)]
res = []
for p in allpairs:
    f = count.get(p, 0)/N
    if p in adjacent: cat = 'adjacent'
    elif p in default: cat = 'default'
    elif f >= 0.95: cat = 'always'
    elif f == 0: cat = 'never'
    else: cat = 'check'
    res.append(dict(a=p[0], b=p[1], cat=cat, freq=f, zero_pose=p in default,
                    depth_mm=round(depth.get(p, 0)*1000, 1), note=adjacent.get(p, '')))
json.dump(dict(N=N, active=len(active), clinks=len(clinks), pairs=res), open('selfcoll_result.json', 'w'), indent=1, ensure_ascii=False)
from collections import Counter
print('samples', N, 'active joints', len(active), 'links w/ collision', len(clinks), 'pairs', len(allpairs))
print(Counter(r['cat'] for r in res))
for r in sorted(res, key=lambda r: (r['cat'], -r['freq'])):
    if r['cat'] != 'never':
        print(f"{r['cat']:9s} {r['freq']*100:6.2f}%  zero={r['zero_pose']!s:5s} {r['depth_mm']:6.1f}mm  {r['a']} <-> {r['b']}  {r['note']}")
