#!/usr/bin/env python3
"""Convert a raw LinkForge URDF export into description/inmoov_i2.urdf.xacro:

- meshes\\name.obj (Windows-style, as LinkForge writes them) -> package://inmoov_description/meshes/name.obj
- every <origin xyz>, <mesh>, and primitive collision geometry (<box size>,
  <cylinder radius/length>, <sphere radius>) scaled by a model_scale xacro
  property, and every <inertia> component by model_scale^2 (mass * length^2)
  (LinkForge exports raw Blender scene units, not meters) — the
  same pattern the old Sentience-Robotics inmoov_urdf properties.xacro
  used. properties.xacro itself is untouched; model_scale is hand-maintained.
- every <inertial> then replaced by the mass/COM/inertia estimate from
  inertia_estimate.json (see apply_inertia.py / claude_compute_inertia.py),
  in SI units. The model_scale^2 inertia above is only the fallback for
  --no-inertia or links missing from the estimate.

Usage: xacrify_scale.py <in.urdf> <out.urdf.xacro> [--inertia FILE.json | --no-inertia]
  (default --inertia: inertia_estimate.json next to this script)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import argparse
import os
import re

from apply_inertia import apply_inertia, load_estimate

DEFAULT_INERTIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inertia_estimate.json")

MESH_PATH_RE = re.compile(r'(<mesh\b[^>]*\bfilename=")meshes\\([^"]+)(")')
ORIGIN_XYZ_RE = re.compile(r'(<origin\b[^>]*\bxyz=")([^"]+)(")')
MESH_RE = re.compile(r'(<mesh\b[^>]*\bfilename="[^"]+")\s*/>')
BOX_SIZE_RE = re.compile(r'(<box\b[^>]*\bsize=")([^"]+)(")')
CYLINDER_RE = re.compile(r'(<cylinder\b[^>]*\bradius=")([^"]+)("[^>]*\blength=")([^"]+)(")')
SPHERE_RADIUS_RE = re.compile(r'(<sphere\b[^>]*\bradius=")([^"]+)(")')
INERTIA_RE = re.compile(r'<inertia\b[^>]*/>')
INERTIA_ATTR_RE = re.compile(r'\b(i(?:xx|xy|xz|yy|yz|zz))="([^"]+)"')
ROBOT_OPEN_RE = re.compile(r'<robot\b([^>]*)>')


def fix_mesh_path(m):
    prefix, name, suffix = m.group(1), m.group(2), m.group(3)
    return f'{prefix}package://inmoov_description/meshes/{name}{suffix}'


def _scaled(value):
    return "0" if float(value) == 0 else f"${{model_scale*{value}}}"


def scale_xyz(m):
    prefix, xyz, suffix = m.group(1), m.group(2), m.group(3)
    return prefix + " ".join(_scaled(p) for p in xyz.split()) + suffix


def add_mesh_scale(m):
    return m.group(1) + ' scale="${model_scale} ${model_scale} ${model_scale}"/>'


def scale_box_size(m):
    prefix, size, suffix = m.group(1), m.group(2), m.group(3)
    return prefix + " ".join(_scaled(p) for p in size.split()) + suffix


def scale_cylinder(m):
    prefix, radius, mid, length, suffix = m.groups()
    return f"{prefix}{_scaled(radius)}{mid}{_scaled(length)}{suffix}"


def scale_sphere_radius(m):
    prefix, radius, suffix = m.group(1), m.group(2), m.group(3)
    return f"{prefix}{_scaled(radius)}{suffix}"


def scale_inertia(m):
    # Inertia is mass * length^2, so it scales by model_scale^2 (mass is untouched).
    def attr(a):
        name, value = a.group(1), a.group(2)
        v = "0" if float(value) == 0 else f"${{model_scale*model_scale*{value}}}"
        return f'{name}="{v}"'
    return INERTIA_ATTR_RE.sub(attr, m.group(0))


def add_xacro_ns(m):
    attrs = m.group(1)
    if 'xmlns:xacro' in attrs:
        return m.group(0)
    return f'<robot xmlns:xacro="http://www.ros.org/wiki/xacro"{attrs}>\n  <xacro:include filename="properties.xacro"/>'


def main():
    ap = argparse.ArgumentParser(description="Raw LinkForge URDF export -> inmoov_i2.urdf.xacro")
    ap.add_argument("in_path", help="raw LinkForge .urdf export")
    ap.add_argument("out_path", help="output .urdf.xacro")
    ap.add_argument("--inertia", default=DEFAULT_INERTIA, help="mass/inertia estimate JSON (default: %(default)s)")
    ap.add_argument("--no-inertia", action="store_true", help="keep LinkForge's inertia (scaled by model_scale^2)")
    args = ap.parse_args()
    text = open(args.in_path, encoding="utf-8").read()

    text, n_paths = MESH_PATH_RE.subn(fix_mesh_path, text)
    text = ORIGIN_XYZ_RE.sub(scale_xyz, text)
    text, n_mesh = MESH_RE.subn(add_mesh_scale, text)
    text, n_box = BOX_SIZE_RE.subn(scale_box_size, text)
    text, n_cyl = CYLINDER_RE.subn(scale_cylinder, text)
    text, n_sph = SPHERE_RADIUS_RE.subn(scale_sphere_radius, text)
    text, n_inertia = INERTIA_RE.subn(scale_inertia, text)
    text = ROBOT_OPEN_RE.sub(add_xacro_ns, text, count=1)
    if not args.no_inertia:
        text, n_est, total = apply_inertia(text, load_estimate(args.inertia))

    open(args.out_path, "w", encoding="utf-8").write(text)
    print(
        f"mesh paths fixed: {n_paths}, meshes scaled: {n_mesh}, "
        f"box scaled: {n_box}, cylinder scaled: {n_cyl}, sphere scaled: {n_sph}, "
        f"inertia scaled: {n_inertia}"
    )
    if not args.no_inertia:
        print(f"inertia estimate applied to {n_est} links, total mass {total:.2f} kg ({args.inertia})")


if __name__ == "__main__":
    main()
