#!/usr/bin/env python3
"""Convert a raw LinkForge URDF export into description/inmoov_i2.urdf.xacro:

- meshes\\name.obj (Windows-style, as LinkForge writes them) -> package://inmoov_description/meshes/name.obj
- every <origin xyz>, <mesh>, and primitive collision geometry (<box size>,
  <cylinder radius/length>, <sphere radius>) scaled by a model_scale xacro
  property (LinkForge exports raw Blender scene units, not meters) — the
  same pattern the old Sentience-Robotics inmoov_urdf properties.xacro
  used. properties.xacro itself is untouched; model_scale is hand-maintained.

Usage: xacrify_scale.py <in.urdf> <out.urdf.xacro>

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
import re
import sys

MESH_PATH_RE = re.compile(r'(<mesh\b[^>]*\bfilename=")meshes\\([^"]+)(")')
ORIGIN_XYZ_RE = re.compile(r'(<origin\b[^>]*\bxyz=")([^"]+)(")')
MESH_RE = re.compile(r'(<mesh\b[^>]*\bfilename="[^"]+")\s*/>')
BOX_SIZE_RE = re.compile(r'(<box\b[^>]*\bsize=")([^"]+)(")')
CYLINDER_RE = re.compile(r'(<cylinder\b[^>]*\bradius=")([^"]+)("[^>]*\blength=")([^"]+)(")')
SPHERE_RADIUS_RE = re.compile(r'(<sphere\b[^>]*\bradius=")([^"]+)(")')
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


def add_xacro_ns(m):
    attrs = m.group(1)
    if 'xmlns:xacro' in attrs:
        return m.group(0)
    return f'<robot xmlns:xacro="http://www.ros.org/wiki/xacro"{attrs}>\n  <xacro:include filename="properties.xacro"/>'


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    text = open(in_path, encoding="utf-8").read()

    text, n_paths = MESH_PATH_RE.subn(fix_mesh_path, text)
    text = ORIGIN_XYZ_RE.sub(scale_xyz, text)
    text, n_mesh = MESH_RE.subn(add_mesh_scale, text)
    text, n_box = BOX_SIZE_RE.subn(scale_box_size, text)
    text, n_cyl = CYLINDER_RE.subn(scale_cylinder, text)
    text, n_sph = SPHERE_RADIUS_RE.subn(scale_sphere_radius, text)
    text = ROBOT_OPEN_RE.sub(add_xacro_ns, text, count=1)

    open(out_path, "w", encoding="utf-8").write(text)
    print(
        f"mesh paths fixed: {n_paths}, meshes scaled: {n_mesh}, "
        f"box scaled: {n_box}, cylinder scaled: {n_cyl}, sphere scaled: {n_sph}"
    )


if __name__ == "__main__":
    main()
