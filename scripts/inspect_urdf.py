"""Print the URDF's links, joints, limits, and mesh paths, plus the role mapping Body will use.

Run this first. ROLES/POSES in app/body/behaviors.py are guesses until checked against this output.
"""
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.body.behaviors import ROLES, load_joints  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "robot/dummy_lamp_6dof.urdf"
if not os.path.exists(path):
    sys.exit(f"URDF not found at {path}. Put the supplied robot/ folder at the repo root.")
root = ET.parse(path).getroot()
base = os.path.dirname(os.path.abspath(path))

print(f"robot: {root.get('name')}\n\nLINKS")
for link in root.findall("link"):
    meshes = [m.get("filename") for m in link.iter("mesh")]
    for m in meshes:
        rel = m.replace("package://", "").split("/", 1)[-1] if m.startswith("package://") else m
        ok = os.path.exists(os.path.join(base, rel)) or os.path.exists(os.path.join(base, m))
        print(f"  {link.get('name'):24s} mesh={m} {'OK' if ok else 'MISSING'}")
    if not meshes:
        print(f"  {link.get('name')}")

print("\nJOINTS")
for j in root.findall("joint"):
    lim = j.find("limit")
    axis = j.find("axis")
    print(f"  {j.get('name'):24s} {j.get('type'):10s} {j.find('parent').get('link')} -> {j.find('child').get('link')}"
          f"  axis={axis.get('xyz') if axis is not None else '-'}"
          f"  limit={(lim.get('lower'), lim.get('upper')) if lim is not None else '-'}")

print("\nROLE MAPPING used by app/body/behaviors.py (movable joints in file order)")
movable = load_joints(path)
for role, (name, lo, hi) in zip(ROLES, movable, strict=False):
    print(f"  {role:12s} -> {name}  [{lo:+.2f}, {hi:+.2f}]")
if len(movable) != len(ROLES):
    print(f"  NOTE: {len(movable)} movable joints vs {len(ROLES)} roles; extra joints stay at zero.")
