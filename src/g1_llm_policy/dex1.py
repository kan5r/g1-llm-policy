"""Official mode-15 wrist + Dex1-1 subtrees, attached to the existing G1 elbows."""

import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS = Path(__file__).parent / "assets/dex1"
# Symmetric prismatic coordinates from the official model. 1 = open.
CLOSED = -0.02
OPEN = 0.0245
# Jaw contact area spans wrist-local x ~= .1206..1848 m.
TCP_X = 0.155


def apply_dex1(root):
    urdf = ET.parse(ASSETS / "g1_dex1.urdf").getroot()
    links = {x.get("name"): x for x in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    asset = root.find("asset")
    actuators = root.find("actuator")
    for actuator in list(actuators):
        if any(s in actuator.get("joint", "") for s in ("wrist_", "hand_")):
            actuators.remove(actuator)
    registered = set()

    def add_joint(parent, joint):
        name = joint.find("child").get("link")
        origin = joint.find("origin")
        body = ET.SubElement(
            parent,
            "body",
            name=name,
            pos=origin.get("xyz", "0 0 0"),
            euler=origin.get("rpy", "0 0 0"),
        )
        link = links[name]
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        origin_i = inertial.find("origin")
        if origin_i.get("rpy", "0 0 0") != "0 0 0":
            raise ValueError("Nonzero URDF inertia rotation needs explicit conversion")
        ET.SubElement(
            body,
            "inertial",
            pos=origin_i.get("xyz"),
            mass=inertial.find("mass").get("value"),
            fullinertia=" ".join(
                inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
        )
        if joint.get("type") != "fixed":
            limit = joint.find("limit")
            is_finger = joint.get("type") == "prismatic"
            ET.SubElement(
                body,
                "joint",
                name=joint.get("name"),
                type="slide" if is_finger else "hinge",
                axis=joint.find("axis").get("xyz"),
                range=f"{limit.get('lower')} {limit.get('upper')}",
                actuatorfrcrange=f"-{limit.get('effort')} {limit.get('effort')}",
                frictionloss="0.1" if is_finger else "0.3",
            )
            ET.SubElement(
                actuators,
                "position",
                name=joint.get("name"),
                joint=joint.get("name"),
                **{"class": "g1", "kp": "1000" if is_finger else "120", "dampratio": "1"},
            )
        for kind in ("visual", "collision"):
            for geom in link.findall(kind):
                mesh = geom.find("geometry/mesh")
                if mesh is None:
                    raise ValueError("Expected mesh geometry in official Dex1 wrist")
                filename = mesh.get("filename")
                meshname = "dex1_" + Path(filename).stem
                if meshname not in registered:
                    ET.SubElement(asset, "mesh", name=meshname, file=str(ASSETS / filename))
                    registered.add(meshname)
                origin_g = geom.find("origin")
                kwargs = {
                    "class": kind,
                    "mesh": meshname,
                    "pos": origin_g.get("xyz", "0 0 0"),
                    "euler": origin_g.get("rpy", "0 0 0"),
                }
                if kind == "visual":
                    kwargs["material"] = "black" if "dex1" in name else "metal"
                else:
                    kwargs["friction"] = "1 .01 .001"
                ET.SubElement(body, "geom", **kwargs)
        for child in children.get(name, []):
            add_joint(body, child)

    for side in ("left", "right"):
        elbow = root.find(f".//body[@name='{side}_elbow_link']")
        elbow.remove(elbow.find(f"body[@name='{side}_wrist_roll_link']"))
        add_joint(
            elbow,
            next(
                j
                for j in children[f"{side}_elbow_link"]
                if j.get("name") == f"{side}_wrist_roll_joint"
            ),
        )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for side in ("left", "right"):
        ET.SubElement(
            equality,
            "joint",
            name=f"{side}_dex1_sync",
            joint1=f"{side}_dex1_finger_joint_1",
            joint2=f"{side}_dex1_finger_joint_2",
            polycoef="0 1 0 0 0",
        )
