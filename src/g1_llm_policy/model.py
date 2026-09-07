"""Apply scene-specific changes in memory; vendored Menagerie XML stays intact."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets/g1"


def build_model(object_shape="box", hand="dex1"):
    if hand not in ("dex1", "dex3"):
        raise ValueError("hand must be dex1 or dex3")
    if object_shape not in ("cylinder", "box"):
        raise ValueError("object_shape must be cylinder or box")
    root = ET.parse(ASSETS / "g1_with_hands.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "assets"))
    root.find("option").set("timestep", "0.002")
    if hand == "dex1":
        from .dex1 import apply_dex1

        apply_dex1(root)
    # Remove floating base and leg/waist joints: this is an upper-body experiment.
    for parent in root.iter():
        for child in list(parent):
            name = child.get("name", "")
            if child.tag == "freejoint" or (
                child.tag == "joint"
                and any(s in name for s in ("hip_", "knee_", "ankle_", "waist_"))
            ):
                parent.remove(child)
    for geom in root.findall(".//geom"):
        if geom.get("class") == "visual" and "hand_" in geom.get("mesh", ""):
            geom.set("material", "black")
    joints = {e.get("name") for e in root.findall(".//worldbody//joint")}
    actuators = root.find("actuator")
    for actuator in list(actuators):
        if actuator.get("joint") not in joints:
            actuators.remove(actuator)
        else:
            is_hand = "hand" in actuator.get("joint")
            actuator.set(
                "kp",
                "1000" if "dex1_finger" in actuator.get("joint") else "10" if is_hand else "120",
            )
            actuator.set("dampratio", "1")
    for key in list(root.findall("keyframe")):
        root.remove(key)
    for side in ("left", "right"):
        palm = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(
            palm,
            "site",
            name=f"{side}_palm",
            pos="0.155 0 0" if hand == "dex1" else "0.08 0 0",
            size="0.008",
            rgba="0 0.8 0.8 1",
        )
    torso = root.find(".//body[@name='torso_link']")
    # Approximate head-mounted RGB camera, pitched 40 degrees down; not hardware-calibrated.
    ET.SubElement(
        torso, "camera", name="ego", pos="0.065 0 0.41", xyaxes="0 -1 0 0.6427876 0 0.7660444", fovy="80"
    )
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1", rgba=".22 .25 .3 1")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos=".40 0 .80",
        size=".28 .5 .035",
        rgba=".55 .38 .22 1",
    )
    # VIRAL simple/bottle.usd: radius .5 * .06; height 1 * .15 metres.
    half_height = 0.075 if object_shape == "cylinder" else 0.025
    object_x = 0.40 if object_shape == "cylinder" else 0.24
    object_y = -0.06 if object_shape == "cylinder" else 0.0
    cube = ET.SubElement(
        world, "body", name="cube", pos=f"{object_x} {object_y} {0.835 + half_height + 0.005}"
    )
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(
        cube,
        "geom",
        name="cube_geom",
        type=object_shape,
        size=".03 .075" if object_shape == "cylinder" else ".025 .025 .025",
        mass=".08",
        rgba=".85 .08 .07 1",
        friction="1 .01 .001",
    )
    for name, pos, target in [
        ("overview", [1.8, -1.7, 1.55], [0.25, 0, 0.8]),
        ("front", [1.8, 0, 1.2], [0.3, 0, 0.9]),
    ]:
        z = np.asarray(pos) - target
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ET.SubElement(
            world,
            "camera",
            name=name,
            pos=" ".join(map(str, pos)),
            xyaxes=" ".join(map(str, [*x, *y])),
            fovy="48",
        )
    ET.SubElement(world, "light", pos="1 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
