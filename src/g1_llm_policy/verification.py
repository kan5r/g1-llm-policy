"""Independent simulation evidence for pickup, never an LLM self-report."""

import re

import mujoco
import numpy as np


def resolve_task(task, instruction):
    if task != "auto":
        return task
    return (
        "pickup"
        if re.search(r"pick[ -]?up|lift|持ち上げ|拾|掴|つか", instruction, re.I)
        else "none"
    )


class PickupMonitor:
    clearance_required = 0.02
    hold_required = 0.5
    max_speed = 0.15

    def __init__(self):
        self.held_s = 0.0
        self.side = None
        self.clearance = 0.0
        self.speed = 0.0
        self.contacts = {"left": [], "right": []}

    def update(self, clearance, speed, contacts, dt):
        self.clearance, self.speed, self.contacts = clearance, speed, contacts
        side = next((s for s, regions in contacts.items() if len(set(regions)) >= 2), None)
        candidate = clearance >= self.clearance_required and speed <= self.max_speed and side
        if candidate:
            self.held_s = self.held_s + dt if self.side == side else dt
        else:
            self.held_s = 0.0
        self.side = side

    def sample(self, model, data):
        cube = model.geom("cube_geom").id
        table = model.geom("table").id
        # Support distance along world Z, including tilted cylinders.
        rotation = data.geom_xmat[cube].reshape(3, 3)
        size = model.geom_size[cube]
        if model.geom_type[cube] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis_z = float(np.clip(rotation[2, 2], -1, 1))
            extent = size[1] * abs(axis_z) + size[0] * np.sqrt(max(0, 1 - axis_z**2))
        else:
            extent = np.abs(rotation[2]) @ size
        bottom = data.geom_xpos[cube, 2] - extent
        top = data.geom_xpos[table, 2] + model.geom_size[table, 2]
        adr = model.joint("cube_free").dofadr[0]
        speed = float(np.linalg.norm(data.qvel[adr : adr + 3]))
        contacts = {"left": set(), "right": set()}
        force = np.zeros(6)
        for i, contact in enumerate(data.contact):
            if cube not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == cube else contact.geom1
            body = model.body(model.geom_bodyid[other]).name
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] <= 0.01:
                continue
            for side in contacts:
                if body.startswith(side + "_hand_"):
                    # Two phalanges of the same finger count as one region.
                    contacts[side].add(body.split("_hand_")[1].split("_")[0])
                elif body.startswith(side + "_dex1_finger_link_"):
                    contacts[side].add("jaw_" + body.rsplit("_", 1)[1])
                elif body == side + "_wrist_yaw_link":
                    contacts[side].add("palm")
        self.update(
            float(bottom - top),
            speed,
            {s: sorted(v) for s, v in contacts.items()},
            model.opt.timestep,
        )

    def report(self):
        success = self.held_s >= self.hold_required
        reasons = []
        if self.clearance < self.clearance_required:
            reasons.append("object has not cleared the table by 2 cm")
        if not self.side:
            reasons.append(
                "no hand has force-bearing contact through two distinct fingers/palm regions"
            )
        if self.speed > self.max_speed:
            reasons.append("object is moving too fast to count as stable holding")
        if not success:
            reasons.append("stable holding has not lasted 0.5 simulation seconds")
        return {
            "task": "pickup",
            "source": "mujoco_physics",
            "success": success,
            "clearance_m": self.clearance,
            "object_speed_m_s": self.speed,
            "contact_regions": self.contacts,
            "stable_hold_s": self.held_s,
            "reason": "; ".join(reasons) if reasons else "verified lifted and held",
        }
