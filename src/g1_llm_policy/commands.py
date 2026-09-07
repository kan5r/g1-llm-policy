"""One absolute, world-frame action shared by all policy transports."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class Pose(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = Field(description="Unit quaternion w,x,y,z")

    @field_validator("quaternion")
    @classmethod
    def unit_quaternion(cls, value):
        norm = np.linalg.norm(value)
        if abs(norm - 1) > 0.02:
            raise ValueError("quaternion must have unit norm (w,x,y,z)")
        return tuple(np.asarray(value) / norm)


class Command(BaseModel):
    _waypoints: list = PrivateAttr(default_factory=list)
    _control_hz: float = PrivateAttr(default=10.0)
    _action_chunk: object = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    status: Literal["move", "done", "give_up"]
    left: Pose | None
    right: Pose | None
    left_open: float | None = Field(ge=0, le=1)
    right_open: float | None = Field(ge=0, le=1)
    note: str = Field(min_length=1, max_length=1500)

    @classmethod
    def hold(cls, note="Hold current target"):
        return cls(status="move", left=None, right=None, left_open=None, right_open=None, note=note)


def output_schema():
    """Convert homogeneous tuple schemas to the Structured Outputs array subset."""
    schema = Command.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            if "prefixItems" in node:
                items = node.pop("prefixItems")
                node["items"] = items[0]
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return schema
