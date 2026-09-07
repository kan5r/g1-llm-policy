"""Quaternion extension to Inspect's pinned scalar tool and approval machinery."""

import math
from dataclasses import replace

import numpy as np
from inspect_robots.policy import PolicyInfo
from inspect_robots.types import Action
from inspect_robots_agent._tools import ToolResult, Toolset, build_toolset
from inspect_robots_agent.policy import LLMAgentPolicy

QUATS = (slice(3, 7), slice(11, 15))
SCALARS = np.array([0, 1, 2, 7, 8, 9, 10, 15])


def unit(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or abs(norm - 1) > 0.02:
        raise ValueError("Quaternion must have unit norm (w,x,y,z)")
    return q / norm


def angle(a, b):
    return 2 * math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1)))


def slerp(a, b, t):
    if np.dot(a, b) < 0:
        b = -b
    dot = float(np.clip(np.dot(a, b), 0, 1))
    if dot > 0.9995:
        q = (1 - t) * a + t * b
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


class QuaternionToolset(Toolset):
    def schemas(self):
        schemas = super().schemas()
        schemas[0]["function"]["description"] = (
            "Move to absolute world-frame TCP targets. Positions are metres; "
            "orientations are unit quaternions (qw,qx,qy,qz), not angles. "
            "Supply all four components of a hand's quaternion or omit all four. "
            "Omitted dimensions hold the measured current value. Positions/openings "
            "interpolate linearly; rotations follow shortest-path SLERP. "
            f"Angular speed limit is {math.degrees(self.angular_step * self._resolved_hz):g} degrees/s. "
            + self._bounds_text
        )
        return schemas

    def residual(self, target, observation):
        state = self._current_state(observation).copy()
        for group in QUATS:
            if np.dot(state[group], target[group]) < 0:
                state[group] *= -1
        aligned = replace(observation, state={**observation.state, self._state_key: state})
        return super().residual(target, aligned)

    def _move_absolute(self, values, vector, named_indices, current):
        current = current.copy()
        target = current.copy()
        target[named_indices] = vector[named_indices]
        if np.any(target[SCALARS] < self._low[SCALARS]) or np.any(
            target[SCALARS] > self._high[SCALARS]
        ):
            return ToolResult(error="target is outside the declared position/opening bounds")
        try:
            for group in QUATS:
                count = sum(i in named_indices for i in range(group.start, group.stop))
                if count not in (0, 4):
                    return ToolResult(
                        error="Supply all four quaternion components or omit all four"
                    )
                current[group] = unit(current[group])
                target[group] = unit(target[group])
                if np.dot(current[group], target[group]) < 0:
                    target[group] *= -1
        except ValueError as exc:
            return ToolResult(error=str(exc))
        ratios = list(abs(target[SCALARS] - current[SCALARS]) / self._step_limits[SCALARS])
        ratios.extend(angle(current[g], target[g]) / self.angular_step for g in QUATS)
        steps = max(1, math.ceil(max(ratios) / (1 - 1e-6)))
        if steps > self._max_steps:
            return self._cap_error()
        actions = []
        for i in range(1, steps + 1):
            t = i / steps
            point = current + t * (target - current)
            for group in QUATS:
                point[group] = slerp(current[group], target[group], t)
            actions.append(Action(point))
        if self._pre_check:
            error = self._pre_check(np.stack([a.data for a in actions]))
            if error:
                return ToolResult(error="pre-check rejected this motion: " + error)
        return self._success(actions, steps, target=actions[-1].data)


class QuaternionAgent(LLMAgentPolicy):
    def bind(self, info):
        # Reuse upstream scalar bounds/step-budget derivation without pretending
        # quaternions can be interpolated componentwise. The public contract is quat.
        scalar_space = replace(
            info.action_space, semantics=replace(info.action_space.semantics, rotation_repr="none")
        )
        base = build_toolset(
            scalar_space,
            info.observation_space,
            info.control_hz,
            self._max_speed_frac,
            images=self._images,
            pre_check=self._pre_check,
        )
        toolset = QuaternionToolset.__new__(QuaternionToolset)
        toolset.__dict__.update(base.__dict__)
        toolset.angular_step = min(math.pi * self._max_speed_frac / info.control_hz, math.pi * 0.05)
        self._toolset = toolset
        self._state_labels = toolset.state_labels()
        self._embodiment_name = info.name
        self._embodiment_docs = info.docs
        self.info = PolicyInfo(
            name="agent",
            action_space=info.action_space,
            observation_space=info.observation_space,
            control_hz=info.control_hz,
        )


class QuaternionApprover:
    def __init__(self, space):
        self.space = space

    def review(self, action, store):
        data = np.array(action.data, dtype=float, copy=True)
        if not np.all(np.isfinite(data)):
            raise ValueError("Non-finite action")
        for group in QUATS:
            data[group] = unit(data[group])
        data[SCALARS] = np.clip(data[SCALARS], self.space.low[SCALARS], self.space.high[SCALARS])
        previous = store.get("quaternion_previous")
        if previous is not None:
            limit = 0.05 * (self.space.high - self.space.low)
            data[SCALARS] = np.clip(
                data[SCALARS],
                previous[SCALARS] - limit[SCALARS],
                previous[SCALARS] + limit[SCALARS],
            )
            for group in QUATS:
                distance = angle(previous[group], data[group])
                data[group] = slerp(
                    previous[group], data[group], min(1, math.pi * 0.05 / max(distance, 1e-12))
                )
        store["quaternion_previous"] = data.copy()
        changed = not np.allclose(data, action.data, atol=1e-12, rtol=0)
        return replace(
            action, data=data, meta={**action.meta, **({"delta_clamped": True} if changed else {})}
        )
