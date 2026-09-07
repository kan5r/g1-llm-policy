"""Use Inspect's actual agent; adapt only its LLM transport to Codex app-server."""

import json
import uuid

import httpx
from inspect_robots.scene import Scene

from .. import inspect_contract as contract
from ..commands import Command
from ..quaternion_tools import QuaternionAgent
from .codex import CodexPolicy


class CodexTransport(httpx.BaseTransport):
    def __init__(self, server, model, effort):
        self.server, self.model, self.effort = server, model, effort

    def handle_request(self, request):
        body = json.loads(request.content)
        messages, tools = body["messages"], body["tools"]
        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        instructions += (
            "\n\nTransport: represent exactly one of the following tool calls as JSON "
            "with name and arguments (arguments is a JSON-encoded string). "
            "Only these robot tools are available. Do not use shell, files, network or computer tools.\n"
            + json.dumps(tools)
        )
        inputs = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                inputs.append({"type": "text", "text": f"{msg['role']}:", "text_elements": []})
                for part in content:
                    if part["type"] == "image_url":
                        inputs.append({"type": "image", "url": part["image_url"]["url"]})
                    elif part["type"] == "text":
                        inputs.append({"type": "text", "text": part["text"], "text_elements": []})
            else:
                inputs.append({"type": "text", "text": json.dumps(msg), "text_elements": []})
        # A fresh ephemeral session receives exactly Inspect's selected history,
        # avoiding extra server-side history that would defeat image_horizon.
        session = CodexPolicy(self.server, self.model, self.effort, instructions=instructions)
        response = session.generate(
            inputs,
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [t["function"]["name"] for t in tools]},
                    "arguments": {"type": "string"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": str(uuid.uuid4()),
                                    "type": "function",
                                    "function": response,
                                }
                            ],
                        }
                    }
                ]
            },
        )


class InspectPolicy:
    def __init__(
        self,
        server,
        model="gpt-6-astra",
        effort="low",
        *,
        hand="dex1",
        cameras=("ego",),
        max_llm_calls=100,
        images="always",
        max_speed_frac=0.1,
    ):
        self.agent = QuaternionAgent(
            model=model,
            base_url="http://codex.invalid",
            env={},
            wire="chat",
            wire_capture=False,
            transport=CodexTransport(server, model, effort),
            effort=effort,
            max_llm_calls=max_llm_calls,
            max_speed_frac=max_speed_frac,
            images=images,
            pre_check=contract.pre_check,
        )
        self.agent.bind(contract.info(cameras, hand))
        self.started = False
        self.step = 0
        self.last_chunk = None

    def act_observation(self, obs):
        if not self.started:
            self.agent.reset(Scene(id="tabletop", instruction=obs.instruction))
            self.started = True
        self.last_chunk = self.agent.act(obs)
        return self.last_chunk

    def act(self, instruction, state, images):
        chunk = self.act_observation(contract.observation(state, images, instruction, self.step))
        self.step += len(chunk.actions)
        stop = chunk.actions[0].meta
        if stop.get("request_stop"):
            return Command.hold(stop.get("stop_detail") or stop["stop_reason"]).model_copy(
                update={"status": stop["stop_reason"]}
            )
        # Inspect stores the move note in the transcript, not in waypoint actions.
        note = "Inspect motion"
        for message in reversed(self.agent.transcript() or []):
            calls = message.get("tool_calls", [])
            if calls:
                for call in calls:
                    args = json.loads(call["function"]["arguments"])
                    if "targets" in args:
                        note = args["note"]
                        break
                if note != "Inspect motion":
                    break
        command = contract.decode(chunk.actions[-1].data, note)
        command._waypoints = [contract.decode(a.data, note) for a in chunk.actions]
        command._control_hz = chunk.control_hz
        command._action_chunk = chunk
        return command

    def transcript(self):
        return self.agent.transcript()

    def close(self):
        self.agent._client.close()
