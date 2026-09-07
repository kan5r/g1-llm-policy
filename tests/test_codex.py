import sys

import numpy as np
import pytest

from g1_llm_policy.policies.codex import AppServer, CodexPolicy, RpcError


@pytest.fixture
def server_executable(tmp_path):
    script = tmp_path / "fake-codex"
    script.write_text(
        f"#!{sys.executable}\n"
        + """
import json,sys
for line in sys.stdin:
    req=json.loads(line)
    method=req.get("method")
    if "id" not in req: continue
    result={}
    if method=="model/list": result={"data":[{"model":"gpt-6-astra"}],"nextCursor":None}
    if method=="thread/start":
        result={"thread":{"id":"thread-1"},"model":req["params"]["model"]}
    if method=="turn/start":
        assert req["params"]["input"][2]["url"].startswith("data:image/png;base64,")
        assert "prefixItems" not in json.dumps(req["params"]["outputSchema"])
        action={"status":"move","left":None,"right":{"position":[0.4,-0.2,0.9],"quaternion":[1,0,0,0]},"left_open":None,"right_open":1.0,"note":"Reach"}
        # Notifications can precede the RPC response; client must buffer them.
        print(json.dumps({"method":"item/completed","params":{"threadId":"thread-1","turnId":"turn-1","item":{"type":"agentMessage","text":json.dumps(action)}}}),flush=True)
        result={"turn":{"id":"turn-1"}}
    print(json.dumps({"id":req["id"],"result":result}),flush=True)
    if method=="turn/start":
        print(json.dumps({"method":"turn/completed","params":{"threadId":"thread-1","turn":{"id":"turn-1","status":"completed"}}}),flush=True)
"""
    )
    script.chmod(0o755)
    return str(script)


def test_schema_image_and_out_of_order_events(server_executable):
    with AppServer(executable=server_executable, timeout=3) as server:
        policy = CodexPolicy(server)
        command = policy.act("Reach", {}, {"ego": np.zeros((8, 8, 3), dtype=np.uint8)})
        assert command.right.position == (0.4, -0.2, 0.9)
    assert server.process.poll() is not None


def test_model_not_silently_replaced(server_executable):
    with AppServer(executable=server_executable, timeout=3) as server:
        with pytest.raises(ValueError, match="model/list"):
            CodexPolicy(server, model="missing-model")


def test_dead_server_does_not_hang(tmp_path):
    script = tmp_path / "dead"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)
    with pytest.raises(RpcError, match="exited"):
        AppServer(executable=str(script), timeout=2)
