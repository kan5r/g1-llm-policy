"""Fetch the official Dex1-1 G1 wrist/gripper assets at an immutable revision."""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = "unitreerobotics/unitree_ros"
COMMIT = "7d6075f7f58588b189b940130e3edab3c839b2df"
DEST = Path(__file__).resolve().parents[1] / "src/g1_llm_policy/assets/dex1"
URDF = "robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"


def main():
    base = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
    content = urllib.request.urlopen(base + URDF).read()
    root = ET.fromstring(content)
    selected = [
        link
        for link in root.findall("link")
        if "wrist" in link.get("name") or "dex1" in link.get("name")
    ]
    meshes = sorted({m.get("filename") for link in selected for m in link.findall(".//mesh")})
    files = {"g1_dex1.urdf": content, "LICENSE": urllib.request.urlopen(base + "LICENSE").read()}
    for path in meshes:
        files[path] = urllib.request.urlopen(base + "robots/g1_description/" + path).read()
    for path, data in files.items():
        target = DEST / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (DEST / "SOURCE.json").write_text(
        json.dumps(
            {
                "repository": REPO,
                "commit": COMMIT,
                "urdf": URDF,
                "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Fetched {len(files)} files")


if __name__ == "__main__":
    main()
