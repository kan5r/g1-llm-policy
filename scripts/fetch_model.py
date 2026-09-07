"""Vendor only the G1 asset directory at a recorded Menagerie commit."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = "google-deepmind/mujoco_menagerie"
COMMIT = "ac6b2b09983786f3036cab1000221017fa2193b4"
DEST = Path(__file__).resolve().parents[1] / "src/g1_llm_policy/assets/g1"


def main():
    tree = json.load(urlopen(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    files = {}
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob" or not path.startswith("unitree_g1/"):
            continue
        relative = path.removeprefix("unitree_g1/")
        if not (
            relative.startswith("assets/")
            or relative in ("g1_with_hands.xml", "LICENSE", "README.md")
        ):
            continue
        content = urlopen(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}").read()
        output = DEST / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        files[relative] = hashlib.sha256(content).hexdigest()
    (DEST / "SOURCE.json").write_text(
        json.dumps({"repository": REPO, "commit": COMMIT, "sha256": files}, indent=2) + "\n"
    )
    print(f"Vendored {len(files)} files at {COMMIT}")


if __name__ == "__main__":
    main()
