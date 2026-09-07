"""Source-tree launcher, also works when the host disables editable .pth files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from g1_llm_policy.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
