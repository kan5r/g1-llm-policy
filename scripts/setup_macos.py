"""Make uv's libpython discoverable to mjpython, without editing installed tools."""

import sys
from pathlib import Path

if sys.platform == "darwin":
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run inside the project venv: uv run python scripts/setup_macos.py")
    name = f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib"
    source = Path(sys.base_prefix) / "lib" / name
    if not source.exists():
        raise SystemExit(f"Cannot locate {source}; check your Python distribution")
    for directory in (Path(sys.prefix), Path(sys.prefix) / "lib"):
        target = directory / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise SystemExit(f"Refusing to replace existing {target}")
        else:
            target.symlink_to(source)
        print(f"{target} -> {source}")
