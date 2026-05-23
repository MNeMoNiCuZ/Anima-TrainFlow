#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"


def main() -> int:
    if not APP.exists():
        print(f"[ERROR] Missing app.py at: {APP}")
        return 1

    print("Starting Anima TrainFlow...")
    print()
    return subprocess.call([sys.executable, str(APP)] + sys.argv[1:], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
