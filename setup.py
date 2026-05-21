#!/usr/bin/env python3
"""
Environment installer for Anima TrainFlow.

Uses the currently active Python interpreter to:
1) install root dependencies with uv
2) install PyTorch CUDA 12.8 wheels with uv
3) install sd-scripts dependencies with uv
4) install sd-scripts in editable mode with uv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SD_SCRIPTS_DIR = ROOT / "training" / "sd-scripts"
ROOT_REQUIREMENTS = ROOT / "requirements.txt"

SD_REQUIREMENT_CANDIDATES = (
    "requirements.txt",
    "requirements_windows.txt",
    "requirements-win.txt",
)
SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
SD_SCRIPTS_BRANCH = "main"
TORCH_VERSION = "2.11.0+cu128"
TORCHVISION_VERSION = "0.26.0+cu128"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_sd_requirements(sd_dir: Path) -> Path | None:
    for name in SD_REQUIREMENT_CANDIDATES:
        candidate = sd_dir / name
        if candidate.exists():
            return candidate
    return None


def ensure_sd_scripts() -> None:
    training_dir = ROOT / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    if SD_SCRIPTS_DIR.exists() and any(SD_SCRIPTS_DIR.iterdir()):
        print(f"Using existing sd-scripts at: {SD_SCRIPTS_DIR}")
        return

    if not SD_SCRIPTS_DIR.exists():
        print(f"Cloning sd-scripts from {SD_SCRIPTS_REPO}...")
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                SD_SCRIPTS_BRANCH,
                SD_SCRIPTS_REPO,
                str(SD_SCRIPTS_DIR),
            ]
        )
        return

    # Existing but empty directory: initialize and fetch into it.
    print(f"Initializing sd-scripts in existing directory: {SD_SCRIPTS_DIR}")
    run(["git", "init"], cwd=SD_SCRIPTS_DIR)
    try:
        run(["git", "remote", "add", "origin", SD_SCRIPTS_REPO], cwd=SD_SCRIPTS_DIR)
    except subprocess.CalledProcessError:
        run(
            ["git", "remote", "set-url", "origin", SD_SCRIPTS_REPO],
            cwd=SD_SCRIPTS_DIR,
        )
    run(
        ["git", "fetch", "--depth", "1", "origin", SD_SCRIPTS_BRANCH],
        cwd=SD_SCRIPTS_DIR,
    )
    run(
        ["git", "checkout", "-B", SD_SCRIPTS_BRANCH, "FETCH_HEAD"],
        cwd=SD_SCRIPTS_DIR,
    )


def main() -> int:
    if not ROOT_REQUIREMENTS.exists():
        print(f"[ERROR] Missing requirements file: {ROOT_REQUIREMENTS}")
        return 1

    ensure_sd_scripts()

    py = sys.executable
    print(f"Using Python: {py}")
    run([py, "-m", "uv", "pip", "install", "-r", str(ROOT_REQUIREMENTS)])

    # Force CUDA wheels so a previously installed CPU-only torch does not remain.
    run(
        [
            py,
            "-m",
            "uv",
            "pip",
            "install",
            "--reinstall",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
    )

    sd_requirements = find_sd_requirements(SD_SCRIPTS_DIR)
    if sd_requirements is None:
        print(
            "[ERROR] Could not find sd-scripts requirements file. "
            f"Tried: {', '.join(SD_REQUIREMENT_CANDIDATES)} in {SD_SCRIPTS_DIR}"
        )
        return 1

    run(
        [py, "-m", "uv", "pip", "install", "-r", str(sd_requirements)],
        cwd=SD_SCRIPTS_DIR,
    )
    run([py, "-m", "uv", "pip", "install", "-e", str(SD_SCRIPTS_DIR)])

    print("Installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
