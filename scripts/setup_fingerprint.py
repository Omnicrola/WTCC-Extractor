"""
Setup (one-time): Build the audfprint fingerprint database from the game intro WAV.
Run once. The resulting .db file is used by 01_find_game_intro.py for every episode.

Requires:
    git clone https://github.com/dpwe/audfprint
    pip install -r audfprint/requirements.txt

Usage:
    python scripts/setup_fingerprint.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from config import AUDFPRINT_PY, GAME_INTRO_WAV, FINGERPRINT_DB, AUDFPRINT_DENSITY, AUDFPRINT_SHIFTS


def create_fingerprint():
    if not os.path.exists(GAME_INTRO_WAV):
        print(f"ERROR: Jingle WAV not found: {GAME_INTRO_WAV}")
        sys.exit(1)

    if not os.path.exists(AUDFPRINT_PY):
        print(f"ERROR: audfprint.py not found at: {AUDFPRINT_PY}")
        print("Clone it with: git clone https://github.com/dpwe/audfprint")
        sys.exit(1)

    os.makedirs(os.path.dirname(FINGERPRINT_DB), exist_ok=True)

    cmd = [
        sys.executable, AUDFPRINT_PY,
        "new",
        "--dbase", FINGERPRINT_DB,
        "--density", str(AUDFPRINT_DENSITY),
        "--shifts", str(AUDFPRINT_SHIFTS),
        GAME_INTRO_WAV,
    ]

    print(f"Building fingerprint database from: {GAME_INTRO_WAV}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)

    print(f"Fingerprint database created: {FINGERPRINT_DB}")


if __name__ == "__main__":
    create_fingerprint()
