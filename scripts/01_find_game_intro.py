"""
Step 1: Use audfprint to locate the WTCC game intro within a podcast episode.
Returns the timestamp (in seconds) where the game intro begins.

Usage:
    python scripts/01_find_game_intro.py <episode_audio_file>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import re
from config import AUDFPRINT_PY, FINGERPRINT_DB


def find_game_intro_timestamp(audio_file: str) -> float | None:
    """
    Run audfprint match against the episode audio.
    Returns the start time in seconds, or None if not found.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if not os.path.exists(FINGERPRINT_DB):
        raise FileNotFoundError(
            f"Fingerprint database not found: {FINGERPRINT_DB}\n"
            "Run scripts/setup_fingerprint.py first."
        )

    cmd = [
        sys.executable, AUDFPRINT_PY,
        "match",
        "--dbase", FINGERPRINT_DB,
        "--find-time-range",
        "--sortbytime",
        audio_file,
    ]

    print(f"Scanning for game intro in: {os.path.basename(audio_file)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0 and "Matched" not in output:
        print("audfprint output:\n", output)
        return None

    return _parse_timestamp(output, audio_file)


def _parse_timestamp(output: str, audio_file: str) -> float | None:
    """
    Parse audfprint match output to extract the match start time.

    audfprint prints lines like:
        Matched  query.wav  as  ref.wav  at  t=45.23
    or with --find-time-range:
        Matched  query.wav  ...  start  45.23  ...
    or:
        Rank   1: ... offset=  45.23 ...

    We try multiple patterns to be robust across audfprint versions.
    """
    print("audfprint output:\n", output)

    # Pattern 1: "start  45.23" (--find-time-range flag output)
    m = re.search(r'start\s+([\d.]+)', output)
    if m:
        return float(m.group(1))

    # Pattern 2: "at  t=45.23" or "at  45.23"
    m = re.search(r'\bat\s+(?:t=)?([\d.]+)', output)
    if m:
        return float(m.group(1))

    # Pattern 3: "offset=  45.23" or "offset=45.23"
    m = re.search(r'offset\s*=\s*([\d.]+)', output)
    if m:
        return float(m.group(1))

    # Pattern 4: "t_offset  45.23"
    m = re.search(r't_offset\s+([\d.]+)', output)
    if m:
        return float(m.group(1))

    if "Matched" in output:
        print("WARNING: Match found but could not parse timestamp from output.")
        print("Please inspect the output above and update _parse_timestamp() accordingly.")
    else:
        print("No match found for game intro in this episode.")

    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <episode_audio_file>")
        sys.exit(1)

    ts = find_game_intro_timestamp(sys.argv[1])
    if ts is not None:
        print(f"\nGame intro found at: {ts:.2f} seconds ({ts/60:.1f} minutes)")
    else:
        print("\nGame intro not found.")
        sys.exit(1)
