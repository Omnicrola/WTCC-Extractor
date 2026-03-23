"""
Step 2: Extract the game segment from the full episode using ffmpeg.
Cuts from the game intro timestamp for SEGMENT_WINDOW_SECONDS.

Usage:
    python scripts/02_extract_segment.py <episode_audio_file> <game_intro_timestamp_seconds>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from config import SEGMENTS_DIR, SEGMENT_WINDOW_SECONDS


def extract_segment(audio_file: str, start_seconds: float, duration: int = SEGMENT_WINDOW_SECONDS) -> str:
    """
    Extract a segment from audio_file starting at start_seconds for duration seconds.
    Saves to SEGMENTS_DIR and returns the output file path.
    """
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(audio_file))[0]
    out_file = os.path.join(SEGMENTS_DIR, f"{base}_segment.wav")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_seconds),
        "-i", audio_file,
        "-t", str(duration),
        "-ar", "16000",      # 16 kHz — optimal for Whisper
        "-ac", "1",          # mono
        "-c:a", "pcm_s16le", # uncompressed WAV for WhisperX
        out_file,
    ]

    print(f"Extracting segment: {start_seconds:.1f}s + {duration//60} min from {os.path.basename(audio_file)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr)
        sys.exit(1)

    print(f"Segment saved: {out_file}")
    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <episode_audio_file> <jingle_timestamp_seconds>")
        sys.exit(1)

    out = extract_segment(sys.argv[1], float(sys.argv[2]))
    print(f"Output: {out}")
