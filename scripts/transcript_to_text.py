"""
Utility: Convert a WhisperX JSON transcript to a readable text file.

Outputs one line per speaker turn:
    [MM:SS] SpeakerName: sentence text

Usage:
    python scripts/transcript_to_text.py <transcript.json>
    python scripts/transcript_to_text.py <transcript.json> -o output.txt
"""

import sys, os
import json
import argparse


def format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def segments_to_turns(segments: list[dict]) -> list[tuple[str, float, str]]:
    """
    Collapse consecutive words with the same speaker into speaker turns.
    Returns list of (speaker, start_time, text).
    Falls back to segment-level text when word-level speaker data is absent.
    """
    turns: list[tuple[str, float, str]] = []

    for seg in segments:
        words = seg.get("words", [])
        seg_text = seg.get("text", "").strip()

        if not words:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0.0)
            if seg_text:
                turns.append((speaker, start, seg_text))
            continue

        current_speaker = None
        current_start = 0.0
        current_words: list[str] = []

        for w in words:
            speaker = w.get("speaker") or seg.get("speaker") or "UNKNOWN"
            word_text = w.get("word", "")
            if not word_text:
                continue

            if speaker != current_speaker:
                if current_words and current_speaker is not None:
                    turns.append((current_speaker, current_start, " ".join(current_words)))
                current_speaker = speaker
                current_start = w.get("start", seg.get("start", 0.0))
                current_words = [word_text]
            else:
                current_words.append(word_text)

        if current_words and current_speaker is not None:
            turns.append((current_speaker, current_start, " ".join(current_words)))

    return turns


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No segments found in transcript.")
        return

    turns = segments_to_turns(segments)

    lines = []
    for speaker, start, text in turns:
        lines.append(f"[{format_timestamp(start)}] {speaker}: {text.strip()}")

    output = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
        f.write("\n")

    print(f"Wrote {len(lines)} speaker turns to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WhisperX JSON transcript to readable text.")
    parser.add_argument("input", help="Path to the transcript JSON file")
    parser.add_argument("-o", "--output", help="Output text file path (default: same name as input with .txt)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = args.output or os.path.splitext(input_path)[0] + ".txt"
    convert(input_path, output_path)
