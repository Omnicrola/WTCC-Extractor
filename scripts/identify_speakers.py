"""
Utility: Identify speakers across source audio files.

For each audio file in SOURCE_AUDIO_DIR, diarizes the first 2 minutes and
attempts to match each speaker cluster to a known profile in SPEAKER_PROFILES_DIR.
Unknown speakers have a sample clip saved to UNKNOWN_SPEAKERS_DIR for manual
labeling and profile building.

Episodes where every speaker is matched are recorded in the database
(all_speakers_identified = 1) and skipped on subsequent runs.

Usage:
    python scripts/identify_speakers.py
    python scripts/identify_speakers.py --force   # re-check already-completed files
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import sqlite3
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)

import numpy as np
import torch
import soundfile as sf
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_distance
from config import (
    SOURCE_AUDIO_DIR,
    SQLITE_DB,
    SPEAKER_PROFILES_DIR,
    SPEAKER_SIMILARITY_THRESHOLD,
    UNKNOWN_SPEAKERS_DIR,
    HF_TOKEN,
    WHISPER_DEVICE,
)

SAMPLE_RATE    = 16_000
PROBE_SECONDS  = 120  # diarize only the first 2 minutes
MIN_CLIP_SECS    = 3    # ignore speaker segments shorter than this
MAX_CLIP_SECS    = 30   # cap saved clips at this length
UNKNOWN_CLIP_LIMIT = 10  # stop after saving this many unknown speaker clips

SOURCE_AUDIO_EXTS  = {".mp3", ".wav", ".m4a", ".ogg"}
PROFILE_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return sqlite3.connect(SQLITE_DB)


def is_already_identified(audio_file: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT all_speakers_identified FROM episodes WHERE audio_file = ?",
            (os.path.abspath(audio_file),)
        ).fetchone()
    return bool(row and row[0])


def mark_all_identified(audio_file: str):
    abs_path = os.path.abspath(audio_file)
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO episodes (audio_file, status, all_speakers_identified)
               VALUES (?, 'pending', 1)
               ON CONFLICT(audio_file) DO UPDATE SET all_speakers_identified = 1""",
            (abs_path,)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Speaker profile loading
# ---------------------------------------------------------------------------

def load_profiles(inference) -> dict[str, np.ndarray]:
    import whisperx

    groups: dict[str, list] = defaultdict(list)
    for fname in sorted(os.listdir(SPEAKER_PROFILES_DIR)):
        if os.path.splitext(fname)[1].lower() not in PROFILE_AUDIO_EXTS:
            continue
        name = fname.split("_")[0]
        groups[name].append(os.path.join(SPEAKER_PROFILES_DIR, fname))

    profiles: dict[str, np.ndarray] = {}
    for name, paths in groups.items():
        embeddings = []
        for path in paths:
            clip = whisperx.load_audio(path)
            waveform = torch.from_numpy(clip[None])
            emb = np.array(
                inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
            ).flatten()
            embeddings.append(emb)
        profiles[name] = np.mean(embeddings, axis=0)

    return profiles


def match_embedding(
    emb: np.ndarray,
    profiles: dict[str, np.ndarray],
    threshold: float,
) -> str | None:
    best_name, best_sim = None, -1.0
    for name, ref_emb in profiles.items():
        sim = 1.0 - cosine_distance(emb, ref_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name if best_sim >= threshold else None


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    audio_path: str,
    inference,
    profiles: dict[str, np.ndarray],
    device: str,
) -> tuple[bool, int]:
    """
    Diarize the first PROBE_SECONDS of audio_path, match speakers to profiles,
    and save clips for any that cannot be identified.
    Returns (all_identified, unknown_count).
    """
    import whisperx
    import whisperx.diarize

    basename = os.path.basename(audio_path)
    stem     = os.path.splitext(basename)[0]

    audio = whisperx.load_audio(audio_path)
    probe = audio[: PROBE_SECONDS * SAMPLE_RATE]

    print(f"  Diarizing first {PROBE_SECONDS // 60} min...")
    diarize_model = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=device)
    diarize_df    = diarize_model(probe)

    # Collect audio chunks per speaker
    speaker_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in diarize_df.itertuples():
        duration = row.end - row.start
        if duration < MIN_CLIP_SECS:
            continue
        s = int(row.start * SAMPLE_RATE)
        e = int(row.end   * SAMPLE_RATE)
        speaker_chunks[row.speaker].append(probe[s:e])

    if not speaker_chunks:
        print("  No usable speaker segments found.")
        return False, 0

    os.makedirs(UNKNOWN_SPEAKERS_DIR, exist_ok=True)

    all_identified = True
    unknown_count  = 0
    for speaker, chunks in sorted(speaker_chunks.items()):
        combined = np.concatenate(chunks)
        waveform = torch.from_numpy(combined[None])
        emb = np.array(
            inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        ).flatten()

        name = match_embedding(emb, profiles, SPEAKER_SIMILARITY_THRESHOLD)
        if name:
            print(f"  {speaker} → {name!r}")
        else:
            all_identified = False
            unknown_count += 1
            longest = max(chunks, key=len)
            clip    = longest[: MAX_CLIP_SECS * SAMPLE_RATE]
            out_path = os.path.join(UNKNOWN_SPEAKERS_DIR, f"{stem}_{speaker}.wav")
            sf.write(out_path, clip, SAMPLE_RATE)
            print(f"  {speaker} → unknown  (clip saved: {os.path.basename(out_path)})")

    return all_identified, unknown_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Identify speakers in source audio files")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-check files already marked as fully identified"
    )
    args = parser.parse_args()

    if not os.path.exists(SQLITE_DB):
        print(f"Database not found: {SQLITE_DB}")
        print("Run scripts/setup_db.py first.")
        sys.exit(1)

    if not SPEAKER_PROFILES_DIR or not os.path.isdir(SPEAKER_PROFILES_DIR):
        print(f"Speaker profiles directory not found: {SPEAKER_PROFILES_DIR}")
        print("Add speaker profile clips to that directory first.")
        sys.exit(1)

    files = sorted(
        os.path.join(SOURCE_AUDIO_DIR, f)
        for f in os.listdir(SOURCE_AUDIO_DIR)
        if os.path.splitext(f)[1].lower() in SOURCE_AUDIO_EXTS
    )
    if not files:
        print(f"No audio files found in {SOURCE_AUDIO_DIR}")
        sys.exit(1)

    device = _resolve_device(WHISPER_DEVICE)

    print(f"Found {len(files)} audio file(s). Loading embedding model...")
    from pyannote.audio import Model, Inference
    emb_model = Model.from_pretrained("pyannote/embedding", token=HF_TOKEN)
    inference = Inference(emb_model, window="whole")

    print("Loading speaker profiles...")
    profiles = load_profiles(inference)
    if not profiles:
        print("No speaker profiles found. Add clips to SPEAKER_PROFILES_DIR.")
        sys.exit(1)
    print(f"  Loaded {len(profiles)} profile(s): {', '.join(sorted(profiles))}")

    completed = skipped = total_unknown = 0
    for audio_path in files:
        basename = os.path.basename(audio_path)

        if not args.force and is_already_identified(audio_path):
            print(f"\n  [skip] {basename}  (all speakers already identified)")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {basename}")
        print(f"{'='*60}")

        all_identified, unknown_count = process_file(audio_path, inference, profiles, device)
        total_unknown += unknown_count

        if all_identified:
            mark_all_identified(audio_path)
            print("  All speakers identified — recorded in database.")
            completed += 1
        else:
            print(f"  Unknown speaker clips saved to: {UNKNOWN_SPEAKERS_DIR}")

        if total_unknown >= UNKNOWN_CLIP_LIMIT:
            print(f"\nReached limit of {UNKNOWN_CLIP_LIMIT} unknown speaker clips — stopping early.")
            break

    print(f"\n{'='*60}")
    print(f"Done.  Fully identified: {completed}  |  Skipped: {skipped}  |  Unknown clips saved: {total_unknown}")


if __name__ == "__main__":
    main()
