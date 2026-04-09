"""
WTCC Extraction Pipeline — Orchestrator
Processes one or more podcast episodes end-to-end:
  1. Find WTCC game intro via audfprint fingerprinting
  2. Extract game segment audio with ffmpeg
  3. Transcribe with WhisperX + diarization
  4. Extract structured game data with Qwen 2.5 14B (Ollama)
  5. Store results in SQLite

Usage:
    # Process a single episode
    python run_pipeline.py source_audio/episode.mp3

    # Process all .mp3/.wav files in source_audio/
    python run_pipeline.py --all

    # Re-process even if already in the database
    python run_pipeline.py source_audio/episode.mp3 --force
"""

import argparse
import glob
import os
import sqlite3
import sys
import json
from contextlib import contextmanager
from datetime import datetime, timezone

# Add scripts/ to sys.path so log_utils (and importlib-loaded scripts) can find it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from config import SOURCE_AUDIO_DIR, SQLITE_DB, SEGMENTS_DIR, TRANSCRIPTS_DIR, LOGS_DIR
from log_utils import setup_logger

logger = setup_logger("run_pipeline", LOGS_DIR)

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """Open a SQLite connection, commit on success, rollback on error, always close."""
    conn = sqlite3.connect(SQLITE_DB)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def episode_status(audio_file: str) -> tuple[str | None, bool]:
    """Return (status, game_intro_found) for an episode, or (None, True) if not yet recorded."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, game_intro_found FROM episodes WHERE audio_file = ?",
            (os.path.abspath(audio_file),)
        ).fetchone()
    if row:
        return row[0], bool(row[1])
    return None, True


def upsert_episode(audio_file: str) -> int:
    """Insert episode row if it doesn't exist; return its id."""
    abs_path = os.path.abspath(audio_file)
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO episodes (audio_file, status) VALUES (?, 'pending')",
            (abs_path,)
        )
        row = conn.execute(
            "SELECT id FROM episodes WHERE audio_file = ?", (abs_path,)
        ).fetchone()
    return row[0]


def update_episode(episode_id: int, **kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [episode_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE episodes SET {sets} WHERE id = ?", values)


def store_game_data(episode_id: int, game_data, raw_json: str, transcript_path: str,
                    round_timestamps: list):
    """Store extracted game data into game_rounds and clues tables."""
    with get_conn() as conn:
        # Remove any previous results for this episode (idempotent re-runs)
        old_rounds = conn.execute(
            "SELECT id FROM game_rounds WHERE episode_id = ?", (episode_id,)
        ).fetchall()
        for (rid,) in old_rounds:
            conn.execute("DELETE FROM clues WHERE round_id = ?", (rid,))
        conn.execute("DELETE FROM game_rounds WHERE episode_id = ?", (episode_id,))

        # Read raw transcript text for storage
        raw_transcript = ""
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                segments = transcript_data.get("segments", [])
                raw_transcript = " ".join(s.get("text", "") for s in segments)

        for round_data, round_ts in zip(game_data.rounds, round_timestamps):
            cur = conn.execute(
                """INSERT INTO game_rounds
                   (episode_id, answer, submitted_by, raw_json, raw_transcript, round_start_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (episode_id, round_data.answer, round_data.submitted_by,
                 raw_json, raw_transcript, round_ts)
            )
            round_id = cur.lastrowid

            for i, clue in enumerate(round_data.clues):
                conn.execute(
                    "INSERT INTO clues (round_id, clue_order, clue_text) VALUES (?, ?, ?)",
                    (round_id, i + 1, clue)
                )

        # Update episode metadata from extracted data
        conn.execute(
            """UPDATE episodes
               SET episode_title = COALESCE(NULLIF(?, ''), episode_title),
                   release_date  = COALESCE(NULLIF(?, ''), release_date)
               WHERE id = ?""",
            (game_data.episode_title, game_data.release_date, episode_id)
        )


# ---------------------------------------------------------------------------
# Pipeline steps — loaded via importlib because numeric filename prefixes
# are invalid Python identifiers and can't be imported directly.
# ---------------------------------------------------------------------------

def _load_script(name: str, module_alias: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        module_alias,
        os.path.join(SCRIPTS_DIR, name),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_find_game_intro(audio_file: str) -> float | None:
    mod = _load_script("01_find_game_intro.py", "find_game_intro")
    return mod.find_game_intro_timestamp(audio_file)


def step_extract_segment(audio_file: str, timestamp: float) -> str:
    mod = _load_script("02_extract_segment.py", "extract_segment")
    return mod.extract_segment(audio_file, timestamp)


def step_transcribe(segment_wav: str) -> str:
    mod = _load_script("03_transcribe.py", "transcribe")
    return mod.transcribe_segment(segment_wav)


def step_extract_game_data(transcript_json: str, game_intro_offset: float = 0.0):
    # Register transcribe module under a stable name so extract_game_data can find it
    transcribe_mod = _load_script("03_transcribe.py", "transcribe")
    sys.modules["transcribe_mod"] = transcribe_mod

    mod = _load_script("04_extract_game_data.py", "extract_game_data")
    game_data, raw_json = mod.extract_game_data(transcript_json)
    round_timestamps = mod.find_round_timestamps(transcript_json, game_data.rounds, game_intro_offset)
    return game_data, raw_json, round_timestamps


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _expected_segment_path(audio_file: str) -> str:
    base = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join(SEGMENTS_DIR, f"{base}_segment.wav")


def _expected_transcript_path(segment_file: str) -> str:
    base = os.path.splitext(os.path.basename(segment_file))[0]
    return os.path.join(TRANSCRIPTS_DIR, f"{base}_transcript.json")


def process_episode(audio_file: str, force: bool = False, skip_extraction: bool = False):
    abs_path = os.path.abspath(audio_file)
    basename = os.path.basename(abs_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {basename}")
    logger.info(f"{'='*60}")

    status, game_intro_found = episode_status(abs_path)
    if not force:
        if status == "done":
            logger.info(f"  Already processed (status=done). Use --force to reprocess.")
            return

    episode_id = upsert_episode(abs_path)

    try:
        game_intro_offset = 0.0  # seconds from episode start to segment WAV start
        expected_segment = _expected_segment_path(abs_path)
        if skip_extraction:
            logger.info(f"\n[Step 1+2] Skipping extraction (--skip-extraction) — using full audio file.")
            segment_file = abs_path
            update_episode(episode_id, segment_file=segment_file)
        elif not force and os.path.exists(expected_segment):
            logger.info(f"\n[Step 1+2] Segment already exists, skipping.")
            logger.info(f"  {os.path.basename(expected_segment)}")
            segment_file = expected_segment
            update_episode(episode_id, segment_file=segment_file)
            # Recover the stored offset so timestamp matching is still accurate
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT game_intro_timestamp FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
            if row and row[0] is not None:
                game_intro_offset = row[0]
        else:
            update_episode(episode_id, status="finding_game_intro")
            logger.info("\n[Step 1] Finding game intro timestamp...")
            timestamp = step_find_game_intro(abs_path)
            if timestamp is None:
                logger.info(f"  Game intro not found — transcribing full episode.")
                update_episode(episode_id, game_intro_found=0)
                segment_file = abs_path
                update_episode(episode_id, segment_file=segment_file)
                # game_intro_offset stays 0.0; timestamps are episode-absolute already
            else:
                game_intro_offset = timestamp
                logger.info(f"  Game intro found at {timestamp:.2f}s ({timestamp/60:.1f} min)")
                update_episode(episode_id, game_intro_timestamp=timestamp)

                update_episode(episode_id, status="extracting_segment")
                logger.info("\n[Step 2] Extracting audio segment...")
                segment_file = step_extract_segment(abs_path, timestamp)
                update_episode(episode_id, segment_file=segment_file)

        expected_transcript = _expected_transcript_path(segment_file)
        if not force and os.path.exists(expected_transcript):
            logger.info(f"\n[Step 3] Transcript already exists, skipping.")
            logger.info(f"  {os.path.basename(expected_transcript)}")
            transcript_file = expected_transcript
            update_episode(episode_id, transcript_file=transcript_file)
        else:
            update_episode(episode_id, status="transcribing")
            logger.info("\n[Step 3] Transcribing with WhisperX...")
            transcript_file = step_transcribe(segment_file)
            update_episode(episode_id, transcript_file=transcript_file)

        update_episode(episode_id, status="extracting_data")
        logger.info("\n[Step 4] Extracting game data via LM Studio...")
        game_data, raw_json, round_timestamps = step_extract_game_data(transcript_file, game_intro_offset)

        logger.info("\n[Step 5] Storing results in database...")
        store_game_data(episode_id, game_data, raw_json, transcript_file, round_timestamps)
        update_episode(
            episode_id,
            status="done",
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(f"\nDone: {basename}")
        logger.info(f"  Rounds : {len(game_data.rounds)}")
        for i, (r, ts) in enumerate(zip(game_data.rounds, round_timestamps), 1):
            ts_str = f"{ts:.1f}s ({ts/60:.1f} min)" if ts is not None else "not found"
            logger.info(f"  Round {i}: {r.answer} ({len(r.clues)} clues) @ {ts_str}")

    except Exception as e:
        update_episode(episode_id, status=f"error: {e}")
        logger.error(f"\nERROR processing {basename}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="WTCC extraction pipeline")
    parser.add_argument("files", nargs="*", help="Episode audio file(s) to process")
    parser.add_argument("--all", action="store_true", help="Process all audio files in source_audio/")
    parser.add_argument("--force", action="store_true", help="Re-process even if already done")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip game intro detection and segment extraction (single file only); "
                             "requires the segment WAV to already exist in segments/")
    args = parser.parse_args()

    if args.skip_extraction and args.all:
        parser.error("--skip-extraction cannot be used with --all")

    if not os.path.exists(SQLITE_DB):
        logger.error(f"Database not found: {SQLITE_DB}")
        logger.error("Run scripts/setup_db.py first.")
        sys.exit(1)

    if args.all:
        patterns = [
            os.path.join(SOURCE_AUDIO_DIR, "*.mp3"),
            os.path.join(SOURCE_AUDIO_DIR, "*.wav"),
            os.path.join(SOURCE_AUDIO_DIR, "*.m4a"),
            os.path.join(SOURCE_AUDIO_DIR, "*.ogg"),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        if not files:
            logger.error(f"No audio files found in {SOURCE_AUDIO_DIR}")
            sys.exit(1)
    elif args.files:
        files = args.files
    else:
        parser.print_help()
        sys.exit(1)

    errors = []
    for f in sorted(files, key=lambda p: os.path.basename(p).lower()):
        try:
            process_episode(f, force=args.force, skip_extraction=args.skip_extraction)
        except Exception as e:
            errors.append((f, str(e)))

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete: {len(files) - len(errors)}/{len(files)} succeeded")
    if errors:
        logger.warning("Errors:")
        for f, e in errors:
            logger.warning(f"  {os.path.basename(f)}: {e}")


if __name__ == "__main__":
    main()
