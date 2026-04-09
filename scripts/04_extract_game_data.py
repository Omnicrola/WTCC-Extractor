"""
Step 4: Extract structured WTCC game data from the transcript using
a local model served by LM Studio with JSON schema constrained output.

Requires:
    pip install openai pydantic
    LM Studio running with a model loaded and the local server started.
    Set context length to at least 40960 in LM Studio's server settings.

Usage:
    python scripts/04_extract_game_data.py <transcript_json_file>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import difflib
import json
import sqlite3
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field
from config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, SQLITE_DB, SOURCE_AUDIO_DIR, LOGS_DIR
from log_utils import setup_logger

logger = setup_logger("04_extract_game_data", LOGS_DIR)


# --- Pydantic schema ---

class WTCCRound(BaseModel):
    submitted_by: str = Field(description="Name of the person who submitted this round's clues, typically introduced as 'sent in by [Name]' or 'submitted by [Name]'. Empty string if not mentioned.")
    answer: str = Field(description="The correct Cosmere character name for this round, revealed when the host confirms a correct guess")
    clues: List[str] = Field(description="Ordered list of clues read by the host for this round, cleaned of disfluencies and filler words but with meaning preserved")

class WTCCRoundsOnly(BaseModel):
    """Schema used for Ollama's constrained output — rounds only, no metadata."""
    rounds: List[WTCCRound] = Field(description="All rounds of the game played in this episode, in the order they appear in the transcript")

class WTCCGameData(BaseModel):
    episode_title: str
    release_date: str
    rounds: List[WTCCRound]


# --- Prompts ---

SYSTEM_PROMPT = """You are extracting structured data from a podcast game segment transcript.

The podcast is "Who's That Cosmere Character?" — a guessing game where a host reads a series of clues \
about a character from Brandon Sanderson's Cosmere universe, and the other participants try to guess who it is.

HOW THE GAME WORKS:
- The host introduces who submitted the clues: e.g. "This one is sent in by [Name]" or "submitted by [Name]". This is not always mentioned, if it is missing leave it blank.
- The host reads each clue aloud, typically prefixed with "Clue one", "Clue two", "Clue 3", etc. It is important to retain the same order they are given in the transcript
- There are always five clues for each round
- Players call out guesses after each clue; the host says whether they are correct or not
- The round ends when a player guesses correctly and the host confirms it (e.g. "It is Helleran", "Yes, that's right")
- A single episode may contain MULTIPLE rounds back-to-back; each starts with a new submitter introduction or by the host stating that they are going to start another set of clues

THE TRANSCRIPT may contain:
- Overlapping speech and crosstalk between multiple speakers
- Disfluencies: "um", "uh", "like", false starts, self-corrections, repeated words
- Speaker labels like [eric], [SPEAKER_00], [SPEAKER_01], etc. (roles are not labeled — infer from context)
- Incorrect guesses and discussion between clues — ignore these, extract only the clues themselves

YOUR TASK:
1. Identify every distinct round in the transcript (each has its own submitter, list of five clues, and revealed answer)
2. For each round, extract:
   a. The submitter's name (introduced by the host before the first clue of that round)
   b. The ordered list of five clues read by the host, cleaned of disfluencies
   c. The confirmed correct answer (the character name the host affirms at the end of the round)

IMPORTANT: Do not invent or guess data. If a field is not present in the transcript, use an empty string.
If only one round is present, return a list with a single entry."""


# --- Few-shot example loaded from files ---
# Transcript : examples/526735692-17thshard-frost-and-dragons_TRANSCRIPT.txt
# Expected   : examples/526735692-17thshard-frost-and-dragons_expected_output.json

def _load_few_shot_example() -> str:
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    transcript_path = os.path.join(examples_dir, "526735692-17thshard-frost-and-dragons_TRANSCRIPT.txt")
    expected_path   = os.path.join(examples_dir, "526735692-17thshard-frost-and-dragons_expected_output.json")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read().strip()
    with open(expected_path, "r", encoding="utf-8") as f:
        expected_data = json.load(f)

    # Strip metadata fields — the LLM only outputs rounds
    expected_data.pop("episode_title", None)
    expected_data.pop("release_date", None)
    expected_text = json.dumps(expected_data, indent=2)

    return (
        "--- EXAMPLE ---\n\n"
        f"Input transcript:\n{transcript_text}\n\n"
        f"Expected output:\n{expected_text}\n\n"
        "--- END EXAMPLE ---"
    )

FEW_SHOT_EXAMPLE = _load_few_shot_example()

# --- Chunking ---
# Transcripts can be very long; chunking keeps each LLM call manageable.
# Overlap ensures rounds that straddle a chunk boundary are still captured.
CHUNK_LINES   = 250  # lines per chunk
OVERLAP_LINES = 60   # lines of overlap between consecutive chunks (must cover a full round)


def _chunk_plain_text(plain_text: str) -> list[str]:
    """Split a transcript into overlapping line-based chunks."""
    lines = [l for l in plain_text.split("\n") if l.strip()]
    if len(lines) <= CHUNK_LINES:
        return [plain_text]
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        chunks.append("\n".join(lines[start:end]))
        if end == len(lines):
            break
        start = end - OVERLAP_LINES
    return chunks


def _merge_rounds(all_rounds: list[WTCCRound]) -> list[WTCCRound]:
    """Deduplicate rounds from multiple chunks by answer name.

    When the same answer appears more than once (e.g. captured by two
    overlapping chunks), keep the version with the most clues.  Insertion
    order is preserved so rounds stay in transcript order.
    """
    seen: dict[str, WTCCRound] = {}
    for r in all_rounds:
        key = r.answer.strip().lower()
        if not key:
            continue
        if key not in seen or len(r.clues) > len(seen[key].clues):
            seen[key] = r
    return list(seen.values())


def find_round_timestamps(
    transcript_json_path: str,
    rounds: list,
    game_intro_offset: float,
) -> list:
    """
    For each extracted round, find its episode-absolute start timestamp by
    searching the WhisperX transcript segments.

    Strategy (in order):
      1. Submitter name — case-insensitive substring search (most reliable;
         proper nouns survive both WhisperX and LLM cleanup intact)
      2. First clue text — fuzzy match via SequenceMatcher (fallback when
         submitter is blank or not found)

    Searches forward through segments so each successive round anchors past
    the previous one, handling back-to-back rounds correctly.

    game_intro_offset: seconds from episode start to segment WAV start.
      Pass 0.0 when the full episode was transcribed (no segment extraction).

    Returns a list of floats (episode-absolute seconds), one per round.
    None entries mean no match was found for that round.
    """
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # (segment_index, start_seconds, lowercased_text)
    seg_list = [
        (i, s.get("start", 0.0), s.get("text", "").strip().lower())
        for i, s in enumerate(transcript.get("segments", []))
    ]

    results = []
    search_from = 0  # advance forward as we find each round

    for round_data in rounds:
        match_seg_idx = None
        match_time = None

        # --- Primary: submitter name substring search ---
        submitter = (round_data.submitted_by or "").strip().lower()
        if submitter:
            for seg_idx, start, text in seg_list[search_from:]:
                if submitter in text:
                    match_seg_idx = seg_idx
                    match_time = start
                    break

        # --- Fallback: fuzzy match on first clue text ---
        if match_time is None and round_data.clues:
            query = round_data.clues[0].strip().lower()
            best_ratio = 0.0
            for seg_idx, start, text in seg_list[search_from:]:
                ratio = difflib.SequenceMatcher(None, query, text).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    if ratio >= 0.45:
                        match_seg_idx = seg_idx
                        match_time = start

        if match_seg_idx is not None:
            search_from = match_seg_idx  # next round must start after this
            results.append(game_intro_offset + match_time)
        else:
            results.append(None)

    return results


def _title_from_path(path: str) -> str:
    """Derive a human-readable episode title from the transcript filename.

    Filename pattern: {id}-{platform}-{title-words}_segment_transcript.json
    Example: 526735692-17thshard-frost-and-dragons_segment_transcript.json → Frost and Dragons
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_transcript", "_segment"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("-")
    # Skip leading numeric ID and platform slug (e.g. "526735692", "17thshard")
    start = 0
    if parts and parts[0].isdigit():
        start = 2
    return " ".join(parts[start:]).title()


def _find_audio_file(transcript_path: str) -> str | None:
    """Try to locate the source audio file corresponding to a transcript."""
    stem = os.path.splitext(os.path.basename(transcript_path))[0]
    for suffix in ("_transcript", "_segment"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for ext in (".mp3", ".wav", ".m4a", ".ogg"):
        candidate = os.path.join(SOURCE_AUDIO_DIR, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def extract_game_data(transcript_json_path: str) -> tuple[WTCCGameData, str]:
    """
    Load transcript JSON, build plain text, chunk it, and call LM Studio on each chunk.
    Results are merged and deduplicated by answer name.
    Episode title is derived from the transcript filename.
    Returns (WTCCGameData, raw_json_string).
    """
    import importlib.util

    # Load build_plain_text from 03_transcribe.py via importlib
    # (numeric filename prefix makes it an invalid identifier for direct import)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "transcribe_mod",
        os.path.join(scripts_dir, "03_transcribe.py"),
    )
    transcribe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transcribe_mod)

    plain_text = transcribe_mod.build_plain_text(transcript_json_path)

    if not plain_text.strip():
        raise ValueError("Transcript is empty — nothing to extract.")

    chunks = _chunk_plain_text(plain_text)
    logger.info(f"  Transcript: {len(plain_text):,} chars, {len(chunks)} chunk(s)")

    from openai import OpenAI
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, timeout=300)

    # Resolve the identifier of whichever model is currently loaded in LM Studio
    loaded_models = client.models.list().data
    if not loaded_models:
        raise RuntimeError("No model is loaded in LM Studio. Load a model and start the server first.")
    model_id = loaded_models[0].id
    logger.info(f"  Using model: {model_id}")

    all_rounds: list[WTCCRound] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  Chunk {i}/{len(chunks)} ({len(chunk):,} chars)...")
        user_message = (
            f"{FEW_SHOT_EXAMPLE}\n\n"
            f"Now extract from this transcript:\n\n"
            f"{chunk}"
        )
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "WTCCRoundsOnly",
                    "schema": WTCCRoundsOnly.model_json_schema(),
                },
            },
            temperature=0,
            extra_body={"reasoning_effort": "none"},  # disable thinking tokens for structured extraction
        )
        chunk_json = response.choices[0].message.content
        if not chunk_json or not chunk_json.strip():
            logger.warning(f"    → Empty response from model (finish_reason={response.choices[0].finish_reason!r}), skipping chunk {i}")
            continue
        chunk_rounds = WTCCRoundsOnly.model_validate_json(chunk_json)
        logger.info(f"    → {len(chunk_rounds.rounds)} round(s) found")
        all_rounds.extend(chunk_rounds.rounds)

    merged = _merge_rounds(all_rounds)
    rounds_data = WTCCRoundsOnly(rounds=merged)
    raw_json = rounds_data.model_dump_json(indent=2)

    episode_title = _title_from_path(transcript_json_path)
    game_data = WTCCGameData(
        episode_title=episode_title,
        release_date="",
        rounds=rounds_data.rounds,
    )

    logger.info(f"  Episode title : {episode_title}")
    logger.info(f"  Rounds found  : {len(game_data.rounds)}")
    for i, r in enumerate(game_data.rounds, 1):
        logger.info(f"  Round {i}: answer={r.answer!r}, submitted_by={r.submitted_by or '(unknown)'!r}, clues={len(r.clues)}")

    return game_data, raw_json


def _save_to_db(transcript_path: str, game_data: WTCCGameData, raw_json: str):
    """Upsert the episode and store extracted rounds in the database."""
    audio_file = _find_audio_file(transcript_path)
    if audio_file is None:
        logger.warning("Could not locate source audio file — storing transcript path as key.")
        audio_file = os.path.abspath(transcript_path)
    else:
        audio_file = os.path.abspath(audio_file)

    transcript_abs = os.path.abspath(transcript_path)

    with sqlite3.connect(SQLITE_DB) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO episodes (audio_file, status) VALUES (?, 'pending')",
            (audio_file,)
        )
        row = conn.execute(
            "SELECT id, game_intro_timestamp FROM episodes WHERE audio_file = ?", (audio_file,)
        ).fetchone()
        episode_id = row[0]
        game_intro_offset = row[1] or 0.0

        # Clear any previous results for this episode
        old_rounds = conn.execute(
            "SELECT id FROM game_rounds WHERE episode_id = ?", (episode_id,)
        ).fetchall()
        for (rid,) in old_rounds:
            conn.execute("DELETE FROM clues WHERE round_id = ?", (rid,))
        conn.execute("DELETE FROM game_rounds WHERE episode_id = ?", (episode_id,))

        raw_transcript = ""
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                raw_transcript = " ".join(
                    s.get("text", "").strip()
                    for s in transcript_data.get("segments", [])
                )

        round_timestamps = find_round_timestamps(transcript_abs, game_data.rounds, game_intro_offset)

        for round_data, round_ts in zip(game_data.rounds, round_timestamps):
            cur = conn.execute(
                """INSERT INTO game_rounds
                   (episode_id, answer, submitted_by, raw_json, raw_transcript, round_start_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (episode_id, round_data.answer, round_data.submitted_by,
                 raw_json, raw_transcript, round_ts)
            )
            for i, clue in enumerate(round_data.clues):
                conn.execute(
                    "INSERT INTO clues (round_id, clue_order, clue_text) VALUES (?, ?, ?)",
                    (cur.lastrowid, i + 1, clue)
                )

        conn.execute(
            """UPDATE episodes
               SET episode_title = COALESCE(NULLIF(?, ''), episode_title),
                   transcript_file = ?,
                   status = 'done',
                   processed_at = ?
               WHERE id = ?""",
            (game_data.episode_title, transcript_abs,
             datetime.now(timezone.utc).isoformat(), episode_id)
        )
        conn.commit()
    conn.close()

    for i, (r, ts) in enumerate(zip(game_data.rounds, round_timestamps), 1):
        ts_str = f"{ts:.1f}s ({ts/60:.1f} min)" if ts is not None else "not found"
        logger.info(f"  Round {i} ({r.answer}): episode timestamp = {ts_str}")
    logger.info(f"  Saved to database: {len(game_data.rounds)} round(s) for episode_id={episode_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info(f"Usage: python {sys.argv[0]} <transcript_json_file>")
        sys.exit(1)

    transcript_path = sys.argv[1]
    game_data, raw_json = extract_game_data(transcript_path)

    logger.info(f"\n--- Extracted Game Data ({len(game_data.rounds)} round(s)) ---")
    logger.info(json.dumps(game_data.model_dump(), indent=2))

    if os.path.exists(SQLITE_DB):
        logger.info("\nSaving to database...")
        _save_to_db(transcript_path, game_data, raw_json)
    else:
        logger.warning(f"Database not found at {SQLITE_DB} — skipping save.")
        logger.warning("Run scripts/setup_db.py first if you want results stored.")
