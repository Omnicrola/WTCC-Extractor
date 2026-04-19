"""
Step 5: Resolve each character's canonical name using three stages.

Stage 1 — Transcript Mining (free):
  Loads the episode's WhisperX transcript JSON and extracts only the segments
  that fall within this round's time window (round_start_timestamp → next
  round's start).  Searches that bounded text for host confirmation phrases:
    "the character was X", "it was X", "correct, X", "X is correct", etc.

Stage 2 — Phonetic Pre-filter (fast, no LLM):
  Scores every character name in the Coppermind cache against the transcribed
  name and any transcript-mined names using Jaro-Winkler similarity.
  Returns the top CANDIDATE_COUNT (default 15) closest matches as a short
  candidate list.

Stage 3 — LM Studio identification (one call per character):
  Sends the transcribed name, any transcript-mined name, the round's clues,
  and the short candidate list to LM Studio.  The model selects the best
  canonical match and returns a confidence level.

Confidence → verified mapping:
  high   → verified = 1
  medium → verified = 0  (written, but flagged for review)
  low    → verified = 0  (written, but flagged for review)
  UNKNOWN → skipped, logged for manual review

Usage:
    python scripts/05_data_cleanup.py              # process only unverified rows
    python scripts/05_data_cleanup.py --all        # reprocess all rows
    python scripts/05_data_cleanup.py --test 20    # dry run: 20 rows, no DB writes
    python scripts/05_data_cleanup.py --all --test 20
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import sqlite3
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import process as fuzz_process
from rapidfuzz.distance import JaroWinkler

from config import SQLITE_DB, COPPERMIND_CACHE_DB, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LOGS_DIR
from log_utils import setup_logger

logger = setup_logger("05_data_cleanup", LOGS_DIR)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Number of phonetically similar Coppermind names to pass to the LLM.
CANDIDATE_COUNT = 15

# Maximum seconds to include after a round's start timestamp when mining the
# transcript.  A real game round (5 clues + guessing + reveal) fits well within
# 15 minutes.  This cap prevents false positives in full-episode transcripts
# where non-game content fills the gap between rounds or follows the last round.
MAX_ROUND_DURATION = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Stage 1: Transcript mining
# ---------------------------------------------------------------------------

_REVEAL_PATTERNS = [
    # "the character is/was Kaladin" / "the answer is/was Kaladin"
    r"\bthe (?:character|answer) (?:is|was)\s+([A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+){0,3})",
    # "it is/was Kaladin" / "it's Kaladin"
    r"\bit(?:'s| is| was)\s+([A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+){0,3})",
    # "yes, Kaladin" / "correct, Kaladin" / "that's right, Kaladin"
    r"\b(?:correct|that'?s right|well done|yes)[,!.]\s+([A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+){0,3})",
    # "Kaladin is correct" / "Kaladin is right"
    r"\b([A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+){0,3})\s+is (?:correct|right)\b",
]
_REVEAL_RE = [re.compile(p) for p in _REVEAL_PATTERNS]

_STOPWORDS = frozenset({
    "the", "that", "this", "what", "who", "yes", "no", "oh", "well",
    "correct", "right", "wrong", "it", "he", "she", "they", "you",
    "your", "we", "our", "a", "an", "and", "but", "or", "for",
    "their", "there", "i", "me", "my", "actually", "so", "okay",
    "alright", "great", "nice", "good", "next", "round", "clue",
    "everyone", "somebody", "someone",
})


def _load_round_text(
    transcript_file: str,
    game_intro_ts: float,
    round_start_ts: float,
    next_round_start_ts: float | None,
) -> str:
    """
    Return the joined text of WhisperX segments that fall within this round's
    time window.  round_start_ts and next_round_start_ts are episode-absolute
    seconds; game_intro_ts converts them to segment-relative.
    """
    if not transcript_file or not os.path.exists(transcript_file):
        return ""
    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seg_start = round_start_ts - game_intro_ts
    if next_round_start_ts is not None:
        seg_end = min(next_round_start_ts - game_intro_ts, seg_start + MAX_ROUND_DURATION)
    else:
        seg_end = seg_start + MAX_ROUND_DURATION

    texts = [
        seg.get("text", "").strip()
        for seg in data.get("segments", [])
        if seg_start <= seg.get("start", 0.0) < seg_end
        and seg.get("text", "").strip()
    ]
    return " ".join(texts)


def _extract_reveal_candidates(text: str) -> list[str]:
    """Return candidate character names found in a round's bounded transcript text."""
    if not text:
        return []
    candidates: list[str] = []
    for rx in _REVEAL_RE:
        for m in rx.finditer(text):
            name = m.group(1).strip().rstrip(".,!? ")
            if name.lower() in _STOPWORDS or len(name) < 3 or len(name) > 40:
                continue
            candidates.append(name)
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Phonetic pre-filter
# ---------------------------------------------------------------------------

def _load_coppermind_characters(cache_conn: sqlite3.Connection) -> list[str]:
    """
    Return Cosmere entity page titles from the Coppermind cache for use as the
    phonetic pre-filter pool.  Includes:
      - {{character  — individual named characters (2,619 pages)
      - {{shard info — the 16 Shards and related cosmic beings (18 pages)
    """
    rows = cache_conn.execute(
        """SELECT title FROM wiki_pages
           WHERE is_redirect = 0
             AND (wikitext LIKE '%{{character%' OR wikitext LIKE '%{{shard info%')"""
    ).fetchall()
    return [r[0] for r in rows]


def _direct_wiki_matches(names: list[str], cache_conn: sqlite3.Connection) -> list[str]:
    """
    Return any names from the list that exist verbatim as non-redirect pages in
    the Coppermind cache (case-insensitive).  This catches correctly-transcribed
    names that fall outside the {{character category — Shards (Ambition, Honor),
    deity-level entities (Adonalsium), and similar — so they are always present
    in the candidate list even when the phonetic filter misses them.
    """
    matched = []
    for name in names:
        if not name.strip():
            continue
        row = cache_conn.execute(
            "SELECT title FROM wiki_pages WHERE LOWER(title) = LOWER(?) AND is_redirect = 0",
            (name,),
        ).fetchone()
        if row:
            matched.append(row[0])  # use the canonical casing from the wiki
    return matched


def _phonetic_candidates(
    query_names: list[str],
    coppermind_names: list[str],
    top_n: int = CANDIDATE_COUNT,
) -> list[str]:
    """
    Score every Coppermind character name against each query using Jaro-Winkler.
    Return the top_n unique names with the highest score across all queries.
    """
    best: dict[str, float] = {}
    for query in query_names:
        if not query.strip():
            continue
        hits = fuzz_process.extract(
            query,
            coppermind_names,
            scorer=JaroWinkler.similarity,
            limit=top_n,
        )
        for name, score, _ in hits:
            if score > best.get(name, 0.0):
                best[name] = score
    return [name for name, _ in sorted(best.items(), key=lambda x: -x[1])[:top_n]]


# ---------------------------------------------------------------------------
# Stage 3: LM Studio identification
# ---------------------------------------------------------------------------

class CharacterIdentification(BaseModel):
    canonical_name: str = Field(
        description=(
            "The exact character name from the candidates list that best matches "
            "the transcribed name and clues, or 'UNKNOWN' if no candidate fits."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "'high' if very confident (especially when the transcribed name closely "
            "matches a candidate), 'medium' if plausible but uncertain, 'low' if guessing."
        )
    )
    reasoning: str = Field(
        description="One or two sentences explaining why this candidate was chosen."
    )


_SYSTEM_PROMPT = """\
You are an expert on Brandon Sanderson's Cosmere universe. You are helping \
identify the correct character from a podcast game called \
"Who's That Cosmere Character?"

In this game, a host reads clues about a Cosmere character and players try to \
guess who it is. The character's name was captured via audio transcription, which \
often contains phonetic spelling errors (e.g. "Kaliden" instead of "Kaladin", \
"Szeth son son" instead of "Szeth-son-son-Vallano").

You will receive:
- The transcribed name (may be mis-spelled due to speech-to-text)
- Optionally, a name extracted from the reveal moment in the transcript
- The clues read during the game round
- A short list of candidate character names from the Coppermind wiki

Your task: identify which candidate is the actual character.

Key guidance:
- Phonetic similarity between the transcribed/reveal name and a candidate is \
the strongest signal. A candidate that sounds like the transcribed name is very \
likely correct even if the spelling differs.
- The clues are intentionally cryptic and vague (they make the game hard), but \
use them as supporting evidence when the phonetic signal is ambiguous.
- Your answer must be the exact string from the candidates list, or "UNKNOWN" \
if you are confident none of them match.\
"""


def _build_user_prompt(
    transcribed_name: str,
    mined_names: list[str],
    clues: list[str],
    candidates: list[str],
) -> str:
    lines = [f'Transcribed name (may be mis-spelled): "{transcribed_name}"']

    extra = [n for n in mined_names if n.lower() != transcribed_name.lower()]
    if extra:
        lines.append(f'Name(s) extracted from transcript reveal: {", ".join(f"{n!r}" for n in extra)}')

    lines.append("")
    if clues:
        lines.append("Clues read during the game:")
        for i, c in enumerate(clues, 1):
            lines.append(f"  {i}. {c}")
    else:
        lines.append("(No clues available)")

    lines.append("")
    lines.append("Candidate characters from Coppermind:")
    for name in candidates:
        lines.append(f"  - {name}")

    lines.append("")
    lines.append(
        "Which candidate is the correct character? "
        "Return the exact name from the list, or 'UNKNOWN' if none fit."
    )
    return "\n".join(lines)


def _call_llm(client, model_id: str, transcribed_name: str, mined_names: list[str],
              clues: list[str], candidates: list[str]) -> CharacterIdentification | None:
    user_msg = _build_user_prompt(transcribed_name, mined_names, clues, candidates)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CharacterIdentification",
                    "schema": CharacterIdentification.model_json_schema(),
                },
            },
            temperature=0,
            extra_body={"reasoning_effort": "none"},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning("    LLM returned empty response")
            return None
        return CharacterIdentification.model_validate_json(content)
    except Exception as e:
        logger.error(f"    LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(reprocess_all: bool = False, dry_run: int = 0) -> None:
    if dry_run:
        logger.info(f"*** DRY RUN — up to {dry_run} row(s), NO database changes ***")

    logger.info("Loading Coppermind character list...")
    if not os.path.exists(COPPERMIND_CACHE_DB):
        logger.error(f"Coppermind cache not found at {COPPERMIND_CACHE_DB}")
        return
    cache_conn = sqlite3.connect(COPPERMIND_CACHE_DB)
    cache_conn.row_factory = sqlite3.Row
    coppermind_names = _load_coppermind_characters(cache_conn)
    if not coppermind_names:
        logger.error("No Coppermind characters loaded — cannot proceed.")
        cache_conn.close()
        return
    logger.info(f"  {len(coppermind_names)} character names loaded from cache.")

    from openai import OpenAI
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, timeout=120)
    loaded_models = client.models.list().data
    if not loaded_models:
        logger.error("No model loaded in LM Studio. Load a model and start the server.")
        return
    model_id = loaded_models[0].id
    logger.info(f"  LM Studio model: {model_id}")

    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row

    if reprocess_all or dry_run:
        chars = conn.execute(
            "SELECT id, canonical_name FROM characters ORDER BY canonical_name"
        ).fetchall()
    else:
        chars = conn.execute(
            "SELECT id, canonical_name FROM characters WHERE verified = 0 ORDER BY canonical_name"
        ).fetchall()

    if dry_run:
        chars = chars[:dry_run]

    total = len(chars)
    logger.info(f"Characters to process: {total}")

    stats = {
        "high":     0,
        "medium":   0,
        "low":      0,
        "unknown":  0,
        "no_candidates": 0,
        "errors":   0,
    }
    needs_review: list[tuple[str, str, str]] = []  # (orig, resolved, confidence)
    not_found:    list[str] = []

    for idx, row in enumerate(chars, 1):
        char_id        = row["id"]
        orig_name      = row["canonical_name"]
        logger.info(f"[{idx}/{total}] {orig_name!r}")

        try:
            # ----------------------------------------------------------------
            # Fetch all rounds for this character (for transcript + clues)
            # ----------------------------------------------------------------
            round_rows = conn.execute(
                """SELECT gr.id,
                          gr.episode_id,
                          gr.round_start_timestamp,
                          gr.transcribed_answer,
                          e.transcript_file,
                          COALESCE(e.game_intro_timestamp, 0.0) AS game_intro_ts
                   FROM game_rounds gr
                   JOIN episodes e ON e.id = gr.episode_id
                   WHERE gr.character_id = ?
                   ORDER BY gr.episode_id, gr.round_start_timestamp""",
                (char_id,),
            ).fetchall()

            # Clues — use the round with the most clues
            clue_rows = conn.execute(
                """SELECT c.clue_text
                   FROM clues c
                   JOIN game_rounds gr ON c.round_id = gr.id
                   WHERE gr.character_id = ?
                   ORDER BY (SELECT COUNT(*) FROM clues c2 WHERE c2.round_id = gr.id) DESC,
                            gr.id, c.clue_order""",
                (char_id,),
            ).fetchall()
            clues = [r["clue_text"] for r in clue_rows]

            # ----------------------------------------------------------------
            # Stage 1: Transcript mining
            # ----------------------------------------------------------------
            mined_names: list[str] = []
            for rr in round_rows:
                if rr["round_start_timestamp"] is None:
                    continue
                next_row = conn.execute(
                    """SELECT MIN(round_start_timestamp) FROM game_rounds
                       WHERE episode_id = ?
                         AND round_start_timestamp > ?
                         AND round_start_timestamp IS NOT NULL""",
                    (rr["episode_id"], rr["round_start_timestamp"]),
                ).fetchone()
                next_start = next_row[0] if next_row else None

                text  = _load_round_text(
                    rr["transcript_file"], rr["game_intro_ts"],
                    rr["round_start_timestamp"], next_start,
                )
                cands = _extract_reveal_candidates(text)
                mined_names.extend(cands)

            # Deduplicate mined names
            seen: set[str] = set()
            unique_mined: list[str] = []
            for n in mined_names:
                if n.lower() not in seen:
                    seen.add(n.lower())
                    unique_mined.append(n)
            mined_names = unique_mined

            if mined_names:
                logger.debug(f"  Mined names: {mined_names}")

            # ----------------------------------------------------------------
            # Stage 2: Phonetic pre-filter + direct wiki lookup
            # ----------------------------------------------------------------
            query_names = [orig_name] + mined_names
            candidates = _phonetic_candidates(query_names, coppermind_names)

            # Direct lookup: if any query name exists verbatim as a Coppermind
            # page (even outside the {{character category, e.g. Shards or
            # deity-level entities), prepend it so the LLM always sees it.
            direct = _direct_wiki_matches(query_names, cache_conn)
            for d in direct:
                if d not in candidates:
                    candidates.insert(0, d)
                    logger.info(f"  Direct wiki match: {d!r} added to candidates (non-character page)")

            if not candidates:
                logger.warning(f"  No candidates found — skipping")
                stats["no_candidates"] += 1
                not_found.append(orig_name)
                continue

            logger.debug(f"  Candidates: {candidates}")

            # ----------------------------------------------------------------
            # Stage 3: LM Studio
            # ----------------------------------------------------------------
            result = _call_llm(client, model_id, orig_name, mined_names, clues, candidates)

            if result is None:
                stats["errors"] += 1
                continue

            canonical = result.canonical_name.strip()
            confidence = result.confidence

            if canonical == "UNKNOWN" or canonical not in candidates:
                logger.warning(
                    f"  LLM returned {canonical!r} (confidence={confidence}) — skipping"
                )
                stats["unknown"] += 1
                not_found.append(orig_name)
                continue

            verified = 1 if confidence == "high" else 0

            if canonical == orig_name:
                logger.info(f"  Confirmed: {orig_name!r} (confidence={confidence})")
            else:
                logger.info(
                    f"  Resolved:  {orig_name!r} → {canonical!r} "
                    f"(confidence={confidence})"
                )
                logger.debug(f"    Reasoning: {result.reasoning}")

            if confidence != "high":
                needs_review.append((orig_name, canonical, confidence))

            stats[confidence] += 1

            # ----------------------------------------------------------------
            # Write to DB
            # ----------------------------------------------------------------
            existing = conn.execute(
                "SELECT id FROM characters WHERE canonical_name = ? AND id != ?",
                (canonical, char_id),
            ).fetchone()

            if not dry_run:
                if existing:
                    keep_id = existing[0]
                    logger.warning(f"  Merge: {orig_name!r} → existing id={keep_id} ({canonical!r})")
                    conn.execute(
                        "UPDATE game_rounds SET character_id = ? WHERE character_id = ?",
                        (keep_id, char_id),
                    )
                    conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
                    conn.execute(
                        "UPDATE characters SET verified = ? WHERE id = ? AND verified = 0",
                        (verified, keep_id),
                    )
                else:
                    conn.execute(
                        "UPDATE characters SET canonical_name = ?, verified = ? WHERE id = ?",
                        (canonical, verified, char_id),
                    )
                conn.commit()
            else:
                tag = f"(verified={verified})"
                if existing:
                    logger.info(f"  [dry run] Would merge {orig_name!r} → existing ({canonical!r}) {tag}")
                elif canonical != orig_name:
                    logger.info(f"  [dry run] Would update {orig_name!r} → {canonical!r} {tag}")
                else:
                    logger.info(f"  [dry run] Would set verified={verified} on {orig_name!r}")

        except Exception as e:
            logger.error(f"  Unexpected error: {e}", exc_info=True)
            stats["errors"] += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== Summary ===")
    logger.info(f"  High confidence   : {stats['high']}")
    logger.info(f"  Medium confidence : {stats['medium']}  ← review recommended")
    logger.info(f"  Low confidence    : {stats['low']}   ← review recommended")
    logger.info(f"  UNKNOWN / no match: {stats['unknown'] + stats['no_candidates']}")
    logger.info(f"  Errors            : {stats['errors']}")

    if needs_review:
        logger.info("")
        logger.info("--- Needs manual review ---")
        for orig, resolved, conf in needs_review:
            logger.info(f"  [{conf:6s}] {orig!r:40s} → {resolved!r}")

    if not_found:
        logger.info("")
        logger.info("--- No match found (manual identification needed) ---")
        for name in not_found:
            logger.info(f"  {name!r}")

    cache_conn.close()
    conn.close()


if __name__ == "__main__":
    reprocess_all = "--all" in sys.argv

    dry_run = 0
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        try:
            dry_run = int(sys.argv[idx + 1])
            if dry_run < 1:
                raise ValueError
        except (IndexError, ValueError):
            print("Usage: --test <number>  (e.g. --test 10)")
            sys.exit(1)

    run(reprocess_all=reprocess_all, dry_run=dry_run)
