"""
WTCC Report Generator

Queries the pipeline database and writes a text report to the reports/ directory.

Usage:
    python reports.py overview
"""

import argparse
import glob
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SQLITE_DB, SOURCE_AUDIO_DIR

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    if not os.path.exists(SQLITE_DB):
        print(f"Database not found: {SQLITE_DB}")
        print("Run scripts/setup_db.py first.")
        sys.exit(1)
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _write_report(name: str, lines: list[str]) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{name}.txt"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _section(title: str) -> list[str]:
    bar = "=" * 60
    return [bar, title, bar]


def _subsection(title: str) -> list[str]:
    return ["", f"  {title}", "  " + "-" * (len(title) + 2)]


# ---------------------------------------------------------------------------
# Overview report
# ---------------------------------------------------------------------------

def report_overview() -> list[str]:
    conn = _get_conn()
    lines: list[str] = []

    lines += _section("WTCC Pipeline — Overview Report")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Database  : {SQLITE_DB}")
    lines.append("")

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------
    lines += _subsection("Episodes")

    total_episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    done_episodes  = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE status = 'done'"
    ).fetchone()[0]

    wav_files = (
        glob.glob(os.path.join(SOURCE_AUDIO_DIR, "*.mp3"))
        + glob.glob(os.path.join(SOURCE_AUDIO_DIR, "*.wav"))
        + glob.glob(os.path.join(SOURCE_AUDIO_DIR, "*.m4a"))
        + glob.glob(os.path.join(SOURCE_AUDIO_DIR, "*.ogg"))
    )
    audio_file_count = len(wav_files)

    lines.append(f"    Audio files in source_audio/ : {audio_file_count}")
    lines.append(f"    Total episodes in database   : {total_episodes}")
    lines.append(f"    Complete (status = done)     : {done_episodes}")
    if total_episodes > 0:
        pct = done_episodes / total_episodes * 100
        lines.append(f"    Completion rate              : {pct:.1f}%")

    # ------------------------------------------------------------------
    # Rounds per episode distribution
    # ------------------------------------------------------------------
    lines += _subsection("Rounds per Episode")

    # Count rounds per episode (including episodes with 0 rounds via LEFT JOIN)
    rows = conn.execute("""
        SELECT
            CASE WHEN round_count > 10 THEN 11 ELSE round_count END AS bucket,
            COUNT(*) AS episode_count
        FROM (
            SELECT e.id, COUNT(r.id) AS round_count
            FROM episodes e
            LEFT JOIN game_rounds r ON r.episode_id = e.id
            GROUP BY e.id
        )
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()

    buckets = {row["bucket"]: row["episode_count"] for row in rows}

    for n in range(0, 11):
        count = buckets.get(n, 0)
        label = f"{n} round{'s' if n != 1 else ' '}"
        lines.append(f"    {label:<14} : {count}")
    more_than_10 = buckets.get(11, 0)
    lines.append(f"    More than 10   : {more_than_10}")

    # ------------------------------------------------------------------
    # Round quality (clue counts)
    # ------------------------------------------------------------------
    lines += _subsection("Round Quality (across all episodes)")

    clue_counts = conn.execute("""
        SELECT r.id, COUNT(c.id) AS clue_count
        FROM game_rounds r
        LEFT JOIN clues c ON c.round_id = r.id
        GROUP BY r.id
    """).fetchall()

    total_rounds = len(clue_counts)
    complete   = sum(1 for row in clue_counts if row["clue_count"] == 5)
    incomplete = sum(1 for row in clue_counts if row["clue_count"] < 5)
    error      = sum(1 for row in clue_counts if row["clue_count"] > 5)

    lines.append(f"    Total rounds : {total_rounds}")
    lines.append(f"    Complete     : {complete}  (exactly 5 clues)")
    lines.append(f"    Incomplete   : {incomplete}  (fewer than 5 clues)")
    lines.append(f"    Error        : {error}  (more than 5 clues)")

    lines.append("")
    lines.append("=" * 60)

    conn.close()
    return lines


# ---------------------------------------------------------------------------
# Corrections report
# ---------------------------------------------------------------------------

def _fmt_processed(raw: str | None) -> str:
    """Format an ISO timestamp from the DB into a readable string."""
    if not raw:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%B %-d, %Y %-I:%M:%S").rstrip()
    except ValueError:
        return raw


def report_corrections() -> list[str]:
    conn = _get_conn()
    lines: list[str] = []

    lines += _section("WTCC Pipeline — Corrections Report")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Database  : {SQLITE_DB}")
    lines.append(f"  Scope     : Rounds with fewer or more than 5 clues")
    lines.append("")

    # Episodes that contain at least one bad round
    episodes = conn.execute("""
        SELECT DISTINCT e.id, e.audio_file, e.processed_at, e.game_intro_found
        FROM episodes e
        JOIN game_rounds r ON r.episode_id = e.id
        WHERE (
            SELECT COUNT(*) FROM clues c WHERE c.round_id = r.id
        ) != 5
        ORDER BY e.audio_file
    """).fetchall()

    if not episodes:
        lines.append("  No corrections needed — all rounds have exactly 5 clues.")
        lines.append("")
        lines.append("=" * 60)
        conn.close()
        return lines

    lines.append(f"  Episodes with incomplete/errored rounds: {len(episodes)}")

    for ep in episodes:
        episode_name = os.path.splitext(os.path.basename(ep["audio_file"]))[0]
        intro_found  = "True" if ep["game_intro_found"] else "False"
        processed    = _fmt_processed(ep["processed_at"])

        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Episode   : {episode_name}")
        lines.append(f"Episode ID: {ep['id']}")
        lines.append(f"Processed : {processed}")
        lines.append(f"Intro found: {intro_found}")

        # Only the rounds for this episode that have != 5 clues
        rounds = conn.execute("""
            SELECT r.id, r.answer
            FROM game_rounds r
            WHERE r.episode_id = ?
              AND (SELECT COUNT(*) FROM clues c WHERE c.round_id = r.id) != 5
            ORDER BY r.id
        """, (ep["id"],)).fetchall()

        for rnd in rounds:
            clues = conn.execute("""
                SELECT clue_order, clue_text
                FROM clues
                WHERE round_id = ?
                ORDER BY clue_order
            """, (rnd["id"],)).fetchall()

            clue_map = {c["clue_order"]: c["clue_text"] for c in clues}

            lines.append("")
            lines.append(f"  Round     : {rnd['id']}")
            lines.append(f"  Answer    : {rnd['answer'] or '(unknown)'}")
            lines.append("  Clues     :")

            # Always render slots 1–5; extras beyond 5 appended after
            max_slot = max(max(clue_map.keys(), default=0), 5)
            for i in range(1, max_slot + 1):
                text = clue_map.get(i, "-- MISSING CLUE --")
                lines.append(f"\t{i}) {text}")

    lines.append("")
    lines.append("=" * 60)

    conn.close()
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

REPORTS = {
    "overview":    (report_overview,    "overview"),
    "corrections": (report_corrections, "corrections"),
}


def main():
    parser = argparse.ArgumentParser(description="WTCC report generator")
    parser.add_argument(
        "report",
        choices=list(REPORTS.keys()),
        help="Report to generate",
    )
    args = parser.parse_args()

    fn, name = REPORTS[args.report]
    lines = fn()
    path = _write_report(name, lines)

    # Also print to stdout
    print("\n".join(lines))
    print(f"\nReport written to: {path}")


if __name__ == "__main__":
    main()
