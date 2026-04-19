"""
Setup (one-time): Initialize the SQLite database schema.
Run once before processing any episodes.

Usage:
    python scripts/setup_db.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from config import SQLITE_DB


def setup_database():
    os.makedirs(os.path.dirname(SQLITE_DB), exist_ok=True)

    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file              TEXT    UNIQUE NOT NULL,
            episode_title           TEXT,
            release_date            TEXT,
            game_intro_timestamp    REAL,
            segment_file            TEXT,
            transcript_file         TEXT,
            processed_at            TEXT,
            status                  TEXT    NOT NULL DEFAULT 'pending',
            all_speakers_identified INTEGER NOT NULL DEFAULT 0,
            game_intro_found        INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS game_rounds (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id            INTEGER NOT NULL REFERENCES episodes(id),
            transcribed_answer    TEXT,
            submitted_by          TEXT,
            raw_json              TEXT,
            raw_transcript        TEXT,
            round_start_timestamp REAL,
            character_id          INTEGER REFERENCES characters(id)
        );

        CREATE TABLE IF NOT EXISTS clues (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id   INTEGER NOT NULL REFERENCES game_rounds(id),
            clue_order INTEGER NOT NULL,
            clue_text  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS characters (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT    NOT NULL UNIQUE,
            aliases        TEXT    NOT NULL DEFAULT '',
            verified       INTEGER NOT NULL DEFAULT 0
        );
    """)

    # Migrate existing databases that predate newer columns
    for migration in [
        "ALTER TABLE episodes ADD COLUMN all_speakers_identified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN game_intro_found        INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE game_rounds ADD COLUMN round_start_timestamp REAL",
        "ALTER TABLE game_rounds ADD COLUMN character_id INTEGER REFERENCES characters(id)",
        "ALTER TABLE characters ADD COLUMN verified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE game_rounds ADD COLUMN transcribed_answer TEXT",
    ]:
        try:
            cur.execute(migration)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    # Populate characters from distinct transcribed_answer values and link game_rounds
    cur.execute("""
        SELECT DISTINCT transcribed_answer
        FROM game_rounds
        WHERE transcribed_answer IS NOT NULL AND transcribed_answer != ''
    """)
    distinct_answers = [row[0] for row in cur.fetchall()]
    for name in distinct_answers:
        cur.execute("INSERT OR IGNORE INTO characters (canonical_name) VALUES (?)", (name,))
    conn.commit()

    cur.execute("""
        UPDATE game_rounds
        SET character_id = (
            SELECT id FROM characters WHERE canonical_name = game_rounds.transcribed_answer
        )
        WHERE character_id IS NULL
          AND transcribed_answer IS NOT NULL
          AND transcribed_answer != ''
    """)
    conn.commit()

    linked   = cur.execute("SELECT COUNT(*) FROM game_rounds WHERE character_id IS NOT NULL").fetchone()[0]
    unlinked = cur.execute("SELECT COUNT(*) FROM game_rounds WHERE character_id IS NULL").fetchone()[0]
    chars    = cur.execute("SELECT COUNT(*) FROM characters").fetchone()[0]
    print(f"  Characters : {chars} ({len(distinct_answers)} inserted or already present)")
    print(f"  game_rounds linked: {linked}, unlinked: {unlinked}")

    conn.close()
    print(f"Database initialized: {SQLITE_DB}")


if __name__ == "__main__":
    setup_database()
