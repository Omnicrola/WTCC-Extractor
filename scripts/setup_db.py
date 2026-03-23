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
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file       TEXT    UNIQUE NOT NULL,
            episode_title    TEXT,
            release_date     TEXT,
            game_intro_timestamp REAL,
            segment_file     TEXT,
            transcript_file  TEXT,
            processed_at     TEXT,
            status           TEXT    NOT NULL DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS game_rounds (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id     INTEGER NOT NULL REFERENCES episodes(id),
            answer         TEXT,
            submitted_by   TEXT,
            raw_json       TEXT,
            raw_transcript TEXT
        );

        CREATE TABLE IF NOT EXISTS clues (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id   INTEGER NOT NULL REFERENCES game_rounds(id),
            clue_order INTEGER NOT NULL,
            clue_text  TEXT    NOT NULL
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized: {SQLITE_DB}")


if __name__ == "__main__":
    setup_database()
