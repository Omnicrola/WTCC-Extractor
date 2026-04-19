"""
generate_data.py
Reads wtcc.db and writes public/data.json for the React frontend.
Run this whenever the database or aliases.json is updated:  py generate_data.py

Only rounds with exactly 5 clues are included.

Aliases are loaded from data/aliases.json and merged into each round.
See data/aliases.json for the expected format.
"""
import json
import os
import sqlite3

DB_PATH      = os.path.join('output', 'wtcc.db')
ALIASES_PATH = os.path.join('data', 'aliases.json')
OUT_PATH     = os.path.join('output', 'data.json')

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Load aliases (round_id string -> list of alternate accepted answers)
aliases_map = {}
if os.path.exists(ALIASES_PATH):
    with open(ALIASES_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    aliases_map = {int(k): v for k, v in raw.get('aliases', {}).items()}
    print(f"Loaded aliases for {len(aliases_map)} round(s) from {ALIASES_PATH}")
else:
    print(f"No aliases file found at {ALIASES_PATH} — all rounds will have empty aliases.")

# Round IDs with exactly 5 clues
cur.execute("""
    SELECT round_id
    FROM clues
    GROUP BY round_id
    HAVING COUNT(*) = 5
""")
valid_round_ids = {row['round_id'] for row in cur.fetchall()}

# Episode IDs that have at least one valid round
cur.execute("""
    SELECT DISTINCT episode_id FROM game_rounds WHERE id IN ({})
""".format(','.join(str(i) for i in valid_round_ids)))
valid_episode_ids = {row['episode_id'] for row in cur.fetchall()}

# Fetch episodes (descending date order)
cur.execute("""
    SELECT id, episode_title AS title, release_date
    FROM episodes
    WHERE id IN ({})
    ORDER BY release_date DESC, id DESC
""".format(','.join(str(i) for i in valid_episode_ids)))
episodes_rows = cur.fetchall()

# Fetch rounds with row-number as round_order
# Use canonical_name from characters when available, fall back to transcribed_answer
cur.execute("""
    SELECT
        gr.id,
        gr.episode_id,
        ROW_NUMBER() OVER (PARTITION BY gr.episode_id ORDER BY gr.id) AS round_order,
        COALESCE(c.canonical_name, gr.transcribed_answer) AS character_answer,
        gr.submitted_by
    FROM game_rounds gr
    LEFT JOIN characters c ON c.id = gr.character_id
    WHERE gr.id IN ({})
    ORDER BY gr.episode_id, gr.id
""".format(','.join(str(i) for i in valid_round_ids)))
rounds_by_episode = {}
for row in cur.fetchall():
    rounds_by_episode.setdefault(row['episode_id'], []).append({
        'id':               row['id'],
        'round_order':      row['round_order'],
        'character_answer': row['character_answer'],
        'submitted_by':     row['submitted_by'],
        'aliases':          aliases_map.get(row['id'], []),
        'clues':            [],
    })

# Fetch clues
cur.execute("""
    SELECT id, round_id, clue_order, clue_text
    FROM clues
    WHERE round_id IN ({})
    ORDER BY round_id, clue_order
""".format(','.join(str(i) for i in valid_round_ids)))
clues_by_round = {}
for row in cur.fetchall():
    clues_by_round.setdefault(row['round_id'], []).append({
        'id':         row['id'],
        'clue_order': row['clue_order'],
        'clue_text':  row['clue_text'],
    })

# Attach clues to rounds
for ep_rounds in rounds_by_episode.values():
    for rnd in ep_rounds:
        rnd['clues'] = clues_by_round.get(rnd['id'], [])

# Assemble final structure
output = []
for ep in episodes_rows:
    output.append({
        'episode_id':    ep['id'],
        'episode_title': ep['title'],
        'release_date':  ep['release_date'] or '1970-01-01',
        'rounds':        rounds_by_episode.get(ep['id'], []),
    })

conn.close()

os.makedirs('output', exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

total_rounds = sum(len(ep['rounds']) for ep in output)
total_clues  = sum(len(r['clues']) for ep in output for r in ep['rounds'])
aliased      = sum(1 for ep in output for r in ep['rounds'] if r['aliases'])
print(f"Wrote {OUT_PATH}")
print(f"  Episodes : {len(output)}")
print(f"  Rounds   : {total_rounds}  ({aliased} with aliases)")
print(f"  Clues    : {total_clues}")
