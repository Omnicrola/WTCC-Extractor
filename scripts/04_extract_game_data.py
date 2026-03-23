"""
Step 4: Extract structured WTCC game data from the transcript using
Qwen 2.5 14B via Ollama with JSON schema constrained output.

Requires:
    pip install ollama pydantic
    ollama pull qwen2.5:14b

Usage:
    python scripts/04_extract_game_data.py <transcript_json_file>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List
from pydantic import BaseModel, Field
from config import OLLAMA_MODEL, OLLAMA_HOST


# --- Pydantic schema ---

class WTCCRound(BaseModel):
    submitted_by: str = Field(description="Name of the person who submitted this round's clues, typically introduced as 'sent in by [Name]' or 'submitted by [Name]'. Empty string if not mentioned.")
    answer: str = Field(description="The correct Cosmere character name for this round, revealed when the host confirms a correct guess")
    clues: List[str] = Field(description="Ordered list of clues read by the host for this round, cleaned of disfluencies and filler words but with meaning preserved")

class WTCCGameData(BaseModel):
    episode_title: str = Field(description="Full episode title as spoken or mentioned in the transcript")
    release_date: str = Field(description="Release or recording date in YYYY-MM-DD format, or empty string if unknown")
    rounds: List[WTCCRound] = Field(description="All rounds of the game played in this episode, in the order they appear in the transcript")


# --- Prompts ---

SYSTEM_PROMPT = """You are extracting structured data from a podcast game segment transcript.

The podcast is "Who's That Cosmere Character?" — a guessing game where a host reads a series of clues \
about a character from Brandon Sanderson's Cosmere universe, and the other participants try to guess who it is.

HOW THE GAME WORKS:
- The host introduces who submitted the clues: e.g. "This one is sent in by [Name]" or "submitted by [Name]"
- The host reads each clue aloud, typically prefixed with "Clue one", "Clue two", "Clue 3", etc.
- Players call out guesses after each clue; the host says whether they are correct or not
- The round ends when a player guesses correctly and the host confirms it (e.g. "It is Helleran", "Yes, that's right")
- A single episode may contain MULTIPLE rounds back-to-back; each starts with a new submitter introduction

THE TRANSCRIPT may contain:
- Overlapping speech and crosstalk between multiple speakers
- Disfluencies: "um", "uh", "like", false starts, self-corrections, repeated words
- Speaker labels like [SPEAKER_00], [SPEAKER_01], etc. (roles are not labeled — infer from context)
- Incorrect guesses and discussion between clues — ignore these, extract only the clues themselves

YOUR TASK:
1. Identify every distinct round in the transcript (each has its own submitter, clue list, and revealed answer)
2. For each round, extract:
   a. The submitter's name (introduced by the host before the first clue of that round)
   b. The ordered list of clues read by the host, cleaned of disfluencies
   c. The confirmed correct answer (the character name the host affirms at the end of the round)
3. Extract the episode title and date if mentioned anywhere in the transcript (these apply to the whole episode)

IMPORTANT: Do not invent or guess data. If a field is not present in the transcript, use an empty string.
If only one round is present, return a list with a single entry."""


# --- Few-shot example built from the actual Adolin & Maya episode ---
# Transcript: examples/1372981636-17thshard-adolin-and-maya_TRANSCRIPT.json
# Expected output: examples/1372981636-17thshard-adolin-and-maya_final_output.json

FEW_SHOT_EXAMPLE = """\
--- EXAMPLE ---

Input transcript:
[SPEAKER_02] Welcome to Who's That Cosmere Character, the game show where you send five clues and a character to WTCCS7THR.com.
[SPEAKER_02] I read each clue aloud and these guys have a chance to guess who's that Cosmere character.
[SPEAKER_02] This first one is sent in by Redding.
[SPEAKER_02] Clue one. This character was pierced by metal.
[SPEAKER_03] Is it Vin?
[SPEAKER_02] It is not Vin.
[SPEAKER_01] I have a guy in mind. I'm pretty sure his name is Salem. He's one of the soldiers in Alaintris.
[SPEAKER_02] Clue two. This character interacted with main characters both on and off screen.
[SPEAKER_04] Thaddeus.
[SPEAKER_02] It is not Thaddeus.
[SPEAKER_03] Is it Fenderana?
[SPEAKER_02] It's not Fenderana. Oh, that's good. I like that guess.
[SPEAKER_02] Clue three. This character is involved with secret societies.
[SPEAKER_00] How about Malon?
[SPEAKER_02] Not Malon.
[SPEAKER_02] Clue 4, this character's dead.
[SPEAKER_03] Is it the Fused that is like in Stormlight 4 who's like at the prologue?
[SPEAKER_02] No, it's not him.
[SPEAKER_02] Is it Orasur? No, it's not Orasur.
[SPEAKER_02] Clue five. This character has dead hair.
[SPEAKER_02] It's Chana. No, it is not Chana.
[SPEAKER_04] Is it Helleran?
[SPEAKER_02] It is Helleran. Nice.
[SPEAKER_02] Alright, round two! This next one is sent in by Mira.
[SPEAKER_02] Clue one. This character is a scholar.
[SPEAKER_03] Navani?
[SPEAKER_02] Not Navani.
[SPEAKER_02] Clue two. This character has a close bond with a spren.
[SPEAKER_01] Shallan?
[SPEAKER_02] Not Shallan.
[SPEAKER_02] Clue three. This character helped design a famous weapon.
[SPEAKER_04] Is it Rushu?
[SPEAKER_02] It is Rushu! Well done.

Expected output:
{
  "episode_title": "Adolin and Maya",
  "release_date": "2024-02-01",
  "rounds": [
    {
      "submitted_by": "Redding",
      "answer": "Helaran",
      "clues": [
        "This character was pierced by metal.",
        "This character interacted with main characters both on and off screen.",
        "This character is involved in secret societies.",
        "This character is dead.",
        "This character has dead hair."
      ]
    },
    {
      "submitted_by": "Mira",
      "answer": "Rushu",
      "clues": [
        "This character is a scholar.",
        "This character has a close bond with a spren.",
        "This character helped design a famous weapon."
      ]
    }
  ]
}

--- END EXAMPLE ---"""


def extract_game_data(transcript_json_path: str) -> tuple[WTCCGameData, str]:
    """
    Load transcript JSON, build plain text, and call Ollama to extract game data.
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

    user_message = (
        f"{FEW_SHOT_EXAMPLE}\n\n"
        f"Now extract from this transcript:\n\n"
        f"{plain_text}"
    )

    from ollama import Client
    client = Client(host=OLLAMA_HOST)
    print(f"Sending transcript to {OLLAMA_MODEL} ({len(plain_text):,} chars)...")

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=WTCCGameData.model_json_schema(),
        options={"temperature": 0},
    )

    raw_json = response.message.content
    game_data = WTCCGameData.model_validate_json(raw_json)

    print(f"  Rounds found: {len(game_data.rounds)}")
    for i, r in enumerate(game_data.rounds, 1):
        print(f"  Round {i}: answer={r.answer!r}, submitted_by={r.submitted_by or '(unknown)'!r}, clues={len(r.clues)}")

    return game_data, raw_json


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <transcript_json_file>")
        sys.exit(1)

    game_data, raw_json = extract_game_data(sys.argv[1])

    print(f"\n--- Extracted Game Data ({len(game_data.rounds)} round(s)) ---")
    print(json.dumps(game_data.model_dump(), indent=2))
