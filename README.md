# WTCC Extractor

This is an automated pipeline I created for extracting structured game data from episodes of the 17th Shard podcast, wherein they often play a game called *Who's That Cosmere Character?* (WTCC).  The end goal being to produce data in a consistent format that can be used to make a simple web game that other fans can interact with.

Given one or more podcast audio files, it locates the WTCC game segment by detecting the intro "game intro", transcribes the segment with speaker diarization, and uses a local LLM to pull out the clues, answer, and submitter for each round — saving everything to a SQLite database. Episodes with multiple rounds played back-to-back are handled automatically.

This is one of the first projects I've done that heavily used AI. Claude Opus 4.6 was used to conduct research on different ways of extracting the information I wanted and processing it. Claude Code was used to create and refine the python code that actually uses it.  The project is ment to run on Windows, hence why there's some Powershell scripting. I may try to containerize it later just for ease of portability, but we'll see.


## How it works

```
Episode audio (.mp3 / .wav)
        │
        ▼
[Step 1] audfprint — locate WTCC game intro timestamp
        │
        ▼
[Step 2] ffmpeg — extract game segment
        │
        ▼
[Step 3] WhisperX + pyannote — transcribe with speaker diarization
        │
        ▼
[Step 4] Qwen 2.5 14B (Ollama) — extract clues, answer, submitter
        │
        ▼
     SQLite DB (episodes / game_rounds / clues tables)
```

## Prerequisites

Install these tools before setting up the Python environment:

| Tool | Install |
|------|---------|
| **Python 3.10+** | [python.org](https://www.python.org/) |
| **ffmpeg** | `winget install ffmpeg` / `brew install ffmpeg` / `sudo apt install ffmpeg` |
| **Ollama** | Windows PowerShell: `irm https://ollama.com/install.ps1 \| iex` — then `ollama pull qwen2.5:14b` |
| **CUDA-capable GPU** | Recommended. CPU fallback is supported but slow. |

## Installation

**1. Clone this repo and the audfprint submodule**

```bash
git clone --recurse-submodules <this-repo-url>
cd WTCC-Extractor
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is installed with CUDA 12.6 support by default (see `requirements.txt`). Adjust the `--index-url` if you need a different CUDA version or CPU-only.

**3. Configure**

```bash
cp config/__init__.py.example config/__init__.py
```

Edit `config/__init__.py` and set your HuggingFace token (required for speaker diarization):

```python
HF_TOKEN = "hf_..."  # or: export HF_TOKEN=hf_...
```

To get a token:
1. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the pyannote model terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the pyannote embedding model terms at [huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding) (required if using speaker profiles)

**4. One-time setup**

Initialize the SQLite database:
```bash
python scripts/setup_db.py
```

Build the audfprint fingerprint database from the WTCC game intro:
```bash
python scripts/setup_fingerprint.py
```

## Usage

**Process a single episode:**
```bash
python run_pipeline.py source_audio/episode.mp3
```

**Process all audio files in `source_audio/`:**
```bash
python run_pipeline.py --all
```

**Re-process an episode even if already completed:**
```bash
python run_pipeline.py source_audio/episode.mp3 --force
```

Place your podcast episode files (`.mp3`, `.wav`, `.m4a`, `.ogg`) in the `source_audio/` directory before running.

## Running individual steps

Each pipeline step can also be run standalone for debugging:

```bash
# Find the game intro timestamp in an episode
python scripts/01_find_game_intro.py source_audio/episode.mp3

# Extract the game segment (requires knowing the timestamp)
python scripts/02_extract_segment.py source_audio/episode.mp3 <timestamp_seconds>

# Transcribe a segment WAV
python scripts/03_transcribe.py segments/episode_segment.wav

# Extract game data from a transcript
python scripts/04_extract_game_data.py transcripts/episode_segment_transcript.json
```

## Output

Extracted data is stored in `output/wtcc.db` (SQLite) with three tables:

- **`episodes`** — one row per audio file; tracks status, game intro timestamp, file paths, and processing timestamps
- **`game_rounds`** — one row per round played; stores the answer, submitter, raw JSON, and full transcript text (episodes with multiple rounds produce multiple rows)
- **`clues`** — individual clues in order, linked to their game round

Intermediate files are written to:
- `segments/` — extracted game audio (WAV)
- `transcripts/` — WhisperX JSON transcripts with speaker labels and word-level timestamps

## Project structure

```
WTCC-Extractor/
├── run_pipeline.py          # Main orchestrator — run this
├── config/
│   ├── __init__.py          # Your local config (gitignored)
│   └── __init__.py.example  # Template — copy and edit
├── scripts/
│   ├── setup_db.py              # (one-time) Initialize SQLite schema
│   ├── setup_fingerprint.py     # (one-time) Build audfprint game intro database
│   ├── 01_find_game_intro.py    # Locate game intro in episode audio
│   ├── 02_extract_segment.py    # Extract game segment with ffmpeg
│   ├── 03_transcribe.py         # WhisperX transcription + diarization
│   └── 04_extract_game_data.py  # LLM-based structured extraction
├── audfprint/               # Audio fingerprinting library (submodule)
├── resources/
│   ├── wtcc_intro_fingerprint.wav  # Game intro reference audio
│   ├── wtcc_fingerprint.db  # audfprint database (generated)
│   └── speaker_profiles/    # Optional: named speaker audio clips for identification
│       └── Name_clip.wav    #   Filename prefix (before first '_') becomes the speaker label
├── output/
│   └── wtcc.db              # SQLite output database (generated)
├── source_audio/            # Put your episode files here
├── segments/                # Extracted game segments (generated)
├── transcripts/             # WhisperX JSON transcripts (generated)
└── examples/                # Sample transcript and expected output
```

## Configuration reference

All settings live in `config/__init__.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SOURCE_AUDIO_DIR` | `source_audio/` | Where to look for episode files |
| `SEGMENT_WINDOW_SECONDS` | `1800` | Max seconds to extract after the game intro (30 min) |
| `WHISPER_MODEL` | `large-v2` | WhisperX model size |
| `WHISPER_DEVICE` | `cuda` | `"cuda"` or `"cpu"` |
| `WHISPER_COMPUTE_TYPE` | `float16` | `"float16"` (GPU) or `"int8"` (CPU) |
| `HF_TOKEN` | env var | HuggingFace token for pyannote diarization and speaker embedding |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model for game data extraction |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `SPEAKER_PROFILES_DIR` | `resources/speaker_profiles/` | Folder of named audio clips for speaker identification; leave empty to skip |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity to accept a speaker match; unmatched speakers keep their `SPEAKER_XX` label |

### Speaker identification

To have speaker labels replaced with real names in transcripts, add audio clips to `resources/speaker_profiles/`. The filename prefix (everything before the first `_`) becomes the speaker's name:

```
resources/speaker_profiles/
    Eric_01.wav
    Eric_interview.wav
    Brandon_clip1.wav
```

Multiple clips per person are averaged into a single reference embedding. The folder can be left empty to skip identification entirely.
