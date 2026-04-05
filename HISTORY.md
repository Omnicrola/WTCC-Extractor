# Project History

## Objective

WTCC Extractor automates the extraction of structured game data from episodes of the *17th Shard* podcast, specifically the recurring segment *Who's That Cosmere Character?* (WTCC). The end goal is a consistently formatted dataset — clues, answers, and submitters for each round — that can power a web game for Cosmere fans.

Each episode may contain zero, one, or multiple back-to-back rounds. The pipeline must locate the segment in the episode audio, transcribe it, and use an LLM to pull out the structured content without manual intervention.

---

## Evolution

### March 19, 2026 — Initial build

The first working version committed all six pipeline stages:

1. Set up the SQLite database
2. Build an audio fingerprint database from the WTCC game intro jingle
3. Locate the jingle timestamp in an episode (`audfprint` audio fingerprinting library)
4. Extract the game segment using `ffmpeg`
5. Transcribe the segment with WhisperX + pyannote speaker diarization
6. Feed the transcript to a local LLM to extract clue/answer/submitter data

LLM extraction used **Ollama** with **Qwen 2.5 14B**, calling the model with JSON schema constraints for structured output. Sample data and example output were committed to validate the pipeline end-to-end.

**Early config issues fixed:** `requirements.txt` had torch packages on a single line without the `--index-url` flag, which broke installation; and `config/__init__.py` didn't exist (only the `.example` was committed), causing an `ImportError` for `SOURCE_AUDIO_DIR` at runtime.

---

### First LLM dead-end: NuExtract + Ollama M-RoPE failure

Early in development, **NuExtract-2.0-8B** was tried as the extraction model — it's a structured extraction model based on Qwen2.5-VL, and was attractive because it was designed specifically for pulling structured data out of text rather than being a general-purpose chat model.

However, NuExtract uses **M-RoPE** (multi-dimensional rotary position embeddings) as part of its Qwen2.5-VL architecture. This caused a `seq_add()` assertion failure in llama.cpp, the inference engine that Ollama uses under the hood. The architecture was simply incompatible with Ollama at the time. The model was abandoned.

---

### Second LLM dead-end: llama-cpp-python CUDA build failures

After Ollama failed, the approach shifted to calling a GGUF model directly via **llama-cpp-python**, which would bypass Ollama entirely. This ran into its own series of problems:

- The default `pip install llama-cpp-python` build has no CUDA support — GPU offloading was disabled, making inference slow.
- Building a CUDA-enabled wheel requires CMake + the CUDA toolkit. Multiple build attempts failed due to a **CMake 4.x + CUDA 13.x file-locking bug** that prevented the CUDA compiler from running correctly.
- Pre-compiled CUDA wheels from `https://abetlen.github.io/llama-cpp-python/whl/cu122` were not available for the required CUDA version.
- After installing the CUDA toolkit and retrying, the build produced hundreds of warnings and ultimately failed.

The context window was also discovered to be misconfigured at this stage — it was set to 8192 tokens when 36,000+ tokens were needed for longer transcripts.

llama-cpp-python was abandoned in favor of a simpler approach.

---

### March 22, 2026 — Refactor + speaker identification

Scripts were reorganized: setup utilities (`setup_db.py`, `setup_fingerprint.py`) were separated from the four numbered pipeline steps. The transcription step was significantly expanded into its own script (`03_transcribe.py`).

**Speaker identification** was added: named audio clips in `resources/speaker_profiles/` are embedded via pyannote and compared to diarized speakers using cosine similarity. The filename prefix (before the first `_`) becomes the speaker label. Multiple clips per person are averaged into a single embedding. Speakers above a configurable threshold get their `SPEAKER_XX` label replaced with the real name.

Results were mixed — the feature works but is sensitive to sample quality and is considered non-essential. A 10-unknown-clip limit was added to `identify_speakers.py` to prevent long runs from saving too many unidentified samples.

Other changes in this commit:
- SQLite DB moved to the `output/` directory
- `game_intro_found` flag added to the DB schema to track whether the fingerprint matched
- Episodes where the intro was not detected were skipped (later changed — see below)
- `--skip-extraction` CLI flag added for running only the LLM step on a known file
- Scripts and README renamed all instances of "jingle" to "game intro"

---

### March 29, 2026 — Switch from llama-cpp-python to LM Studio

After the failed llama-cpp-python CUDA builds, the LLM integration was rewritten to use **LM Studio** via its OpenAI-compatible local server API (`http://localhost:1234/v1`) with the standard `openai` Python package. This completely bypassed the compilation problem.

Key implementation details resolved:
- LM Studio 0.4.8 requires the `model` field to be the real identifier returned by `/v1/models`, not an arbitrary string. The pipeline was updated to auto-detect whichever model is loaded.
- The `strict` field in `response_format` must be the string `"true"` rather than a boolean in some versions, and was later removed entirely as it caused empty responses.
- **Thinking token suppression**: Qwen3 models produce `<think>` tokens by default. `reasoning_effort: "none"` was added to suppress them, since structured extraction doesn't benefit from chain-of-thought.

**Chunking** was added to handle long transcripts that fell victem to the "lost in the middle" (https://arxiv.org/abs/2307.03172) problem, where the model is much more likely to find data at the beginning or the end of a large piece of text than it is in the middle. The transcript is split into 250-line chunks with 60-line overlaps, processed independently, and merged by deduplicating on answer name.

A second example transcript (`frost-and-dragons`, a multi-round episode) was added to validate the multi-round extraction path.

The extraction schema was also refined: a lighter `WTCCRoundsOnly` model was introduced for constrained JSON output during chunked processing, separate from the outer `WTCCGameData` model.

---

### April 4, 2026 — Podcast download utility + robustness pass

**Podcast download utility**: `download_podcasts.bat` and `shardcast_config.json` were added to wrap the `podcast-downloader` tool, making it easy to pull recent Shardcast episodes from the SoundCloud RSS feed without manually browsing the site.

**Pipeline robustness improvements:**
- All database access was wrapped in a proper context manager to prevent SQLite connection leaks. On Windows, unclosed file handles accumulate and eventually cause `SQLITE_READONLY` errors; this was observed failing after approximately 84 episodes in a batch run.
- Episodes where the game intro jingle is **not detected** now fall through to full-episode transcription rather than being skipped. This was changed because some episodes contain the game but the fingerprint fails (too quiet, slight variation in the intro), and skipping them caused data loss.
- A structured logging system (`log_utils.py`) was added to write timestamped logs to `output/logs/`, making it easier to track progress and diagnose failures during large batch runs.
- Fixed a crash in speaker identification when the submitted audio sample was shorter than 0.5 seconds.

---

### April 5, 2026 — Reporting tools + data corrections

A `reports.py` entry point was added with two report types:

- **overview**: Total episode count, completion status, WAV file count, rounds-per-episode distribution (0 through 10+), and a breakdown of rounds by completeness (exactly 5 clues = complete; fewer = incomplete; more = error).
- **corrections**: Lists all rounds with the wrong number of clues, grouped by episode, with missing clues shown as `-- CLUE MISSING --`. Used to identify and fix extraction errors.

This was driven by observing a recurring error after processing many episodes:

> `Calculated padded input size per channel: (4). Kernel size: (5). Kernel size can't be greater than actual input size`

This error occurs when the audio segment submitted to the speaker identification model is too short (the convolutional kernel is larger than the audio). It was subsequently fixed in the robustness pass above.

---

## What Was Tried and Didn't Work

| Approach | Problem |
|---|---|
| **NuExtract-2.0-8B via Ollama** | Architecture uses M-RoPE (Qwen2.5-VL), which causes a `seq_add()` assertion failure in llama.cpp — fundamentally incompatible |
| **llama-cpp-python with CUDA** | Building a CUDA-enabled wheel requires CMake + CUDA toolkit; hit a CMake 4.x / CUDA 13.x file-locking bug; pre-compiled wheels unavailable for the required CUDA version |
| **llama-cpp-python without CUDA** | Works but GPU offloading is disabled, making inference impractically slow for long transcripts |
| **Ollama with Qwen 2.5 14B** | Worked initially but was superseded once CUDA build issues pushed the project toward LM Studio |
| **Skipping intro-not-found episodes** | Too aggressive — some episodes have the game but fingerprinting fails due to quiet or slightly different intro audio; changed to full-episode transcription fallback |
| **Speaker identification via audio profiles** | Works but produces mixed results depending on sample quality; left as optional and non-essential |
| **SQLite `with sqlite3.connect() as conn`** | Python's context manager for sqlite3 does NOT close the connection on exit — only commits/rolls back. On Windows, unclosed handles accumulated across 84 episodes before triggering `SQLITE_READONLY` errors; fixed with an explicit `conn.close()` pattern |
