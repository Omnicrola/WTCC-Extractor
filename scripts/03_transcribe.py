"""
Step 3: Transcribe the game segment using WhisperX with speaker diarization.
Outputs a JSON transcript file with word-level timestamps and speaker labels.

Requires:
    pip install whisperx
    A HuggingFace token with pyannote/speaker-diarization-3.1 accepted.
    Set HF_TOKEN in config/__init__.py or export HF_TOKEN=<your_token>

Usage:
    python scripts/03_transcribe.py <segment_wav_file>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import json
import torch
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_distance
from config import (
    TRANSCRIPTS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BATCH_SIZE,
    WHISPER_INITIAL_PROMPT,
    HF_TOKEN,
    SPEAKER_PROFILES_DIR,
    SPEAKER_SIMILARITY_THRESHOLD,
    LOGS_DIR,
)
from log_utils import setup_logger

logger = setup_logger("03_transcribe", LOGS_DIR)


def _resolve_device(requested: str) -> tuple[str, str]:
    """
    Return (device, compute_type) — falls back to CPU if CUDA is requested
    but not available in the current PyTorch build.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        logger.warning("  Install a CUDA-enabled PyTorch build to use GPU acceleration:")
        logger.warning("  https://pytorch.org/get-started/locally/")
        return "cpu", "int8"
    return requested, WHISPER_COMPUTE_TYPE


_PROFILE_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
_WHISPER_SAMPLE_RATE = 16_000


def _build_speaker_profiles(profiles_dir: str, inference) -> dict[str, np.ndarray]:
    """
    Load all audio clips from profiles_dir, group by name prefix (everything
    before the first '_' in the filename), compute an embedding per clip,
    and return a dict of {name: averaged_embedding}.
    """
    groups: dict[str, list] = defaultdict(list)
    for fname in sorted(os.listdir(profiles_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _PROFILE_AUDIO_EXTS:
            continue
        name = fname.split("_")[0]
        groups[name].append(os.path.join(profiles_dir, fname))

    MIN_SAMPLES = 8000  # ~0.5s at 16kHz — minimum for pyannote embedding model
    profiles: dict[str, np.ndarray] = {}
    for name, paths in groups.items():
        embeddings = []
        for path in paths:
            import whisperx
            clip = whisperx.load_audio(path)
            if len(clip) < MIN_SAMPLES:
                logger.warning(f"  Skipping profile clip {os.path.basename(path)!r} — too short ({len(clip)} samples)")
                continue
            waveform = torch.from_numpy(clip[None])
            emb = np.array(
                inference({"waveform": waveform, "sample_rate": _WHISPER_SAMPLE_RATE})
            ).flatten()
            embeddings.append(emb)
        if not embeddings:
            logger.warning(f"  No valid clips for profile {name!r} — skipping")
            continue
        profiles[name] = np.mean(embeddings, axis=0)
        logger.info(f"  Loaded profile: {name!r} ({len(embeddings)} clip(s))")

    return profiles


def _identify_speakers(
    diarize_segments,
    audio: np.ndarray,
    inference,
    profiles: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, str]:
    """
    For each diarized speaker label, concatenate their audio, compute an
    embedding, and match against profiles via cosine similarity.
    Returns {SPEAKER_XX: name_or_original_label}.
    """
    # Collect time ranges per speaker label (WhisperX returns a DataFrame)
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in diarize_segments.itertuples():
        speaker_segments[row.speaker].append((row.start, row.end))

    label_to_name: dict[str, str] = {}
    for label, segments in speaker_segments.items():
        chunks = []
        for start, end in segments:
            s = int(start * _WHISPER_SAMPLE_RATE)
            e = int(end * _WHISPER_SAMPLE_RATE)
            if e > s:
                chunks.append(audio[s:e])

        if not chunks:
            label_to_name[label] = label
            continue

        combined = np.concatenate(chunks)
        MIN_SAMPLES = 8000  # ~0.5s at 16kHz — minimum for pyannote embedding model
        if len(combined) < MIN_SAMPLES:
            label_to_name[label] = label
            logger.info(f"  {label} → kept as-is (audio too short: {len(combined)} samples)")
            continue

        waveform = torch.from_numpy(combined[None])  # (1, samples)
        emb = np.array(
            inference({"waveform": waveform, "sample_rate": _WHISPER_SAMPLE_RATE})
        ).flatten()

        best_name, best_sim = label, -1.0
        for name, ref_emb in profiles.items():
            sim = 1.0 - cosine_distance(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold:
            label_to_name[label] = best_name
            logger.info(f"  {label} → {best_name!r} (similarity {best_sim:.3f})")
        else:
            label_to_name[label] = label
            logger.info(f"  {label} → kept as-is (best similarity {best_sim:.3f} < {threshold})")

    return label_to_name


def _remap_speaker_labels(result: dict, label_to_name: dict[str, str]) -> None:
    """Replace SPEAKER_XX labels in-place throughout a WhisperX result dict."""
    for seg in result.get("segments", []):
        if "speaker" in seg:
            seg["speaker"] = label_to_name.get(seg["speaker"], seg["speaker"])
        for word in seg.get("words", []):
            if "speaker" in word:
                word["speaker"] = label_to_name.get(word["speaker"], word["speaker"])


def transcribe_segment(segment_wav: str) -> str:
    """
    Transcribe segment_wav with WhisperX + diarization.
    Saves a JSON transcript to TRANSCRIPTS_DIR and returns the path.
    """
    import whisperx

    if not os.path.exists(segment_wav):
        raise FileNotFoundError(f"Segment file not found: {segment_wav}")

    if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        logger.warning("HF_TOKEN is not set. Diarization will fail.")
        logger.warning("Set it in config/__init__.py or: export HF_TOKEN=hf_...")

    device, compute_type = _resolve_device(WHISPER_DEVICE)

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(segment_wav))[0]
    out_file = os.path.join(TRANSCRIPTS_DIR, f"{base}_transcript.json")

    logger.info(f"Loading Whisper model: {WHISPER_MODEL} on {device} ({compute_type})")
    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=compute_type,
    )

    # WhisperX's transcribe() doesn't accept initial_prompt directly —
    # it must be injected into the model's TranscriptionOptions dataclass.
    if WHISPER_INITIAL_PROMPT:
        model.options = dataclasses.replace(model.options, initial_prompt=WHISPER_INITIAL_PROMPT)
        logger.info("  Applied Cosmere initial_prompt to Whisper decoder options.")

    logger.info(f"Transcribing: {os.path.basename(segment_wav)}")
    audio = whisperx.load_audio(segment_wav)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)
    logger.info(f"  Detected language: {result.get('language', 'unknown')}")

    # Word-level alignment
    logger.info("Running forced alignment...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    # Speaker diarization
    logger.info("Running speaker diarization...")
    diarize_model = whisperx.diarize.DiarizationPipeline(
        token=HF_TOKEN,
        device=device,
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Speaker identification — remap SPEAKER_XX to real names if profiles exist
    profiles_dir = SPEAKER_PROFILES_DIR
    if profiles_dir and os.path.isdir(profiles_dir) and os.listdir(profiles_dir):
        logger.info("Loading speaker profiles...")
        from pyannote.audio import Model, Inference
        emb_model = Model.from_pretrained(
            "pyannote/embedding",
            token=HF_TOKEN,
        )
        inference = Inference(emb_model, window="whole")
        profiles = _build_speaker_profiles(profiles_dir, inference)
        if profiles:
            logger.info("Identifying speakers...")
            label_to_name = _identify_speakers(
                diarize_segments, audio, inference, profiles,
                threshold=SPEAKER_SIMILARITY_THRESHOLD,
            )
            _remap_speaker_labels(result, label_to_name)

    # Save full transcript JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcript saved: {out_file}")
    return out_file


def build_plain_text(transcript_json_path: str) -> str:
    """
    Convert WhisperX JSON to a plain-text string with speaker labels
    suitable for passing to the LLM.

    Format:
        [SPEAKER_00]: text of the segment
        [SPEAKER_01]: text of the next segment
        ...
    """
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{speaker}]: {text}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info(f"Usage: python {sys.argv[0]} <segment_wav_file>")
        sys.exit(1)

    path = transcribe_segment(sys.argv[1])
    logger.info(f"\nTranscript: {path}")

    # Preview first 20 lines
    plain = build_plain_text(path)
    preview = "\n".join(plain.split("\n")[:20])
    logger.info("\n--- Preview ---")
    logger.info(preview)
