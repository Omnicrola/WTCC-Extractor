"""
Step 5: Transcribe the game segment using WhisperX with speaker diarization.
Outputs a JSON transcript file with word-level timestamps and speaker labels.

Requires:
    pip install whisperx
    A HuggingFace token with pyannote/speaker-diarization-3.1 accepted.
    Set HF_TOKEN in config/__init__.py or export HF_TOKEN=<your_token>

Usage:
    python scripts/05_transcribe.py <segment_wav_file>
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from config import (
    TRANSCRIPTS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BATCH_SIZE,
    HF_TOKEN,
)


def _resolve_device(requested: str) -> tuple[str, str]:
    """
    Return (device, compute_type) — falls back to CPU if CUDA is requested
    but not available in the current PyTorch build.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        print("  Install a CUDA-enabled PyTorch build to use GPU acceleration:")
        print("  https://pytorch.org/get-started/locally/")
        return "cpu", "int8"
    return requested, WHISPER_COMPUTE_TYPE


def transcribe_segment(segment_wav: str) -> str:
    """
    Transcribe segment_wav with WhisperX + diarization.
    Saves a JSON transcript to TRANSCRIPTS_DIR and returns the path.
    """
    import whisperx

    if not os.path.exists(segment_wav):
        raise FileNotFoundError(f"Segment file not found: {segment_wav}")

    if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        print("WARNING: HF_TOKEN is not set. Diarization will fail.")
        print("Set it in config/__init__.py or: export HF_TOKEN=hf_...")

    device, compute_type = _resolve_device(WHISPER_DEVICE)

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(segment_wav))[0]
    out_file = os.path.join(TRANSCRIPTS_DIR, f"{base}_transcript.json")

    print(f"Loading Whisper model: {WHISPER_MODEL} on {device} ({compute_type})")
    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=compute_type,
    )

    print(f"Transcribing: {os.path.basename(segment_wav)}")
    audio = whisperx.load_audio(segment_wav)
    result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)
    print(f"  Detected language: {result.get('language', 'unknown')}")

    # Word-level alignment
    print("Running forced alignment...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    # Speaker diarization
    print("Running speaker diarization...")
    diarize_model = whisperx.diarize.DiarizationPipeline(
        token=HF_TOKEN,
        device=device,
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Save full transcript JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Transcript saved: {out_file}")
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
        print(f"Usage: python {sys.argv[0]} <segment_wav_file>")
        sys.exit(1)

    path = transcribe_segment(sys.argv[1])
    print(f"\nTranscript: {path}")

    # Preview first 20 lines
    plain = build_plain_text(path)
    preview = "\n".join(plain.split("\n")[:20])
    print("\n--- Preview ---")
    print(preview)
