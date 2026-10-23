"""Transcribe videos locally with faster-whisper (no API, no cost).

Drop-in alternative to transcribe.py / transcribe_batch.py. Writes the same
Scribe-shaped JSON to <edit_dir>/transcripts/<video_stem>.json, so
pack_transcripts.py, render.py and timeline_view.py work unchanged:

    {"language_code": "de", "text": "...", "words": [
        {"text": "Hallo", "start": 0.52, "end": 0.81, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.81, "end": 0.90, "type": "spacing", "speaker_id": "speaker_0"},
        ...]}

Differences from Scribe:
  - no speaker diarization (every word is speaker_0)
  - no audio events (laughs, applause)
  - Whisper tends to drop fillers; an initial prompt full of fillers nudges it
    toward verbatim output, but filler detection is weaker than Scribe's

Cached: an existing transcript is never redone.

Usage:
    python helpers/transcribe_local.py <video_or_dir>
    python helpers/transcribe_local.py <video_or_dir> --model medium --language de
    python helpers/transcribe_local.py <video_or_dir> --edit-dir /custom/edit
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from transcribe import count_audio_tracks, extract_audio, peak_dbfs, transcript_path
from transcribe_batch import find_videos


# Whisper mirrors the style of its prompt; a prompt with fillers keeps more of them.
FILLER_PROMPT = {
    "de": "Ähm, also, äh, ich, ähm, weiß nicht, hm, ja, also äh.",
    "en": "Umm, so, uh, I, um, like, you know, hmm, yeah, uh.",
}
DEFAULT_FILLER_PROMPT = "Umm, uh, ähm, äh, hmm."


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono PCM wav -> float32 samples (skips PyAV, which faster-whisper
    would otherwise use to decode)."""
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def to_scribe_words(segments) -> tuple[list[dict], str]:
    """Flatten faster-whisper segments into Scribe's word/spacing list."""
    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            if words:
                prev_end = words[-1]["end"]
                words.append({
                    "text": " ", "start": prev_end, "end": max(prev_end, w.start),
                    "type": "spacing", "speaker_id": "speaker_0",
                })
            words.append({
                "text": text, "start": round(w.start, 3), "end": round(w.end, 3),
                "type": "word", "speaker_id": "speaker_0",
                "logprob": round(w.probability, 4),
            })
    text = " ".join(w["text"] for w in words if w["type"] == "word")
    return words, text


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model,
    language: str | None = None,
    audio_track: int = 0,
) -> Path:
    out_path = transcript_path(edit_dir, video, audio_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"cached: {out_path.name}")
        return out_path

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)
        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(f"track {audio_track + 1} of {video.name} is silent "
                               f"(peak {peak:.1f} dBFS)")

        print(f"  transcribing {video.name}", flush=True)
        segments, info = model.transcribe(
            load_wav(audio),
            language=language,
            word_timestamps=True,
            vad_filter=False,  # VAD trims silences; we need the raw gaps
            condition_on_previous_text=False,
            initial_prompt=FILLER_PROMPT.get(language or "", DEFAULT_FILLER_PROMPT),
        )
        words, text = to_scribe_words(segments)  # consumes the lazy generator

    payload = {
        "language_code": info.language,
        "language_probability": round(info.language_probability, 4),
        "text": text,
        "words": words,
        "source": "faster-whisper",
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    n_words = sum(1 for w in words if w["type"] == "word")
    print(f"  saved: {out_path.name} ({n_words} words, lang={info.language}) "
          f"in {time.time() - t0:.1f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Local transcription with faster-whisper")
    ap.add_argument("path", type=Path, help="Video file or directory of videos")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output directory (default: <videos_dir>/edit)")
    ap.add_argument("--model", default="small",
                    help="Whisper model: tiny, base, small (default), medium, large-v3")
    ap.add_argument("--language", default=None, help="ISO code, e.g. de. Omit to auto-detect.")
    ap.add_argument("--audio-track", type=int, default=0, help="Zero-based audio track")
    args = ap.parse_args()

    path = args.path.resolve()
    if path.is_dir():
        videos, videos_dir = find_videos(path), path
    elif path.is_file():
        videos, videos_dir = [path], path.parent
    else:
        sys.exit(f"not found: {path}")
    if not videos:
        sys.exit(f"no videos found in {path}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    pending = [v for v in videos if not transcript_path(edit_dir, v, args.audio_track).exists()]
    print(f"found {len(videos)} videos ({len(videos) - len(pending)} cached, {len(pending)} to transcribe)")
    if not pending:
        return

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    errors: list[tuple[Path, str]] = []
    for v in pending:
        try:
            transcribe_one(v, edit_dir, model, args.language, args.audio_track)
        except Exception as e:
            errors.append((v, str(e)))
            print(f"  x {v.name} FAILED: {e}")
    if errors:
        sys.exit(f"{len(errors)} failure(s)")


if __name__ == "__main__":
    main()
