# video-use

The video-use skill (https://github.com/browser-use/video-use) is set up with
local transcription instead of ElevenLabs (paid, not wanted).

- New session: run `bash video-use/setup.sh` first.
- Run helpers with `~/Developer/video-use/.venv/bin/python`.
- Transcribe with `helpers/transcribe_local.py <video_or_dir>` (faster-whisper,
  writes Scribe-shaped JSON), never `transcribe.py` / `transcribe_batch.py`.
- Local transcripts have no speaker diarization or audio events, and filler
  tagging is weaker than Scribe's.
