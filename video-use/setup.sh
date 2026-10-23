#!/usr/bin/env bash
# Install video-use with local (free) transcription via faster-whisper.
# Usage: bash video-use/setup.sh
set -euo pipefail

REPO=~/Developer/video-use
HERE="$(cd "$(dirname "$0")" && pwd)"

test -d "$REPO" || git clone https://github.com/browser-use/video-use "$REPO"
cd "$REPO"
uv sync
# After uv sync: it would remove packages not listed in pyproject.toml.
uv pip install --python .venv/bin/python faster-whisper
cp "$HERE/transcribe_local.py" helpers/

mkdir -p ~/.claude/skills
ln -sfn "$REPO" ~/.claude/skills/video-use

command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
.venv/bin/python helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
