#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[1/9] Checking system tools"
for tool in python3 node npm ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required. Install it, then rerun." >&2
    exit 1
  fi
done

if ! command -v piper >/dev/null 2>&1 && ! command -v espeak-ng >/dev/null 2>&1 && ! command -v say >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
    echo "Installing espeak-ng for local TTS"
    apt-get update -y >/dev/null
    apt-get install -y --no-install-recommends espeak-ng >/dev/null
  elif command -v brew >/dev/null 2>&1; then
    echo "Installing espeak with Homebrew for local TTS"
    brew install espeak || true
  fi
fi

echo "[2/9] Preparing Python environment"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
fi
.venv/bin/python -c "import yaml" 2>/dev/null || .venv/bin/python -m pip install pyyaml >/dev/null

echo "[3/9] Preparing Playwright"
[[ -d node_modules/playwright ]] || npm install
node -e "require('playwright')" 2>/dev/null || npm install
npx playwright install chromium 2>/dev/null || true

echo "[4/9] Generating narration, storyboard, captions, and scene metadata"
.venv/bin/python src/generate_content.py demo.yaml

echo "[5/9] Checking narration against Socket terminology rules"
.venv/bin/python src/check_narration.py

echo "[6/9] Creating narration audio (paces the recording)"
bash src/make_audio.sh

echo "[7/9] Re-timing captions and storyboard to the actual narration lengths"
.venv/bin/python src/generate_content.py demo.yaml

echo "[8/9] Recording browser walkthrough (read-only)"
node src/record-demo.mjs

echo "[9/9] Rendering branded MP4 and validating"
node src/make_cards.mjs
bash src/render.sh
bash src/validate_output.sh
