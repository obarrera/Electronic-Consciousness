#!/usr/bin/env bash
set -euo pipefail
# Scene-aligned narration: each scene is synthesized separately and padded to its
# planned duration so voiceover, captions, and the recording stay in sync.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "No project venv at .venv/ — falling back to system python3 (run run_pipeline.sh once to create the venv)." >&2
    PY=python3
  else
    echo "No .venv/bin/python and no python3 on PATH. Run run_pipeline.sh first." >&2
    exit 1
  fi
fi
"$PY" src/make_audio.py "$@"
