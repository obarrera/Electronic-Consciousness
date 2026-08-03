#!/usr/bin/env bash
set -euo pipefail
PY="$PWD/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
  [[ -n "$PY" ]] || { echo "No .venv/bin/python and no python3 on PATH. Run run_pipeline.sh first." >&2; exit 1; }
fi
OUTPUT_NAME="$("$PY" - <<'PY'
import yaml
cfg=yaml.safe_load(open('demo.yaml'))
print(cfg.get('output_name','socket-demo-video.mp4'))
PY
)"
OUT="out/final/$OUTPUT_NAME"

[[ -s "out/narration.md" ]] || { echo "Missing narration" >&2; exit 1; }
[[ -s "out/storyboard.md" ]] || { echo "Missing storyboard" >&2; exit 1; }
[[ -s "out/captions.srt" ]] || { echo "Missing captions" >&2; exit 1; }
[[ -s "out/recordings/browser-recording.webm" || -s "out/recordings/browser-recording.mp4" ]] || { echo "Missing browser recording" >&2; exit 1; }
[[ -s "out/audio/voice.wav" ]] || { echo "Missing voiceover audio" >&2; exit 1; }
[[ -s "$OUT" ]] || { echo "Missing final MP4" >&2; exit 1; }
ffprobe -v error "$OUT" >/dev/null
DUR="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT" || echo unknown)"
AUD="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 out/audio/voice.wav || echo 0)"
"$PY" - "$DUR" "$AUD" <<'PY'
import json
import sys
dur, aud = float(sys.argv[1]), float(sys.argv[2])
drift = dur - aud
# intro/outro cards legitimately add ~6.3s; anything past +10s means real drift
if not (-2.0 <= drift <= 10.0):
    sys.exit(f"FAIL: final MP4 is {dur:.1f}s but narration is {aud:.1f}s (delta {drift:+.1f}s). "
             "The recording likely drifted or a stale segment was used - re-run record + render.")
print(f"A/V agreement OK: video {dur:.1f}s vs narration {aud:.1f}s ({drift:+.1f}s)")
try:
    timing = json.load(open("out/recordings/timing.json"))
    worst = max((abs(m["video_end_s"] - m["audio_end_s"]), m["id"]) for m in timing["scenes"])
    if worst[0] > 1.5:
        sys.exit(f"FAIL: scene '{worst[1]}' is {worst[0]:.2f}s out of sync with its narration. Re-record.")
    print(f"Scene sync OK: worst per-scene drift {worst[0]:.2f}s ('{worst[1]}')")
except Exception as e:
    # Consistent with the frame-QA gate below: a missing OR unreadable
    # timing.json is a validation failure, not a skip.
    sys.exit(f"FAIL: out/recordings/timing.json missing or unreadable ({e}) - re-run the recording.")
PY

# Automated visual QA: extract each scene's midpoint frame; fail on black frames.
"$PY" - "$OUT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out = sys.argv[1]
# Mirror render.sh's card-splice condition exactly (both cards non-empty) — an
# intro-only check would sample frames 2.8s off when outro generation failed.
def _cards_spliced():
    i, o = Path("out/cards/intro.png"), Path("out/cards/outro.png")
    return i.is_file() and i.stat().st_size > 0 and o.is_file() and o.stat().st_size > 0
intro = 2.8 if _cards_spliced() else 0.0
try:
    marks = json.load(open("out/recordings/timing.json"))["scenes"]
except Exception:
    # Missing timing means the recorder did not finish cleanly — that is a
    # validation FAILURE, not a skip (a black video would otherwise pass).
    sys.exit("FAIL: out/recordings/timing.json missing or unreadable - re-run the recording.")
Path("out/qa").mkdir(parents=True, exist_ok=True)
prev = 0.0
bad = []
for m in marks:
    mid = intro + (prev + float(m["audio_end_s"])) / 2
    prev = float(m["audio_end_s"])
    frame = f"out/qa/scene-{m['id']}.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}", "-i", out,
                    "-frames:v", "1", frame], check=True)
    stats = subprocess.run(
        ["ffmpeg", "-i", frame, "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"], capture_output=True, text=True)
    yavg = None
    for line in stats.stderr.splitlines():
        if "YAVG" in line:
            yavg = float(line.rsplit("=", 1)[1])
    if yavg is None:
        # A frame we could not measure counts as bad — a corrupt extraction must
        # not silently pass the "no black frames" claim.
        bad.append((m["id"], -1.0))
    elif yavg < 14:
        bad.append((m["id"], yavg))
if bad:
    sys.exit("FAIL: near-black frames at scene midpoint(s): " +
             ", ".join(f"{i} (luma {y:.0f})" for i, y in bad) +
             " - the page likely never rendered; inspect out/qa/ and re-record.")
print(f"Frame QA OK: {len(marks)} scene midpoints captured to out/qa/, none black.")
PY
echo "Validated $OUT (${DUR}s)"
echo "Review the video for secrets, customer data, and accuracy before sharing."
