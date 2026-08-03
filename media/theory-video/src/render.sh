#!/usr/bin/env bash
set -euo pipefail
mkdir -p out/final
# Same missing-venv behavior as make_audio.sh: prefer the project venv, fall
# back to system python3 with a note, fail with an actionable message otherwise.
PY="$PWD/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
  [[ -n "$PY" ]] || { echo "No .venv/bin/python and no python3 on PATH. Run run_pipeline.sh first." >&2; exit 1; }
fi
# CDP capture (the default) writes .mp4; the legacy recordVideo path writes
# .webm. Prefer whichever the recorder produced (it deletes the other up front).
VIDEO="out/recordings/browser-recording.webm"
[[ -s "out/recordings/browser-recording.mp4" ]] && VIDEO="out/recordings/browser-recording.mp4"
AUDIO="out/audio/voice.wav"
CAPTIONS="out/captions.srt"
OUTPUT_NAME="$("$PY" - <<'PY'
import yaml
cfg=yaml.safe_load(open('demo.yaml'))
print(cfg.get('output_name','socket-demo-video.mp4'))
PY
)"
OUT="out/final/$OUTPUT_NAME"

if [[ ! -s "$VIDEO" ]]; then
  echo "Missing browser recording: $VIDEO" >&2
  exit 1
fi
if [[ ! -s "$AUDIO" ]]; then
  echo "Missing voiceover audio: $AUDIO" >&2
  exit 1
fi

# Trim the recorder's measured lead-in (setup before scene 1) so the first
# scene starts exactly at audio zero.
LEAD="$("$PY" - <<'PY'
import json
try:
    print(max(0.0, float(json.load(open('out/recordings/timing.json')).get('lead_in_seconds', 0))))
except Exception:
    print(0.0)
PY
)"

AUDIO_DUR="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO" || echo 0)"
VIDEO_DUR="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO" || echo 0)"
PAD="$("$PY" - <<PY
import json
def dur(v, fallback):
    try:
        return float(v)
    except Exception:
        return fallback
# Playwright webms often report duration=N/A to ffprobe; fall back to the
# recorder's own timing marks rather than a blind 2s guess (which truncates the
# final scenes' narration when the recording actually ended early).
video = dur('$VIDEO_DUR', None)
if video is None:
    try:
        t = json.load(open('out/recordings/timing.json'))
        video = float(t['scenes'][-1]['video_end_s']) + float(t.get('lead_in_seconds', 0))
    except Exception:
        video = 0.0
audio = dur('$AUDIO_DUR', 0.0)
print(max(0.0, audio - (video - dur('$LEAD', 0.0)) + 1.0))
PY
)"

FADE_OUT_ST="$("$PY" - <<PY
# Guard: with an unknown audio duration a st=0 fade would black out the whole
# video 0.3s in — push the fade past the end instead (a no-op).
try:
    d = float('$AUDIO_DUR')
    print(max(0.5, d - 0.3) if d > 1.0 else 999999)
except Exception:
    print(999999)
PY
)"

# Smooth the concat joins at both ends of the main segment (paired with the
# intro/outro cards' own fade-out/in below) instead of a hard cut straight into
# narrated content.
BASE_FILTER="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,tpad=stop_mode=clone:stop_duration=${PAD},setsar=1,fade=t=in:st=0:d=0.3,fade=t=out:st=${FADE_OUT_ST}:d=0.3"
# Narration mastering: high-pass off the TTS low-end rumble, gentle 3:1 leveling
# (threshold 0.125 ≈ -18dB), then one-pass EBU R128 loudness normalization to
# -16 LUFS / -1.5 dBTP — every video lands at the same broadcast-consistent
# level regardless of TTS engine. Fades come last so they aren't re-leveled.
AUDIO_FADE="highpass=f=80,acompressor=threshold=0.125:ratio=3:attack=10:release=150,loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.3,afade=t=out:st=${FADE_OUT_ST}:d=0.3"

MAIN="out/final/.main-segment.mp4"
CAPTION_STYLE="force_style='FontName=Helvetica,FontSize=17,PrimaryColour=&HFFFFFF&,OutlineColour=&H66000000&,BorderStyle=1,Outline=1,Shadow=0,MarginV=28'"

if [[ -s "$CAPTIONS" ]] && ffmpeg -hide_banner -filters 2>/dev/null | grep -q " subtitles "; then
  ffmpeg -y -ss "$LEAD" -i "$VIDEO" -i "$AUDIO" \
    -vf "${BASE_FILTER},subtitles=${CAPTIONS}:${CAPTION_STYLE}" -af "$AUDIO_FADE" \
    -map 0:v:0 -map 1:a:0 \
    -c:v libx264 -preset slow -crf 18 -tune animation -x264-params aq-mode=3 -pix_fmt yuv420p \
    -c:a aac -b:a 160k -ar 22050 -ac 1 -shortest "$MAIN"
else
  ffmpeg -y -ss "$LEAD" -i "$VIDEO" -i "$AUDIO" \
    -vf "$BASE_FILTER" -af "$AUDIO_FADE" \
    -map 0:v:0 -map 1:a:0 \
    -c:v libx264 -preset slow -crf 18 -tune animation -x264-params aq-mode=3 -pix_fmt yuv420p \
    -c:a aac -b:a 160k -ar 22050 -ac 1 -shortest "$MAIN"
fi

# Optional music: set `music: <path>` in demo.yaml (or drop a file at
# assets/music.mp3). Deliberately scoped to the intro/outro bookend cards only —
# NOT mixed under narration for the whole video — since there's no voice to duck
# under during those cards, it plays at a normal, audible level there instead of
# the old low "background bed" volume.
MUSIC="$("$PY" - <<'PY'
import os
import yaml
cfg = yaml.safe_load(open('demo.yaml'))
m = str(cfg.get('music') or '')
if not m and os.path.isfile('assets/music.mp3'):
    m = 'assets/music.mp3'
print(m if m and os.path.isfile(m) else '')
PY
)"
MUSIC_VOL="$("$PY" - <<'PY'
import yaml
cfg = yaml.safe_load(open('demo.yaml'))
print(cfg.get('music_volume', 0.15))
PY
)"

# Branded intro/outro title cards (generated by src/make_cards.mjs) get spliced
# around the main segment with gentle fades. Music, if set, rides ONLY on these
# two cards (different offsets into the track so they don't sound identical).
if [[ -s out/cards/intro.png && -s out/cards/outro.png ]]; then
  if [[ -n "$MUSIC" ]]; then
    ffmpeg -y -loop 1 -framerate 30 -t 2.8 -i out/cards/intro.png -i "$MUSIC" \
      -vf "scale=1920:1080,setsar=1,fade=t=in:st=0:d=0.4,fade=t=out:st=2.3:d=0.5" \
      -af "volume=${MUSIC_VOL},afade=t=in:st=0:d=0.3,afade=t=out:st=2.3:d=0.5" \
      -c:v libx264 -preset slow -crf 18 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 160k -ar 22050 -ac 1 \
      -shortest out/final/.intro-segment.mp4
    # Seek into the track for the outro only when it's long enough — -ss past the
    # end + -shortest would collapse the outro segment to ~0s.
    MUSIC_DUR="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$MUSIC" 2>/dev/null || echo 0)"
    MUSIC_SEEK="$("$PY" - <<PY
try:
    print(15 if float('$MUSIC_DUR') >= 20 else 0)
except Exception:
    print(0)
PY
)"
    ffmpeg -y -loop 1 -framerate 30 -t 3.5 -i out/cards/outro.png -ss "$MUSIC_SEEK" -i "$MUSIC" \
      -vf "scale=1920:1080,setsar=1,fade=t=in:st=0:d=0.5,fade=t=out:st=3.0:d=0.5" \
      -af "volume=${MUSIC_VOL},afade=t=in:st=0:d=0.5,afade=t=out:st=3.0:d=0.5" \
      -c:v libx264 -preset slow -crf 18 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 160k -ar 22050 -ac 1 \
      -shortest out/final/.outro-segment.mp4
  else
    ffmpeg -y -loop 1 -framerate 30 -t 2.8 -i out/cards/intro.png \
      -f lavfi -t 2.8 -i anullsrc=r=22050:cl=mono \
      -vf "scale=1920:1080,setsar=1,fade=t=in:st=0:d=0.4,fade=t=out:st=2.3:d=0.5" \
      -c:v libx264 -preset slow -crf 18 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 160k -ar 22050 -ac 1 \
      -shortest out/final/.intro-segment.mp4
    ffmpeg -y -loop 1 -framerate 30 -t 3.5 -i out/cards/outro.png \
      -f lavfi -t 3.5 -i anullsrc=r=22050:cl=mono \
      -vf "scale=1920:1080,setsar=1,fade=t=in:st=0:d=0.5,fade=t=out:st=3.0:d=0.5" \
      -c:v libx264 -preset slow -crf 18 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 160k -ar 22050 -ac 1 \
      -shortest out/final/.outro-segment.mp4
  fi
  ffmpeg -y -i out/final/.intro-segment.mp4 -i "$MAIN" -i out/final/.outro-segment.mp4 \
    -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
    -map "[v]" -map "[a]" \
    -c:v libx264 -preset slow -crf 18 -tune animation -x264-params aq-mode=3 -pix_fmt yuv420p -c:a aac -b:a 160k "$OUT"
  rm -f out/final/.intro-segment.mp4 out/final/.outro-segment.mp4 "$MAIN"
  [[ -n "$MUSIC" ]] && echo "Music on intro/outro cards only: $MUSIC (volume $MUSIC_VOL) — main narration is unmixed"
else
  # No cards means no bookend to carry music — leave the main segment as narration-only
  # rather than mixing music under the whole video (that's the behavior being removed).
  mv "$MAIN" "$OUT"
  [[ -n "$MUSIC" ]] && echo "NOTE: music is set but no intro/outro cards were generated, so it wasn't used (music only plays on cards, never under narration)."
fi

"$PY" src/publish_kit.py || echo "WARN: publish kit failed (video itself is fine)"

echo "Rendered $OUT"
