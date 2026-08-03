#!/usr/bin/env python3
"""Publishing kit: embedded MP4 chapters, YouTube chapter list, SRT sidecar
(shifted for the intro card), branded thumbnail, and PUBLISH.md."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("demo.yaml"))
name = cfg.get("output_name", "video.mp4")
out = Path("out/final") / name
if not out.is_file():
    raise SystemExit(f"missing {out}")
stem = out.with_suffix("")

# Must mirror render.sh's splice condition EXACTLY (both cards, non-empty):
# render only prepends the 2.8s intro when both segments were built, and a
# mismatched assumption here shifts every chapter/caption timestamp.
def _cards_spliced() -> bool:
    intro, outro = Path("out/cards/intro.png"), Path("out/cards/outro.png")
    return intro.is_file() and intro.stat().st_size > 0 and outro.is_file() and outro.stat().st_size > 0

INTRO = 2.8 if _cards_spliced() else 0.0
timing = {}
try:
    timing = json.load(open("out/recordings/timing.json"))
except Exception:
    pass
scenes_meta = {s.get("id"): s for s in (json.load(open("out/scenes.json")).get("scenes") or [])}

def fmt_ts(sec: float) -> str:
    m, s2 = divmod(int(sec), 60)
    return f"{m}:{s2:02d}"

# --- chapters ---
chapters = []
start = INTRO
prev_end = 0.0
for mark in timing.get("scenes", []):
    title = (scenes_meta.get(mark["id"], {}).get("title") or mark["id"]).strip()
    chapters.append((INTRO + prev_end, title))
    prev_end = float(mark["audio_end_s"])
if INTRO:
    chapters.insert(0, (0.0, "Intro"))

# A 1-2 entry chapter track is noise (YouTube ignores lists under 3 chapters);
# be honest and skip instead of emitting a useless track.
if len(chapters) < 3:
    print(f"NOTE: only {len(chapters)} chapter(s) derivable (timing.json missing or too few scenes) - skipping chapter track.")
    chapters = []

if chapters:
    meta = ";FFMETADATA1\n"
    total = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)]).strip())
    for i, (t0, title) in enumerate(chapters):
        t1 = chapters[i + 1][0] if i + 1 < len(chapters) else total
        safe = re.sub(r"[=;#\\\n]", " ", title)
        meta += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(t0*1000)}\nEND={int(t1*1000)}\ntitle={safe}\n"
    Path("out/final/.chapters.ffmeta").write_text(meta)
    tmp = out.with_suffix(".chaptered.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                    "-i", "out/final/.chapters.ffmeta", "-map_metadata", "1",
                    "-map", "0", "-c", "copy", str(tmp)], check=True)
    tmp.replace(out)
    Path("out/final/.chapters.ffmeta").unlink(missing_ok=True)
    Path(f"{stem}.chapters.txt").write_text(
        "\n".join(f"{fmt_ts(t)} {title}" for t, title in chapters) + "\n")

# --- SRT sidecar shifted by the intro card ---
src_srt = Path("out/captions.srt")
if src_srt.is_file():
    def shift(match):
        h, m, sec, ms = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        total_ms = ((h * 60 + m) * 60 + sec) * 1000 + ms + int(INTRO * 1000)
        h2, rem = divmod(total_ms, 3600000)
        m2, rem = divmod(rem, 60000)
        s2, ms2 = divmod(rem, 1000)
        return f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"
    text = re.sub(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", shift, src_srt.read_text())
    Path(f"{stem}.srt").write_text(text)

# --- thumbnail ---
if Path("out/cards/intro.png").is_file():
    shutil.copyfile("out/cards/intro.png", f"{stem}-thumbnail.png")
else:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", str(out),
                    "-frames:v", "1", f"{stem}-thumbnail.png"], check=True)

# --- PUBLISH.md ---
narr_first = ""
try:
    narr_first = (json.load(open("out/scenes.json"))["scenes"][0].get("narration") or "")[:220]
except Exception:
    pass
lines = [f"# {cfg.get('feature_name', name)}", "",
         f"**Audience:** {cfg.get('audience', '')}", "",
         "## Suggested description", "",
         narr_first, ""]
# Only advertise a chapter track that actually exists (it's skipped when fewer
# than 3 chapters are derivable — see above).
if chapters:
    lines += ["## Chapters", ""]
    lines += [f"{fmt_ts(t)} {title}" for t, title in chapters]
    lines += [""]
tags = "electronic consciousness, philosophy of mind, plato, flatland, simulation theory"
tags += ", speculative philosophy"
video_note = "final video (chapters embedded)" if chapters else "final video"
lines += ["## Files", "",
          f"- `{name}` — {video_note}",
          f"- `{stem.name}.srt` — captions sidecar",
          f"- `{stem.name}-thumbnail.png` — poster/thumbnail",
          "", f"Tags: {tags}",
          "", "REVIEW before external sharing: watch end to end; confirm no secrets or",
          "customer data; verify claims against docs.socket.dev."]
Path("out/final/PUBLISH.md").write_text("\n".join(lines) + "\n")
chapters_note = "chapters embedded" if chapters else "no chapter track"
print(f"Publish kit: {chapters_note}, {stem.name}.srt, {stem.name}-thumbnail.png, PUBLISH.md")
