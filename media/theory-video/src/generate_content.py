#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import yaml


def srt_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_caption(text: str, max_len: int = 72) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(" ".join(current + [word])) > max_len and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "demo.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = Path("out")
    out.mkdir(exist_ok=True)

    scenes = cfg.get("scenes", [])
    narration_parts = []
    storyboard_lines = [f"# Storyboard: {cfg.get('feature_name', 'Socket Demo')}", ""]
    captions = []
    timeline = []
    t = 0.0

    # Audio paces video: when make_audio.py has already measured the ACTUAL
    # per-scene narration lengths, captions/timeline must use those, not the
    # planned duration_seconds — otherwise burned-in captions finish early
    # while speech continues. run_pipeline.sh re-runs this script after the
    # audio stage so the manifest exists by the time captions are burned in.
    actual_durations: dict[str, float] = {}
    try:
        for entry in json.loads(Path("out/audio/scene_durations.json").read_text(encoding="utf-8")):
            actual_durations[str(entry["id"])] = float(entry["seconds"])
        if actual_durations:
            print("Using actual narration durations from out/audio/scene_durations.json")
    except Exception:
        pass

    for idx, scene in enumerate(scenes, start=1):
        duration = actual_durations.get(str(scene.get("id", idx - 1)),
                                        float(scene.get("duration_seconds", 15)))
        title = scene.get("title", scene.get("id", f"Scene {idx}"))
        narration = scene.get("narration", "")
        narration_parts.append(narration)
        storyboard_lines.extend([
            f"## {idx}. {title}",
            f"- Duration: {duration:.0f}s",
            f"- Action: {scene.get('action', 'none')}",
            f"- Callout: {scene.get('callout', '')}",
            f"- Narration: {narration}",
            "",
        ])
        timeline.append({**scene, "start_seconds": t, "end_seconds": t + duration})

        chunks = re.split(r"(?<=[.!?])\s+", narration.strip())
        chunks = [c.strip() for c in chunks if c.strip()] or [narration]
        chunk_duration = duration / max(1, len(chunks))
        for chunk in chunks:
            start = t
            end = min(t + chunk_duration, timeline[-1]["end_seconds"])
            captions.append((start, end, "\n".join(wrap_caption(chunk))))
            t = end
        t = timeline[-1]["end_seconds"]

    narration = "\n\n".join(narration_parts).strip() + "\n"
    (out / "narration.md").write_text(f"# Narration\n\n{narration}", encoding="utf-8")
    (out / "narration.txt").write_text(narration, encoding="utf-8")
    (out / "storyboard.md").write_text("\n".join(storyboard_lines), encoding="utf-8")
    (out / "scenes.json").write_text(json.dumps({"config": cfg, "scenes": timeline}, indent=2), encoding="utf-8")

    srt_lines = []
    for i, (start, end, text) in enumerate(captions, start=1):
        srt_lines.extend([str(i), f"{srt_timestamp(start)} --> {srt_timestamp(end)}", text, ""])
    (out / "captions.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    print("Generated out/narration.md, out/storyboard.md, out/captions.srt, out/scenes.json")


if __name__ == "__main__":
    main()
