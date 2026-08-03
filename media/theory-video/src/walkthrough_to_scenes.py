#!/usr/bin/env python3
"""Turn a recorded manual walkthrough (out/walkthrough/recorded.json, written by
`npm run walkthrough`) into a demo.yaml `scenes:` list.

Every "Mark scene" click in the recorder starts a new group; the first
click/goto after the marker becomes that scene's action, later scrolls in the
group become scroll_pixels, and the real dwell time (until the next marker)
becomes duration_seconds. Narration/callout are left as TODO placeholders —
fill those in, then run the pipeline normally; record-demo.mjs replays the
exact actions you performed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

RECORDED = Path("out/walkthrough/recorded.json")


def slugify(text: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return text[:40] or fallback


def group_by_marker(actions: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = [[]]
    for action in actions:
        if action.get("type") == "marker":
            groups.append([])
        else:
            groups[-1].append(action)
    # Anything recorded BEFORE the first "Mark scene" click is setup noise
    # (dismissing banners, navigating into position) — not a requested scene.
    # groups[0] holds exactly those pre-marker actions; drop it when markers exist.
    if len(groups) > 1:
        groups = groups[1:]
    return [g for g in groups if g]


def build_scene(group: list[dict], index: int, next_t: int | None) -> dict | None:
    primary = next((a for a in group if a["type"] in ("goto", "click")), None)
    if primary is None:
        return None
    start_t = group[0]["t"]
    end_t = next_t if next_t is not None else group[-1]["t"] + 12000
    duration = max(6, min(30, round((end_t - start_t) / 1000))) if end_t > start_t else 12

    scroll_pixels = sum(abs(a.get("deltaY", 0)) for a in group if a["type"] == "scroll")
    scroll_pixels = max(200, min(1200, scroll_pixels)) if scroll_pixels else 0

    if primary["type"] == "goto":
        scene = {
            "id": slugify(primary["url"].rsplit("/", 1)[-1], f"scene-{index}"),
            "duration_seconds": duration,
            "action": "goto",
            "target": primary["url"],
            "narration": f"TODO narrate: {primary['url']}",
            "callout": "TODO",
        }
    else:
        label = primary.get("text") or primary.get("href") or "element"
        scene = {
            "id": slugify(label, f"scene-{index}"),
            "duration_seconds": duration,
            "action": "click",
            "selector_text": primary.get("text") or "",
            "narration": f"TODO narrate: clicked \"{label}\"",
            "callout": "TODO",
        }
        if primary.get("selector"):
            scene["selector"] = primary["selector"]
    if scroll_pixels:
        scene["scroll_pixels"] = scroll_pixels
    return scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", help="demo.yaml path to splice the scenes into (writes a .bak backup)")
    args = parser.parse_args()

    if not RECORDED.is_file():
        sys.exit(f"{RECORDED} not found — run `npm run walkthrough` first.")
    if yaml is None:
        sys.exit("pyyaml is required: pip install pyyaml")

    data = json.loads(RECORDED.read_text())
    groups = group_by_marker(data.get("actions", []))
    if not groups:
        sys.exit("No scenes were marked — click \"Mark scene\" during the next recording.")

    scenes = []
    for i, group in enumerate(groups):
        next_t = groups[i + 1][0]["t"] if i + 1 < len(groups) else None
        scene = build_scene(group, i, next_t)
        if scene:
            scenes.append(scene)

    if not scenes:
        sys.exit("Could not extract any click/goto actions from the recording.")

    if args.merge:
        target = Path(args.merge)
        cfg = yaml.safe_load(target.read_text()) if target.is_file() else {}
        cfg = cfg or {}
        target.with_suffix(target.suffix + ".bak").write_text(target.read_text()) if target.is_file() else None
        cfg["scenes"] = scenes
        target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"Wrote {len(scenes)} scenes into {target} (backup: {target}.bak)")
        print("Fill in the TODO narration/callout fields, then run the pipeline.")
    else:
        print(yaml.safe_dump({"scenes": scenes}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
