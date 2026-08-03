#!/usr/bin/env python3
"""Auto-build a `scenes:` click path from a dashboard sitemap when nobody hand-
authored scenes or recorded a manual walkthrough. Prefers a fresh, live
`out/sitemap.json` (run `npm run map` first for the freshest, most accurate
click path); falls back to the bundled `assets/dashboard-sitemap.json` shipped
with the project (a point-in-time crawl, curated per video_type) when no live
crawl is available.

Picks pages by `video_type` (see demo.yaml) using a priority order tuned per
type, then narration/callout are left as TODO placeholders grounded in what
each page actually shows — fill those in against docs.socket.dev before
running the pipeline, per the skill's mandatory preflight.
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

LIVE_SITEMAP = Path("out/sitemap.json")
BUNDLED_SITEMAP = Path("assets/dashboard-sitemap.json")

# Generic dashboard chrome — never a strong signal of what a scene is *about*,
# so these are excluded when picking a representative element to highlight.
# Built from what `npm run map` actually captures on socketdev-demo (org
# switcher, sidebar, pagination, panel toggles) — not a guess.
EXACT_CHROME = {
    "collapse sidebar", "open menu", "all repositories", "open user menu", "more",
    "socket products", "help center new", "help center", "request refresh", "refresh",
    "insights", "show options", "filter", "display", "dismiss notification", "all",
    "close insights panel", "go to first page", "go to previous page", "go to next page",
    "go to last page", "load more", "sort by: last activity", "configure repositories",
    "learn more",
}

# Priority order per video_type: regex patterns matched against the URL path,
# in the order pages should appear. The FIRST unused page matching each
# pattern is taken; patterns can repeat implicitly via distinct regexes.
PLANS = {
    "dashboard": [r"^/?$", r"/repositories$", r"/alerts$", r"/scans$", r"/events", r"/dependencies$"],
    "sca": [r"^/?$", r"/alerts$", r"/alert/", r"/dependencies$", r"/threat-intel$", r"/threat-intel/campaign/"],
    "training": [r"^/?$", r"/alerts$", r"/threat-intel$", r"/threat-intel/campaign/", r"/alert/"],
}
DEFAULT_PLAN = [r"^/?$", r"/alerts$", r"/repositories$", r"/dependencies$", r"/scans$"]

# List-style pages read better with a mid-scene scroll; detail pages read
# better with a slow zoom toward the thing the scene is about.
SCROLL_PATH_RE = re.compile(r"^/(repositories|alerts|dependencies|scans|threat-intel)/?$")
DETAIL_PATH_RE = re.compile(r"/(alert|campaign)/")


def slugify(text: str, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return text[:40] or fallback


def org_root_and_slug(demo_url: str) -> tuple[str, str]:
    """The dashboard org root (not whatever sub-page demo_url happens to point
    at — sca/training default to e.g. .../alerts, and pages must NOT nest
    under that) plus the org slug, for filtering it out of highlight picks."""
    m = re.match(r"(https://socket\.dev/dashboard/org/[^/?]+)", demo_url)
    if m:
        return m.group(1), m.group(1).rsplit("/", 1)[-1]
    return demo_url.split("?")[0].rstrip("/"), ""


def load_pages(video_type: str, demo_url: str) -> tuple[list[dict], str]:
    if LIVE_SITEMAP.is_file():
        data = json.loads(LIVE_SITEMAP.read_text())
        org = data.get("org", demo_url).rstrip("/")
        _, org_slug = org_root_and_slug(demo_url)
        pages = []
        for p in data.get("pages", []):
            path = p["url"][len(org):] or "/"
            pages.append({
                "url": p["url"], "path": path, "heading": p.get("heading") or p.get("title") or "",
                "clickable_controls": p.get("clickable_controls") or [],
            })
        return pages, org_slug
    if BUNDLED_SITEMAP.is_file():
        data = json.loads(BUNDLED_SITEMAP.read_text())
        org, org_slug = org_root_and_slug(demo_url)
        pages = []
        for p in data.get("pages", []):
            if video_type not in ("media", None) and p.get("video_types") and video_type not in p["video_types"]:
                continue
            pages.append({
                "url": org + p["path"], "path": p["path"] or "/", "heading": p.get("heading") or "",
                "clickable_controls": p.get("clickable_controls") or [],
            })
        return pages, org_slug
    sys.exit(f"Neither {LIVE_SITEMAP} nor {BUNDLED_SITEMAP} exists. Run `npm run map` first, "
              "or scaffold a project whose video_type ships a bundled sitemap.")


def pick_pages(pages: list[dict], video_type: str, count: int) -> list[dict]:
    plan = PLANS.get(video_type, DEFAULT_PLAN)
    by_path = pages
    chosen: list[dict] = []
    used = set()
    for pattern in plan:
        if len(chosen) >= count:
            break
        for p in by_path:
            if p["path"] in used:
                continue
            if re.search(pattern, p["path"]):
                chosen.append(p)
                used.add(p["path"])
                break
    # Backfill with whatever's left if the plan didn't find enough matches.
    for p in by_path:
        if len(chosen) >= count:
            break
        if p["path"] not in used:
            chosen.append(p)
            used.add(p["path"])
    return chosen


def pick_control(controls: list[str], org_slug: str) -> str:
    for c in controls:
        low = re.sub(r"\s+", " ", c.strip().lower())
        if not low or low in EXACT_CHROME or len(c) > 30:
            continue
        if "enterprise" in low or (org_slug and low == org_slug.lower()):
            continue
        return c
    return ""


def build_scenes(pages: list[dict], duration_seconds: int, org_slug: str) -> list[dict]:
    n = len(pages)
    if n == 0:
        return []
    base = max(10, duration_seconds // n)
    scenes = []
    for i, p in enumerate(pages):
        heading = p["heading"].split(" - Socket")[0].split(" — Socket")[0].strip() or p["path"]
        control = pick_control(p.get("clickable_controls") or [], org_slug)
        goto_duration = base - 6 if control else base

        scene = {
            "id": slugify(heading, f"scene-{i}"),
            "duration_seconds": goto_duration,
            "action": "goto",
            "target": p["url"],
            "narration": f"TODO narrate: {heading}",
            "callout": heading,
        }
        if DETAIL_PATH_RE.search(p["path"]):
            # Slow zoom toward the thing this detail page is about, e.g. the
            # alert ID or campaign name — the last token of the heading.
            scene["zoom"] = 1.2
            scene["zoom_text"] = heading.rsplit(" ", 1)[-1]
        elif SCROLL_PATH_RE.match(p["path"]):
            # List pages (alerts, repositories, dependencies, scans, threat
            # intel) read better with a gentle mid-scene scroll than a static shot.
            scene["scroll_pixels"] = 600
        scenes.append(scene)

        if control:
            # A short follow-up beat that highlights a REAL on-screen control
            # (a filter tab, a column header) instead of just narrating over a
            # static page — non-mutating, so it's safe even in read_only mode.
            scenes.append({
                "id": f"{scene['id']}-highlight",
                "duration_seconds": 6,
                "action": "highlight",
                "selector_text": control,
                "narration": f"TODO narrate: point out \"{control}\" on {heading}",
                "callout": control,
            })
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", help="demo.yaml path to splice the scenes into (writes a .bak backup)")
    parser.add_argument("--count", type=int, default=0, help="Number of scenes (default: duration_seconds // 18, clamped 4-7)")
    args = parser.parse_args()

    if yaml is None:
        sys.exit("pyyaml is required: pip install pyyaml")

    demo_path = Path(args.merge or "demo.yaml")
    cfg = yaml.safe_load(demo_path.read_text()) if demo_path.is_file() else {}
    cfg = cfg or {}
    video_type = cfg.get("video_type") or "media"
    demo_url = cfg.get("demo_url") or "https://socket.dev/dashboard/org/socketdev-demo"
    duration_seconds = int(cfg.get("duration_seconds") or 120)

    count = args.count or max(4, min(7, round(duration_seconds / 18)))
    pages, org_slug = load_pages(video_type, demo_url)
    if not pages:
        if LIVE_SITEMAP.is_file():
            sys.exit(f"out/sitemap.json has no pages for video_type '{video_type}'. "
                      "Its crawl root may not match demo_url, or capture_clickables/exclude "
                      "filtered everything out — check demo.yaml's crawl: block.")
        sys.exit(
            f"{BUNDLED_SITEMAP} only covers the dashboard org (tagged for "
            "video_type dashboard/sca/training), so it has nothing for "
            f"'{video_type}'. Either:\n"
            f"  1. Run `npm run map` against this video_type's own root "
            f"({demo_url}) to produce a live {LIVE_SITEMAP}, then re-run this "
            "script — or\n"
            "  2. Hand-author `scenes:` in demo.yaml instead (the right call for "
            "docs/blog/website/package pages, which don't have a crawlable "
            "dashboard structure to auto-plan from)."
        )
    chosen = pick_pages(pages, video_type, count)
    scenes = build_scenes(chosen, duration_seconds, org_slug)

    if args.merge:
        target = Path(args.merge)
        if target.is_file():
            target.with_suffix(target.suffix + ".bak").write_text(target.read_text())
        cfg["scenes"] = scenes
        target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"Wrote {len(scenes)} auto-planned scenes into {target} (backup: {target}.bak)")
        print("Scenes vary automatically: scroll on list pages, zoom on detail pages, "
              "and a highlight beat wherever a real on-screen control was found.")
        if not cfg.get("music"):
            print("No music set — for a licensed/royalty-free track, set `music: <path>` "
                  "in demo.yaml or drop it at assets/music.mp3 (plays on the branded "
                  "intro/outro cards only — never under the narrated main segment).")
        print("Fill in the TODO narration fields (verify against docs.socket.dev), then run the pipeline.")
    else:
        print(yaml.safe_dump({"scenes": scenes}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
