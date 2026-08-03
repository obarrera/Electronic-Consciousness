#!/usr/bin/env python3
"""Catch narration drift before it reaches TTS/customers. Checks every scene's
narration in out/scenes.json against the terminology rules in the skill's
references/socket-terminology.md and SKILL.md's "no abbreviations" table:
banned abbreviations (spell them out instead) and incorrect/colloquial Socket
terms (use the exact product term). Run after generate_content.py, before
make_audio — run_pipeline.sh does this automatically and stops the pipeline on
a hit so a bad line never reaches a finished video.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCENES_JSON = Path("out/scenes.json")

# From SKILL.md "Narration style: no abbreviations" — write the spoken phrase,
# not the abbreviation, in narration (the TTS-only pronunciation map is a
# separate safety net applied after this check, not a substitute for it).
BANNED_ABBREVIATIONS = {
    r"\bCVEs?\b": "known vulnerability / known vulnerabilities",
    r"\bAPI keys?\b": "Socket key / access token",
    r"\bCLI\b": "the command line",
    r"\bCI/CD\b": "your pipelines / continuous integration",
    r"\bPRs?\b": "pull request(s)",
    r"\bSBOMs?\b": "software bill of materials",
    r"\brepos?\b": "repository / repositories",
}

# From references/socket-terminology.md — colloquial/incorrect phrasing that
# has a specific, verified Socket term instead. Checked case-insensitively.
BANNED_TERMINOLOGY = {
    r"\bmalware alert\b": "Known Malware",
    r"\bflagged as a virus\b": "Known Malware",
    r"\btyposquat warning\b": "Possible typosquat attack",
    r"\bhidden install scripts?\b": "Install scripts",
    r"\bscrambled code\b": "Obfuscated code / Obfuscated file",
    r"\bprotest packages?\b": "Protestware/unwanted behavior",
    r"\bescalat(?:e[sd]?|ing|ions?)\b": "the policy action Socket should take (Block/Warn/Monitor/Ignore) — 'escalate' is not a Socket term",
    r"\btwo (?:levels|tiers|types) of reachability\b": "three levels of reachability (Dependency / Precomputed / Full Application — see references/socket-terminology.md)",
}

# Words local TTS engines (Kokoro et al.) reliably mispronounce in this
# context — found the hard way on real customer videos (2026-07-31 Doug/MLP
# review round). These WARN with a suggested rewording but do not fail the
# pipeline: they are audio-quality risks, not factual errors, and they apply
# to SPOKEN narration only.
PRONUNCIATION_RISKS = {
    r"\blive\b": 'ambiguous /lɪv/ vs /laɪv/ — reword ("right inside the dashboard", "in real time")',
    r"\bremediat(?:e[sd]?|ing|ions?)\b": 'often garbled — reword ("fix", "the fix guidance", "path to a fix")',
    r"\bnoise\b": 'often sounds off — reword ("clutter", "irrelevant alerts", "the alerts that matter")',
}

ALL_RULES = {**{p: (r, "abbreviation") for p, r in BANNED_ABBREVIATIONS.items()},
             **{p: (r, "terminology") for p, r in BANNED_TERMINOLOGY.items()}}


def main() -> None:
    if not SCENES_JSON.is_file():
        sys.exit(f"{SCENES_JSON} not found — run src/generate_content.py first.")

    data = json.loads(SCENES_JSON.read_text())
    scenes = data.get("scenes", [])
    hits = []
    warns = []
    for scene in scenes:
        # Everything customer-visible gets checked, not just what is spoken:
        # callouts, titles, and slide text render into the finished video verbatim.
        slide = scene.get("slide") or {}
        fields = [
            ("narration", scene.get("narration") or ""),
            ("callout", scene.get("callout") or ""),
            ("title", scene.get("title") or ""),
            ("slide.title", slide.get("title") or ""),
            ("slide.subtitle", slide.get("subtitle") or ""),
            ("slide.stat_label", slide.get("stat_label") or ""),
            ("slide.bullets", " ".join(str(b) for b in (slide.get("bullets") or []))),
        ]
        for field, text in fields:
            if not text:
                continue
            for pattern, (suggestion, kind) in ALL_RULES.items():
                # Abbreviation bans exist for the TTS engine's mouth — they apply
                # to narration only. Terminology rules apply to ALL visible text.
                if kind == "abbreviation" and field != "narration":
                    continue
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    hits.append((f"{scene.get('id', '?')}.{field}", kind, m.group(0), suggestion))
            # Pronunciation risks are spoken-audio concerns: narration only,
            # warn without failing (a human decides if the rewording is worth it).
            if field == "narration":
                for pattern, suggestion in PRONUNCIATION_RISKS.items():
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        warns.append((f"{scene.get('id', '?')}.{field}", m.group(0), suggestion))

    if warns:
        print(f"PRONUNCIATION WARNINGS ({len(warns)}) — TTS engines have mispronounced these "
              "on real videos; consider rewording:")
        for scene_id, found, suggestion in warns:
            print(f"  [warn] scene '{scene_id}': \"{found}\" — {suggestion}")
        print()

    if not hits:
        print(f"Narration check passed: {len(scenes)} scenes, no banned terms.")
        return

    print(f"NARRATION CHECK FAILED: {len(hits)} issue(s) across {len(scenes)} scenes.\n")
    for scene_id, kind, found, suggestion in hits:
        print(f"  [{kind}] scene '{scene_id}': found \"{found}\" — use: {suggestion}")
    print("\nFix out/scenes.json's source (demo.yaml scenes[].narration), re-run "
          "src/generate_content.py, then re-check. See references/socket-terminology.md.")
    sys.exit(1)


if __name__ == "__main__":
    main()
