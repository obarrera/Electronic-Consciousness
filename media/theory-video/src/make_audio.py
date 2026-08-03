#!/usr/bin/env python3
"""Generate narration audio aligned to scene durations.

Synthesizes each scene's narration separately (Piper, espeak-ng, or macOS say),
pads it with silence to the scene's planned duration, and concatenates the parts
into out/audio/voice.wav. This keeps voiceover, captions, and the recorded
walkthrough in sync instead of reading the whole script at natural pace.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("out/audio")
SCENES_JSON = Path("out/scenes.json")


def load_scenes() -> tuple[list[dict], dict]:
    data = json.loads(SCENES_JSON.read_text())
    if isinstance(data, dict):
        return data.get("scenes", []), data.get("config", {})
    return data, {}


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


PIPER_HOME = Path.home() / ".local/share/socket-demo-video"

# Abbreviations read badly in TTS. Narration should AVOID them entirely (see
# SKILL.md narration style), but this map is the safety net: applied to the TTS
# input only, never to captions or on-screen text. Extend via tts.pronunciations.
DEFAULT_PRONUNCIATIONS = {
    "CVEs": "C V E's",
    "CVE": "C V E",
    "APIs": "A P I's",
    "API": "A P I",
    "CI/CD": "C I, C D",
    "CLI": "command line",
    "SBOM": "S bomb",
    "SBOMs": "S bombs",
    "URL": "U R L",
    "npm": "N P M",
    "PyPI": "pie P I",
    "SSO": "single sign on",
    "PR": "pull request",
    "PRs": "pull requests",
    "vs.": "versus",
    "e.g.": "for example",
}


def spoken_text(text: str, tts: dict) -> str:
    """Rewrite narration for the TTS engine's mouth, not the reader's eye."""
    mapping = dict(DEFAULT_PRONUNCIATIONS)
    mapping.update(tts.get("pronunciations") or {})
    for abbr in sorted(mapping, key=len, reverse=True):
        # Replacement via lambda: user-supplied pronunciation values are literal
        # text, not re.sub templates (a backslash or \1 must not be interpreted).
        text = re.sub(rf"(?<![\w/]){re.escape(abbr)}(?![\w/])",
                      lambda m, r=mapping[abbr]: r, text)
    return text


KOKORO_BATCH_SCRIPT = """
import json, sys
import soundfile as sf
from kokoro_onnx import Kokoro
model, voices_bin, voice, speed, scenes_file, outdir = sys.argv[1:7]
kokoro = Kokoro(model, voices_bin)
for i, text in enumerate(json.load(open(scenes_file))):
    if not text.strip():
        continue
    samples, sr = kokoro.create(text, voice=voice, speed=float(speed))
    sf.write(f"{outdir}/scene-{i:02d}-raw.wav", samples, sr)
print("kokoro: synthesized", flush=True)
"""


def find_kokoro():
    """Shared-venv python + kokoro model/voices, if installed."""
    py = PIPER_HOME / "piper-venv/bin/python"
    model = PIPER_HOME / "kokoro/kokoro-v1.0.onnx"
    voices = PIPER_HOME / "kokoro/voices-v1.0.bin"
    if not (py.is_file() and model.is_file() and voices.is_file()):
        return None
    probe = subprocess.run([str(py), "-c", "import kokoro_onnx, soundfile"],
                           capture_output=True)
    return (py, model, voices) if probe.returncode == 0 else None


def kokoro_batch(scenes: list[dict], tts: dict) -> bool:
    """Synthesize every scene's raw wav in one process (one model load)."""
    found = find_kokoro()
    if not found:
        return False
    py, model, voices = found
    texts = [spoken_text((s.get("narration") or "").strip(), tts) for s in scenes]
    scenes_file = OUT / "kokoro-texts.json"
    scenes_file.write_text(json.dumps(texts))
    voice = tts.get("kokoro_voice") or "af_heart"
    speed = str(tts.get("kokoro_speed") or 1.0)
    result = subprocess.run(
        [str(py), "-c", KOKORO_BATCH_SCRIPT, str(model), str(voices), voice, speed,
         str(scenes_file), str(OUT)])
    scenes_file.unlink(missing_ok=True)
    return result.returncode == 0


def find_piper() -> str | None:
    """Piper on PATH, or the shared install this skill sets up."""
    found = shutil.which("piper")
    if found:
        return found
    candidate = PIPER_HOME / "piper-venv/bin/piper"
    return str(candidate) if candidate.is_file() else None


def find_piper_model(configured: str) -> Path | None:
    """Configured model, else best local voice (prefer -high, then -medium)."""
    if configured and Path(configured).is_file():
        return Path(configured)
    for voices_dir in (Path("voices"), PIPER_HOME / "voices"):
        if not voices_dir.is_dir():
            continue
        models = sorted(voices_dir.glob("*.onnx"))
        for preference in ("-high", "-medium", ""):
            for model in models:
                if preference in model.stem:
                    return model
    return None


def synth(text: str, wav: Path, tts: dict) -> None:
    engine = (tts.get("engine") or "auto").lower()
    voice = tts.get("voice") or "en-us"
    speed = int(tts.get("speed") or 155)

    if engine in ("piper", "auto"):
        piper_bin = find_piper()
        model = find_piper_model(tts.get("piper_model") or "")
        if piper_bin and model:
            subprocess.run([piper_bin, "-m", str(model), "-f", str(wav)],
                           input=text.encode(), check=True)
            return
        if engine == "piper":
            sys.exit("tts.engine is 'piper' but no piper binary/voice model was found. "
                     "Install: python3 -m venv ~/.local/share/socket-demo-video/piper-venv && "
                     "~/.local/share/socket-demo-video/piper-venv/bin/pip install piper-tts; then "
                     "download a voice with python -m piper.download_voices en_US-ryan-high "
                     "--download-dir ~/.local/share/socket-demo-video/voices")
    if engine in ("espeak-ng", "auto") and shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(wav), text], check=True)
        return
    if engine == "espeak-ng":
        # An explicitly requested engine must never silently become a different
        # voice on another machine.
        sys.exit("tts.engine is 'espeak-ng' but espeak-ng is not installed. "
                 "Install it (apt install espeak-ng / brew install espeak) or set tts.engine: auto.")
    if shutil.which("say"):
        aiff = wav.with_suffix(".aiff")
        subprocess.run(["say", "-r", str(max(120, min(speed, 220))), "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                        "-ar", "22050", "-ac", "1", str(wav)], check=True)
        aiff.unlink(missing_ok=True)
        return
    sys.exit("No TTS engine found. Install piper or espeak-ng (macOS: built-in say works).")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes, config = load_scenes()
    if not scenes:
        sys.exit("out/scenes.json has no scenes — run src/generate_content.py first.")
    tts = config.get("tts") or {}

    engine = (tts.get("engine") or "auto").lower()
    known_engines = {"auto", "kokoro", "piper", "espeak-ng", "say"}
    if engine not in known_engines:
        sys.exit(f"Unknown tts.engine '{engine}'. Use one of: {', '.join(sorted(known_engines))}.")

    # Invalidate previous outputs up front: a leftover scene-NN-raw.wav would
    # resurrect OLD narration text after a demo.yaml edit, and a stale
    # voice.wav/scene_durations.json would silently pace the recorder from a
    # previous run's audio if this run aborts.
    for stale in OUT.glob("scene-*.wav"):
        stale.unlink()
    (OUT / "voice.wav").unlink(missing_ok=True)
    (OUT / "scene_durations.json").unlink(missing_ok=True)

    # Kokoro (most natural local voice, Apache-2.0) synthesizes all scenes in one
    # batch when available; anything it didn't produce falls back per scene.
    if engine in ("kokoro", "auto"):
        if kokoro_batch(scenes, tts):
            print("Narration engine: Kokoro")
        else:
            # A partial batch may have written some raws — remove them so the
            # fallback engine voices EVERY scene (never two voices in one video).
            for partial in OUT.glob("scene-*-raw.wav"):
                partial.unlink()
            if engine == "kokoro":
                sys.exit("tts.engine is 'kokoro' but kokoro-onnx or its model files are missing. "
                         "Install: ~/.local/share/socket-demo-video/piper-venv/bin/pip install "
                         "kokoro-onnx soundfile; then download kokoro-v1.0.onnx and voices-v1.0.bin "
                         "into ~/.local/share/socket-demo-video/kokoro/")

    parts: list[Path] = []
    for i, scene in enumerate(scenes):
        narration = (scene.get("narration") or "").strip()
        target = float(scene.get("duration_seconds") or 12)
        raw = OUT / f"scene-{i:02d}-raw.wav"
        part = OUT / f"scene-{i:02d}.wav"
        if narration and raw.is_file():
            spoken = ffprobe_duration(raw)
        elif narration:
            synth(spoken_text(narration, tts), raw, tts)
            spoken = ffprobe_duration(raw)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "anullsrc=r=22050:cl=mono", "-t", "0.1", str(raw)], check=True)
            spoken = 0.0
        pad = max(0.0, target - spoken)
        if spoken > target + 0.5:
            print(f"WARN: scene '{scene.get('id', i)}' narration runs {spoken:.1f}s "
                  f"but the scene is {target:.1f}s — visuals may drift. "
                  f"Shorten the narration or raise duration_seconds.")
        # Normalize every part to the same format so concat is safe across engines.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-af", f"apad=pad_dur={pad:.3f}",
                        "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", str(part)], check=True)
        raw.unlink(missing_ok=True)
        parts.append(part)

    concat_list = OUT / "concat.txt"
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in parts))
    voice = OUT / "voice.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c", "copy", str(voice)], check=True)

    # Emit the ACTUAL per-scene durations so the recorder can pace the visuals to
    # the narration (a scene whose narration overflows its planned slot gets its
    # real length here). run_pipeline.sh runs audio before recording for this.
    manifest = [
        {"id": scene.get("id", str(i)), "seconds": round(ffprobe_duration(part), 3)}
        for i, (scene, part) in enumerate(zip(scenes, parts))
    ]
    (OUT / "scene_durations.json").write_text(json.dumps(manifest, indent=2))

    total = ffprobe_duration(voice)
    planned = sum(float(s.get("duration_seconds") or 12) for s in scenes)
    print(f"Wrote {voice} — {total:.1f}s from {len(parts)} scenes (planned {planned:.0f}s)")
    print(f"Wrote {OUT / 'scene_durations.json'} for recorder pacing")


if __name__ == "__main__":
    main()
