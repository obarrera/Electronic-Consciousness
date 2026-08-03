#!/usr/bin/env python3
"""Synthesize one short sample line in every voice this machine can actually
produce, so an operator can pick a voice by listening instead of guessing from
a name, or running the full pipeline once per candidate. Writes
out/voice-preview/<engine>-<voice>.wav and prints how to play each one.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("out/voice-preview")
SAMPLE = "This is the Socket demo narration voice. Known vulnerability, reachable, block, warn, monitor."

PIPER_HOME = Path.home() / ".local/share/socket-demo-video"


def kokoro_available():
    py = PIPER_HOME / "piper-venv/bin/python"
    model = PIPER_HOME / "kokoro/kokoro-v1.0.onnx"
    voices = PIPER_HOME / "kokoro/voices-v1.0.bin"
    if not (py.is_file() and model.is_file() and voices.is_file()):
        return None
    probe = subprocess.run([str(py), "-c", "import kokoro_onnx, soundfile"], capture_output=True)
    return (py, model, voices) if probe.returncode == 0 else None


def render_kokoro(py, model, voices, voice, out_wav: Path) -> bool:
    script = (
        "import soundfile as sf\n"
        "from kokoro_onnx import Kokoro\n"
        f"kokoro = Kokoro({str(model)!r}, {str(voices)!r})\n"
        f"samples, sr = kokoro.create({SAMPLE!r}, voice={voice!r}, speed=1.0)\n"
        f"sf.write({str(out_wav)!r}, samples, sr)\n"
    )
    result = subprocess.run([str(py), "-c", script])
    return result.returncode == 0


def piper_voices() -> list[Path]:
    found = []
    for voices_dir in (Path("voices"), PIPER_HOME / "voices"):
        if voices_dir.is_dir():
            found.extend(sorted(voices_dir.glob("*.onnx")))
    return found


def render_piper(model: Path, out_wav: Path) -> bool:
    piper_bin = shutil.which("piper") or str(PIPER_HOME / "piper-venv/bin/piper")
    if not Path(piper_bin).is_file() and not shutil.which("piper"):
        return False
    result = subprocess.run([piper_bin, "-m", str(model), "-f", str(out_wav)],
                             input=SAMPLE.encode())
    return result.returncode == 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    kokoro = kokoro_available()
    if kokoro:
        py, model, voices = kokoro
        for voice in ("af_heart", "am_michael", "af_bella"):
            out_wav = OUT / f"kokoro-{voice}.wav"
            if render_kokoro(py, model, voices, voice, out_wav):
                made.append(out_wav)
    else:
        print("Kokoro not installed — see SKILL.md 'Natural narration' for setup.")

    for model in piper_voices():
        out_wav = OUT / f"piper-{model.stem}.wav"
        if render_piper(model, out_wav):
            made.append(out_wav)
    if not piper_voices():
        print("No Piper voices found — see SKILL.md 'Piper setup' for setup.")

    # Only advertise samples that were actually produced — a failed synth must
    # not print a playback command for a nonexistent file (or mask "no engine
    # produced a sample" by padding `made`).
    if shutil.which("espeak-ng"):
        out_wav = OUT / "espeak-ng-en-us.wav"
        result = subprocess.run(["espeak-ng", "-v", "en-us", "-w", str(out_wav), SAMPLE])
        if result.returncode == 0 and out_wav.is_file():
            made.append(out_wav)

    if shutil.which("say"):
        aiff = OUT / "say-default.aiff"
        result = subprocess.run(["say", "-o", str(aiff), SAMPLE])
        if result.returncode == 0 and aiff.is_file():
            made.append(aiff)

    if not made:
        sys.exit("No TTS engine produced a sample. Install Kokoro, Piper, or espeak-ng.")

    print(f"\nWrote {len(made)} sample(s) to {OUT}/ — play them to pick a voice:")
    for f in made:
        print(f"  afplay {f}" if sys.platform == "darwin" else f"  aplay {f}")
    print("\nSet the winner in demo.yaml: tts.engine + tts.kokoro_voice (or tts.piper_model), then re-run the pipeline.")


if __name__ == "__main__":
    main()
