#!/usr/bin/env python3
"""Generate spoken narration for the in-game Lattice parables.

Synthesizes every entry in lattice.PARABLES to narration/<key>.ogg using the
Kokoro-82M local TTS (expected in ~/.local/share/socket-demo-video, as
installed for the book's video pipeline), encoding with ffmpeg. Re-run after
editing parable texts. Requires: the shared piper-venv with kokoro-onnx, and
ffmpeg on PATH.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lattice import PARABLES, ENDGAME_PARABLES  # noqa: E402

HOME = Path.home() / ".local/share/socket-demo-video"
PY = HOME / "piper-venv/bin/python"
MODEL = HOME / "kokoro/kokoro-v1.0.onnx"
VOICES = HOME / "kokoro/voices-v1.0.bin"
VOICE = "am_michael"  # the elder at the western edge
OUT = Path(__file__).parent / "narration"


def main():
    if not (PY.is_file() and MODEL.is_file() and VOICES.is_file()):
        sys.exit("Kokoro not found — see the book video pipeline's setup notes.")
    OUT.mkdir(exist_ok=True)
    # The milestone parables plus the endgame arc's two narrations (the
    # Pilgrim's parable and O! at the 33rd degree, spoken slightly slower
    # for gravity).
    entries = [(key, title, text, 0.95)
               for key, title, trigger, cond, text in PARABLES]
    entries += [(key, title, text, 0.9) for key, title, text in ENDGAME_PARABLES]
    for key, title, text, speed in entries:
        ogg = OUT / f"{key}.mp3"
        spoken = f"{title}. {text}"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        script = (
            "import soundfile as sf\n"
            "from kokoro_onnx import Kokoro\n"
            f"k = Kokoro({str(MODEL)!r}, {str(VOICES)!r})\n"
            f"s, r = k.create({spoken!r}, voice={VOICE!r}, speed={speed!r})\n"
            f"sf.write({wav!r}, s, r)\n"
        )
        subprocess.run([str(PY), "-c", script], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
                        "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "48k", str(ogg)], check=True)
        os.unlink(wav)
        print(f"{ogg.name}: {ogg.stat().st_size // 1024} KB")
    print(f"Done: {len(entries)} narrations in {OUT}/")


if __name__ == "__main__":
    main()
