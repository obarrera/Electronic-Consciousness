#!/usr/bin/env python3
"""Generate spoken narration for every narrative text in EC-2D-Land.

Coverage (phase 7):
  * the 19 Lattice parables            -> narration/<key>.mp3
  * the 2 endgame narrations           -> narration/pilgrim.mp3, o33.mp3
  * the 8 hero's-journey stage lines   -> narration/journey_<key>.mp3
  * the Oracle fragment pools          -> narration/oracle_intro.mp3,
    (each fragment ONCE — the game        oracle_open_<n>.mp3,
    sequences them at runtime to          oracle_turn_<n>.mp3,
    speak any composed line)              oracle_seal_<n>.mp3

Quality pass: spoken-only pronunciation overrides (on-screen text is never
changed), EBU R128 loudness normalization (ffmpeg loudnorm I=-16 TP=-1.5
LRA=11), then libmp3lame 48k mono 24 kHz. A decode check loads every file
via pygame and prints its duration.

Uses the Kokoro-82M local TTS (voice am_michael — the elder at the western
edge) from ~/.local/share/socket-demo-video, and ffmpeg. All synthesis runs
in ONE subprocess so the model loads once.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402
pygame.init()
from lattice import PARABLES, ENDGAME_PARABLES, JOURNEY_STAGES, SUMMATION  # noqa: E402
import ouroboros  # noqa: E402

HOME = Path.home() / ".local/share/socket-demo-video"
PY = HOME / "piper-venv/bin/python"
MODEL = HOME / "kokoro/kokoro-v1.0.onnx"
VOICES = HOME / "kokoro/voices-v1.0.bin"
VOICE = "am_michael"  # the elder at the western edge
OUT = Path(__file__).parent / "narration"

# Spoken-only pronunciation overrides — the display text never changes.
# Kokoro heteronym traps found by scan: "lives" (verb), "read" (noun/past),
# the bare exclamation "O!", and Roman-numeral stage headings.
SPOKEN_OVERRIDES = [
    ("a stone lives if", "a stone stays alive if"),           # stones
    ("every read is one face", "every reading is one face"),  # die
    ("Havel read his cargo list", "Havel red his cargo list"),  # bridge (past tense)
    ("O!", "Oh!"),                                            # pilgrim, o33, seals
    (": O.", ": Oh."),                                        # o33 title
]

_ROMAN_SPOKEN = {"I.": "Stage one:", "II.": "Stage two:",
                 "III.": "Stage three:", "IV.": "Stage four:",
                 "V.": "Stage five:", "VI.": "Stage six:",
                 "VII.": "Stage seven:", "VIII.": "Stage eight:"}


def spoken(text):
    for old, new in SPOKEN_OVERRIDES:
        text = text.replace(old, new)
    return text


def build_entries():
    """(filename-stem, spoken text, speed) for every narration file."""
    entries = []
    for key, title, trigger, cond, text in PARABLES:
        entries.append((key, spoken(f"{title}. {text}"), 0.95))
    for key, title, text in ENDGAME_PARABLES:
        entries.append((key, spoken(f"{title}. {text}"), 0.9))
    s_key, s_title, s_beats = SUMMATION
    s_text = " ".join(b[1] for b in s_beats)
    entries.append((s_key, spoken(f"{s_title}. {s_text}"), 0.9))
    for key, stage, line in JOURNEY_STAGES:
        head = stage
        for rn, sp in _ROMAN_SPOKEN.items():
            if head.startswith(rn):
                head = head.replace(rn, sp, 1)
                break
        entries.append((f"journey_{key}",
                        spoken(f"{head.title()}. {line}"), 0.95))
    # The Oracle's fragments, one file each; the game sequences them.
    entries.append(("oracle_intro", "And the elder of this turning added:",
                    0.95))
    for i, frag in enumerate(ouroboros._OPENERS):
        entries.append((f"oracle_open_{i}", spoken(frag + ","), 0.95))
    for i, frag in enumerate(ouroboros._TURNS):
        entries.append((f"oracle_turn_{i}", spoken(frag + "."), 0.95))
    for i, frag in enumerate(ouroboros._SEALS):
        entries.append((f"oracle_seal_{i}", spoken(frag), 0.95))
    return entries


def synthesize(entries):
    """All Kokoro synthesis in one subprocess (model loads once)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ec-narrate-"))
    manifest = [(str(tmpdir / f"{stem}.wav"), text, speed)
                for stem, text, speed in entries]
    script = (
        "import json, sys\n"
        "import soundfile as sf\n"
        "from kokoro_onnx import Kokoro\n"
        f"k = Kokoro({str(MODEL)!r}, {str(VOICES)!r})\n"
        f"for wav, text, speed in json.loads(sys.argv[1]):\n"
        f"    s, r = k.create(text, voice={VOICE!r}, speed=speed)\n"
        "    sf.write(wav, s, r)\n"
        "    print('synth', wav.rsplit('/', 1)[-1], flush=True)\n"
    )
    subprocess.run([str(PY), "-c", script, json.dumps(manifest)], check=True)
    return tmpdir


def encode(entries, tmpdir):
    """Loudness-normalize and encode every wav to narration/<stem>.mp3."""
    for stem, text, speed in entries:
        wav = tmpdir / f"{stem}.wav"
        mp3 = OUT / f"{stem}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "48k",
             str(mp3)], check=True)
        os.unlink(wav)
        print(f"{mp3.name}: {mp3.stat().st_size // 1024} KB")


def decode_check():
    """Load every narration file via pygame; print durations; flag runts."""
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except pygame.error as exc:
        print(f"decode check: no mixer ({exc}) — lengths via Sound only")
    problems = 0
    for mp3 in sorted(OUT.glob("*.mp3")):
        try:
            length = pygame.mixer.Sound(str(mp3)).get_length()
        except pygame.error as exc:
            print(f"  DECODE FAIL {mp3.name}: {exc}")
            problems += 1
            continue
        # oracle seals are genuinely short lines ("Walk on.", "Oh!")
        short_ok = mp3.name.startswith("oracle_seal_")
        flag = "" if (length >= 1.5 or short_ok) else "  <-- RUNT"
        print(f"  {mp3.name}: {length:.1f}s{flag}")
        if flag:
            problems += 1
    print(f"decode check: {problems} problem(s)")
    return problems


def main():
    if not (PY.is_file() and MODEL.is_file() and VOICES.is_file()):
        sys.exit("Kokoro not found — see the book video pipeline's setup notes.")
    OUT.mkdir(exist_ok=True)
    entries = build_entries()
    if len(sys.argv) > 1:      # regenerate only the named stems
        want = set(sys.argv[1:])
        entries = [e for e in entries if e[0] in want]
        if not entries:
            sys.exit(f'no entries match {sorted(want)}')
    tmpdir = synthesize(entries)
    encode(entries, tmpdir)
    print(f"Done: {len(entries)} narrations in {OUT}/")
    sys.exit(1 if decode_check() else 0)


if __name__ == "__main__":
    main()
