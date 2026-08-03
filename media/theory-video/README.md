# Theory video — source kit

`../electronic-consciousness-theory.mp4` is generated from `scenes.json` with
[slidecast](https://github.com/obarrera/slidecast), a generic JSON → slides →
local-TTS → ffmpeg video pipeline (Python stdlib + Pillow, no cloud services).

## Rebuild

```bash
git clone https://github.com/obarrera/slidecast
python3 slidecast/slidecast.py build scenes.json -n electronic-consciousness-theory
# → out/electronic-consciousness-theory.mp4
```

Requires Python 3.10+, Pillow, and ffmpeg; narration uses Kokoro TTS if
installed (see the slidecast README), otherwise macOS `say`.

## Contents

- `scenes.json` — the full script: eleven scenes tracing Plato's cave →
  Abbott's Flatland → the EC-2D-Land simulation → the dimensional-perception
  thesis, BC vs EC, and recursive simulation.
- `assets/` — EC-2D-Land screenshots composited into the game scenes.

Edit `scenes.json` (narration drives scene timing) and re-run the build to
re-cut the video.
