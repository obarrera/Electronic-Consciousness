# Electronic Consciousness: The Theory Demo Video

Renders a branded Socket browser demo video to `out/final/electronic-consciousness-the-theory.mp4`.
Recording is READ-ONLY by design: URL navigation, link clicks, popups, scrolling,
and zooms only — never Settings, billing, or anything mutating.

## Run

```bash
bash run_pipeline.sh
```

(Behind Socket Firewall install enforcement: `sfw bash run_pipeline.sh`.)

## One-time setup on a new machine

1. **Login/profile prime** (live dashboard demos): `npm run prime` opens a headed
   Chrome on the demo org — sign in there once (email magic links must be PASTED
   into that window, not clicked in your mail client). The window closes itself.
   Re-run this any time the profile gets signed out (`npm run test-firewall`
   failing to reach the Events page, or any recording landing on a login screen,
   are both signs of this).
2. **Natural voice (Kokoro)**: see the skill's SKILL.md "Natural narration" section
   for the shared install; `tts.engine: auto` picks it up automatically.
3. **Socket Firewall testing** (Firewall videos only): create a `SOCKET_API_KEY`
   INSIDE the org whose Events page should show the traffic, with scopes
   `packages`, `entitlements:list`, and `security-policy:read` (the last one is
   undocumented but required — see "Testing Socket Firewall" below). Install
   [Docker](https://docs.docker.com/get-docker/) for `registry` mode (the
   default, most reliable mode) — `wrapper` mode works without Docker but needs
   the Firewall Enterprise binary, which auto-downloads on first use.

## Plan scenes from a site map

```bash
npm run map                 # crawls the demo org (read-only)
npm run map -- https://docs.socket.dev/docs   # or any Socket root
```

Writes `out/sitemap.json` + `out/sitemap.md` to pick real URLs (and, per
page, real button/tab text for `click` scenes) — no guessing nav labels.
Control how the crawl explores with a `crawl:` block in `demo.yaml`
(`seeds`, `max_pages`, `max_depth`, `include`/`exclude` regexes,
`capture_clickables`) — see `demo.yaml`'s schema notes in the skill's
`references/demo-yaml-schema.md`.

## Auto-build a click path (no walkthrough, no hand-written scenes)

```bash
python src/build_scenes_from_sitemap.py --merge demo.yaml
```

Picks a `video_type`-aware sequence of real pages (from `out/sitemap.json`
if you ran `npm run map`, else the bundled `assets/dashboard-sitemap.json`)
and varies the visuals automatically — scroll on list pages, zoom on detail
pages, and a highlight beat on a real on-screen control. Narration comes
back as `TODO narrate: ...` — fill it in against docs.socket.dev, then run
the pipeline.

## Record your own click path (manual walkthrough)

Prefer to just show Claude what you want in the video instead of writing
`scenes:` by hand? Drive the product yourself and replay it exactly:

```bash
npm run walkthrough                                   # opens a headed Chrome window
npm run walkthrough -- https://socket.dev/dashboard/org/socketdev-demo/alerts
```

Click **"Mark scene"** right before each beat you want in the final video,
do the click/scroll/navigation, then click **"Finish"** (or just close the
window). This writes `out/walkthrough/recorded.json`. Convert it straight
into `demo.yaml`:

```bash
python src/walkthrough_to_scenes.py --merge demo.yaml
```

This replaces `demo.yaml`'s `scenes:` list with exactly what you clicked
through (a `.bak` backup is written first) and fills in placeholder
`narration: "TODO narrate: ..."` fields for you to write — the rest of the
pipeline replays your recorded clicks/scrolls/navigations unchanged.

## Pick a voice before rendering

```bash
python src/voice_preview.py
```

Synthesizes a short sample line in every TTS engine/voice actually installed
(Kokoro, Piper, espeak-ng, `say`) to `out/voice-preview/*.wav` and prints the
exact playback command for each, so you can pick one by ear before rendering
the full video. Set the winner via `tts.engine`/`tts.kokoro_voice`/`tts.piper_model`.

## Testing Socket Firewall (live proof)

For a Firewall video that should show it actually working instead of narrating
over docs — verified end-to-end (confirmed live in a real org's Events page):

```bash
export SOCKET_API_KEY=<key>   # create INSIDE the org whose Events page should show this
npm run test-firewall              # registry mode (default): real Warn + Monitor
npm run test-firewall wrapper       # wrapper mode: real Block, but seen intermittent
```

Token needs scopes `packages`, `entitlements:list`, AND `security-policy:read` —
that last one is undocumented but required for real enforcement (without it,
every install shows "ignore" in Events regardless of severity). `registry` mode
runs Socket's official registry-mode Docker image locally (needs Docker, pulls
the image automatically) and produces real Warn/Monitor. `wrapper` mode
auto-downloads the Firewall Enterprise binary and produces real Block for
Critical CVEs on npm, but has shown intermittent flakiness — re-verify it's
working (the script's canary check will warn if it isn't) before relying on it
for a recording.

Run either a few minutes before recording, then filter the Events page to the
last hour (`?tp=1h`), 24 hours (`?tp=24h`), or 7 days (`?tp=7d`) to film it —
needs `demo_mode: live` for the Events scene. See SKILL.md for the full
findings, verified package list, and known product gaps.

## Outputs

- `out/narration.md`, `out/storyboard.md`, `out/captions.srt`
- `out/audio/voice.wav` + `out/audio/scene_durations.json` (paces the recording)
- `out/recordings/browser-recording.webm` + `out/recordings/timing.json` (sync proof)
- `out/cards/intro.png`, `out/cards/outro.png` (branded title cards)
- `out/final/electronic-consciousness-the-theory.mp4`

## Editing the video

`demo.yaml` is the source of truth — scenes support `goto`/`click`/`highlight`/
`scroll`/`wait`, plus `scroll_pixels`, `zoom` (e.g. 1.25), `zoom_selector`/
`zoom_text`, and per-scene `callout`. Narration must avoid abbreviations (the
audio stage maps any stragglers for TTS only). After edits, re-run the pipeline.

Do not commit or share `auth/` or the browser profile. Review the final MP4 before
sharing externally to confirm no secrets or customer data are visible.
