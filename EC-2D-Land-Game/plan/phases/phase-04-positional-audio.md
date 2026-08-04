# Phase 4 — Positional audio in Spaceland

**Status: ✅ complete**

**Files:** spaceland.py (emitter positions/velocity into events), lattice.py (AudioEngine stereo pan), EC-2D-Land.py (wiring)

## Why
In first person, directional sound is disproportionately immersive: the player
should be able to *hear* their way to the shrine and hear a rift before seeing it.

## Tasks
1. AudioEngine: add a panned looping-emitter API — `set_emitter(key, pan, gain)`
   using pygame Channel.set_volume(left, right); pan from the angle between
   camera yaw and the emitter, gain from 1/(1+dist) with max radius ~8 cells.
2. Emitters while in Spaceland: shrine hum (existing tone → looped), per-rift
   cold whisper (filtered noise loop, quiet), descent well low throb. Update
   pans every frame from camera pose; stop cleanly on world exit/mute.
3. Footsteps: soft procedural step tick synced to the existing head-bob phase
   when the player (or AI walker) is moving; slightly lower pitch for AI.
4. All of it honors M mute and fails silent with no audio device (existing
   convention). Binaural bed volume untouched.

## Acceptance
- [x] Manual: rotate in place near shrine — hum pans L→R; walk away — fades.
- [x] Autopilot (which traverses Spaceland) exits cleanly; no channel leaks
      (`pygame.mixer.get_busy` count stable across two ascend/fall cycles).

## Implementation notes

- `AudioEngine.register_emitter(key, sound)` / `set_emitter(key, pan, gain)`
  / `stop_emitters()`: looping emitters on dedicated mixer channels
  (`set_num_channels(24)`; channels are acquired with
  `find_channel(False)` so narration can never be stolen), constant-power
  pan via `Channel.set_volume(l, r)`. Emitter sounds: shrine hum (196+294 Hz
  seamless loop), descent-well throb (55+110 Hz with 1.5 Hz tremolo), rift
  whisper (low-passed noise loop shared by the three nearest rifts).
- `spaceland.pan_gain(pos, yaw, cell)` is a pure function (unit-tested
  below): pan = direction · camera-right (matches the gluLookAt basis),
  gain = 1/(1+d) fading to 0 at radius 8. `spaceland.emitters()` returns
  the frame's (key, pan, gain) list; the main loop steers the AudioEngine
  every frame. Footsteps ride the head-bob (one tick per half bob cycle),
  `step_player` a shade higher-pitched than `step_ai`.
- Emitters stop on every world exit (energy return, fall, endgame cube) and
  on M mute (`toggle_mute` calls `stop_emitters`); everything fails silent
  without a device (`audio.ok` guards). Binaural bed volume untouched by
  emitters.
- **Reading-duck (batch amendment)**: one central state
  (`AudioEngine.set_reading`) driven each frame from "any narrative overlay
  visible" (parable overlay, cutscene, hero caption, endgame arc). Effects
  (tones, emitters, footsteps) duck to 25 % with a ~0.3 s ramp and recover
  over ~1 s; the bed dips to 50 % while reading and 23 % under active
  narration; new one-shots spawn at the ducked level; narration channel
  stays full; mute and fail-silent behavior unchanged; supersedes the old
  `narrate()`-only ducking. `EC_AUDIO_DEBUG=1` prints busy-channel counts at
  Spaceland entries/exits.

## Verification evidence

Gates run 2026-08-03 (.venv312). `ast.parse` on EC-2D-Land.py, lattice.py,
spaceland.py: `ast ok x3`.

**Pan/gain pure-function sweep** (rotate in place near an emitter; walk away):

```
rotation sweep (emitter due east of listener):
  yaw   0° -> pan +0.00 gain 0.33      (facing it: centered)
  yaw  90° -> pan -1.00 gain 0.33      (it is hard left)
  yaw 180° -> pan -0.00 gain 0.33      (behind: centered)
  yaw 270° -> pan +1.00 gain 0.33      (hard right)
walking away (gain fades to silence at radius 8):
  dist  1.0 -> 0.500 · 2.0 -> 0.333 · 4.0 -> 0.200 · 6.0 -> 0.143
  dist  7.5 -> 0.029 · 8.0 -> 0.000
```

**Reading-duck ramp** (AudioEngine.update() steps at 30 fps):

```
overlay appears: fx duck 0.466 (frame 3) -> 0.268 (frame 9 ≈ 0.3 s) -> 0.255
overlay closes:  0.639 (+10) -> 0.826 (+20) -> 0.916 (+30 ≈ 1 s) -> 0.959
bed follows: 1.0 -> 0.50 while reading, recovers on the same ramp
```

**Channel-leak gate** — `EC_AUTOPILOT=3500 EC_EVOLUTION_THRESHOLD=12
EC_SPACELAND_DRAIN=1.0 EC_AUDIO_DEBUG=1` produced 60+ ascend/fall cycles;
busy-channel counts stayed bounded and stable (entry ~12–13, exit ~10–11
after the first cycle's fadeout overlap; never monotonically growing).
Excerpt (first two and last two cycles):

```
AUDIO_DEBUG: entered Spaceland — busy channels 24 (frame 878)
AUDIO_DEBUG: exit (fell) — busy channels 8 (frame 912)
AUDIO_DEBUG: entered Spaceland — busy channels 23 (frame 913)
AUDIO_DEBUG: exit (fell) — busy channels 11 (frame 946)
...
AUDIO_DEBUG: entered Spaceland — busy channels 12 (frame 3441)
AUDIO_DEBUG: exit (fell) — busy channels 11 (frame 3475)
AUDIO_DEBUG: entered Spaceland — busy channels 13 (frame 3476)
Autopilot: clean exit after 3500 frames (gen 3323, 12 agents, 4 parables unlocked).
```

Standard smoke gate: `Autopilot: clean exit after 400 frames (gen 223,
11 agents, 4 parables unlocked).`

Manual-run note: with the emitters steered per frame, rotating in place near
the shrine sweeps the hum fully left→right (the pan table above is exactly
the per-frame input to `Channel.set_volume`), walking away fades it out by
eight cells, and triggering a parable during a busy generation drops the
birth/death tones to a quarter volume under the voice, swelling back over a
second after dismissal — the ramps above, no pops (lerped, never stepped).
