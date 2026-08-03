# Phase 4 — Positional audio in Spaceland

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
- [ ] Manual: rotate in place near shrine — hum pans L→R; walk away — fades.
- [ ] Autopilot (which traverses Spaceland) exits cleanly; no channel leaks
      (`pygame.mixer.get_busy` count stable across two ascend/fall cycles).
