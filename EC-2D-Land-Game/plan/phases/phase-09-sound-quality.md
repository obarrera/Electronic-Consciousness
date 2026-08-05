# Phase 9 — Sound quality: declick, mix, Hemi-Sync shimmer

**Status: awaiting verification**

**Files:** lattice.py (AudioEngine synthesis + playback), EC-2D-Land.py (cold tone params)

## Why
Every one-shot effect ended with an audible click/pop, and busy generations
stacked identical tones into a blare that fought the binaural bed. The bed
(Hemi-Sync / Gateway-style theta drone) is the sound identity of the game;
the effects should sit *inside* it, not cut across it.

Root causes found:
1. **Truncated tails** — `_tone()` shaped with `exp(-decay·t)` but the buffer
   simply ended mid-decay (cold ended at ~2 % full scale, parable/prime ~1 %):
   a hard step to zero at the end of every effect → the pop.
2. **Volume set after play** — `play()` started the sound then set channel
   volume, so the first mixer buffer played at 100 %; a `vol=0.10` birth
   opened 10× too loud for ~12 ms.
3. **Voice pileups** — five same-generation deaths spawned five overlapping
   identical tones (phasing, loudness spikes). No cooldown, no voice cap.
4. **Harsh timbre** — footsteps were raw 950/1400 Hz sine beeps; every tone
   had an instant attack. Alarm-like against a theta drone.

## Tasks
1. `_tone()`: half-cosine attack (per-tone, 8–60 ms), exponential decay
   finished by a raised-cosine tail forced to exactly zero; partial-sum
   normalization (no clipping regardless of chord size); stereo Hemi-Sync
   shimmer — right ear runs `beat` Hz (default 6.1, matching the theta bed)
   above the left.
2. `_tick()`: footsteps become low-passed noise taps (player brighter,
   AI walker duller), same declick envelope, vol 0.05.
3. `play()`: per-tone cooldown table + two-voice cap per tone name; channel
   volume set BEFORE playback via `find_channel(False)` (never steals
   narration/emitter channels).
4. Retuned mix: all one-shot volumes down ~25–50 % (birth/death 0.16,
   parable 0.18, ascend 0.20, train 0.07, prime 0.05, cold 0.18); slow
   Gateway-style swells on the narrative tones (ascend 60 ms, parable 30 ms).
5. Gateway-style masking bed: a very quiet (0.05) heavily low-passed noise
   loop with an 8 s swell under the ambience; follows the bed duck and mute.
6. Mixer buffer 512 → 1024 (headroom against underrun crackle with
   24 channels + GL on one thread; +12 ms latency).

## Acceptance
- [ ] Every generated tone's first and last samples are exactly 0; no
      sample reaches int16 full scale.
- [ ] Same-frame double-play of one tone yields one voice; paced repeats
      (≥ cooldown apart) all play.
- [ ] One-shots spawn at the current duck level with volume set pre-play.
- [ ] Emitter register/steer/stop lifecycle unchanged; mute silences the
      surf bed too.
- [ ] Standard smoke gate passes.

## Verification evidence (implementer's run, 2026-08-05, .venv312)

`ast ok` × 2 (EC-2D-Land.py, lattice.py).

**Tone buffer sweep** (dummy audio driver; abs sample values of 32767):

```
        tone  old-end-amp   new-start  new-end   new-peak
      ascend          60          0        0      5881
       birth          20          0        0      4812
        cold         602          0        0      5229
       death          16          0        0      4881
     parable         274          0        0      4960
       prime         249          0        0      1560
     step_ai         106          0        0      1063
 step_player          46          0        0       957
       train          41          0        0      2103
```

**Playback guards**: double-play → 1 voice; 3 distinct tones → 3 voices;
5 steps paced 120 ms apart → 5 voices; ducked spawn opens at 0.25.
**Emitters**: register/steer/stop clean; surf bed live and mute-aware.

**Standard smoke gate**: `Autopilot: clean exit after 400 frames (gen 399,
13 agents, 2 parables unlocked).`

**Channel-leak gate caveat**: `EC_AUTOPILOT=3500 EC_EVOLUTION_THRESHOLD=12`
under `SDL_VIDEODRIVER=dummy` segfaults at the Spaceland GL transition —
**pre-existing on unmodified main** (verified via stash), an environment
limit (GL under macOS dummy video), not a regression. Verifier should run
the leak gate windowed as phase 4 did.
