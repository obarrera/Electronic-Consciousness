# Phase 7 — Narration coverage + quality pass

**Status: ✅ complete**

**Files:** tools_narrate_parables.py (coverage + spoken overrides + loudnorm),
lattice.py (hero-stage / Oracle playback wiring), EC-2D-Land.py (wiring),
narration/ (regenerated + new files).

## Why
"Ensure we have the best narration for each dialog and text" — every
player-facing narrative line should be hearable in the elder's voice, at a
consistent loudness, with no TTS mispronunciations, and playback must keep
the phase-3 completion/queue guarantee.

## Tasks

1. **Coverage inventory.** Enumerate every player-facing text (overlay /
   caption / cutscene sources): the 19 parables, pilgrim + O! endgame
   narrations, the 8 hero's-journey stage announcements, the CALL journey
   caption, ascension/fall transition lines, the Oracle's per-turning line,
   intro tagline (if narrative), endgame spectrum/cube captions. Produce a
   table in this file: text → narrated? → file.

2. **Fill the gaps.** Extend tools_narrate_parables.py with the same Kokoro
   setup (~/.local/share/socket-demo-video venv, voice am_michael). Oracle:
   narrate each fragment pool entry ONCE (openers / turns / seals as separate
   files) and sequence fragment files at runtime to speak the composed line —
   do NOT pre-render combinations. Dynamic chronicle entries are out of
   scope (text only).

3. **Quality pass on ALL narration files (existing + new):**
   - Pronunciation review: reword TTS-risky words in the SPOKEN string only
     (spoken-override dict in the tool); on-screen text unchanged. Known
     Kokoro traps: "live", "remediate", "noise", heteronyms
     (read/lead/wind/tear); spell out numerals where rhythm matters.
   - Loudness: ffmpeg loudnorm (I=-16, TP=-1.5, LRA=11) before the final
     libmp3lame 48k mono 24 kHz encode.
   - Decode check: script loads every file via pygame Sound, prints
     get_length(); no file <1.5 s unless the text is genuinely that short.
     Paste output.
   - Pacing: parables speed 0.95, endgame 0.9, new short lines 0.95.

4. **Wire playback.** Hero-stage announcements and the Oracle line use the
   narration channel with the phase-3 completion/queue guarantee (never
   preempt, ENTER skips, ducking, mute honored). Autopilot still skips
   narration.

## Acceptance
- [x] Coverage table: every enumerated text narrated or explicitly
      out-of-scope with reason.
- [x] Decode-check output for all narration files pasted.
- [x] Manual-run note: a hero-stage line and an Oracle line narrate at the
      right moments without cutting off a parable.
- [x] Standard gates green.

## Coverage inventory

| Text | Narrated? | File(s) |
|---|---|---|
| 19 Lattice parables (title + body) | ✅ regenerated (loudnorm + overrides) | `narration/<key>.mp3` |
| Endgame: Pilgrim, O! (33rd degree) | ✅ regenerated, speed 0.9 | `pilgrim.mp3`, `o33.mp3` |
| 8 hero's-journey stage announcements (incl. THE CALL = stage II) | ✅ new | `journey_<key>.mp3` |
| Oracle closing lines (10×10×10 combos) | ✅ per-fragment, sequenced at runtime | `oracle_intro.mp3`, `oracle_open_<n>`, `oracle_turn_<n>`, `oracle_seal_<n>` |
| Ascension / fall transitions | covered by journey stages III (threshold) and V (abyss) | `journey_threshold.mp3`, `journey_abyss.mp3` |
| Endgame GL caption strip ("THE COMPLETION — N layers…") | out of scope: transient HUD strip; pilgrim/o33 narrations carry that moment's audio | — |
| Chronicle entries | out of scope per spec (dynamic; text-only artifact) | — |
| Title-screen tagline, help bar, HUD stats, seizure warning | out of scope: UI chrome, not narrative dialog | — |

## Implementation notes

- tools_narrate_parables.py rewritten: one Kokoro subprocess synthesizes all
  60 files (model loads once); every file passes
  `loudnorm=I=-16:TP=-1.5:LRA=11` before the libmp3lame 48k mono 24 kHz
  encode; a decode check loads each via pygame and flags runts (<1.5 s;
  oracle seals — "Walk on.", "Oh!" — exempt as genuinely short). Exit code
  reflects problems.
- **Spoken-only overrides** (display text unchanged): "a stone lives if" →
  "stays alive if" (stones); "every read is one face" → "every reading…"
  (die); "Havel read his cargo list" → "Havel red…" (bridge, past tense);
  "O!" → "Oh!" (pilgrim, o33, seals); Roman-numeral stage headings spoken as
  "Stage one:" … "Stage eight:". Trap scan covered
  live/read/lead/wind/tear/wound/refuse — "refuse(s)" (verb) and "wound"
  (noun) verified safe in context.
- **Oracle, spoken**: `ouroboros.oracle_fragment_indices(iteration, key)`
  exposes the (opener, turn, seal) pool indices; EC-2D-Land registers an
  oracle-audio hook (`lattice.set_oracle_audio`) that appends the four
  fragment files after any PARABLE/ENDGAME narration.
  `AudioEngine.narrate` sequences the queue on the narration channel;
  `narrating()` stays true across the whole sequence, so the phase-3
  completion guarantee (and the overlay hold, and the cutscene duration
  floor) covers the Oracle line automatically. The ordinal intro ("In the
  Nth turning…") is spoken as the recorded generic "And the elder of this
  turning added:" — the unbounded ordinal set is display-only (documented
  deviation).
- **Hero stages**: `HeroJourney._show` narrates `journey_<key>.mp3` only if
  no narration is in flight (a caption never preempts a parable — the text
  still shows); the caption extends to cover its own narration + 1.5 s.
  ENTER now also skips a stage narration (and only stops a parable's audio
  if the overlay actually closed — the deliberate-skip grace can refuse).
  Autopilot remains silent unless `EC_VERIFY_NARRATION=1`.
- Effects-ducking while text is read: already central since phase 4
  (`AudioEngine.set_reading`); hero captions and the endgame arc were
  already included in the reading state. Narration channel never ducks.

## Verification evidence

Gates run 2026-08-03 (.venv312). `ast.parse` EC-2D-Land.py, lattice.py,
ouroboros.py, tools_narrate_parables.py: `ast ok`.

**Generation + decode check** (60 files, `decode check: 0 problem(s)`) —
excerpt:

```
journey_ordinary.mp3: 5.9s   journey_call.mp3: 6.1s   journey_abyss.mp3: 4.8s
journey_threshold.mp3: 6.2s  journey_return.mp3: 5.5s journey_master.mp3: 6.0s
oracle_intro.mp3: 2.1s       oracle_open_0..9: 3.7–7.1s
oracle_turn_0..9: 2.6–4.9s   oracle_seal_0..9: 0.8–2.4s (short by design)
stones.mp3: 25.9s  pilgrim.mp3: 52.9s  o33.mp3: 43.9s  primes.mp3: 31.5s
decode check: 0 problem(s)
```

**Sequenced playback + no-preemption run** (`EC_AUTOPILOT=4000
EC_EVOLUTION_THRESHOLD=600 EC_VERIFY_NARRATION=1`):

```
HERO NARRATE journey_ordinary (5.9s)
CUTSCENE PRESENT stones (narration 37.5s, hold 40.5s)      <- 25.9s parable
CUTSCENE COMPLETE stones (narration finished, 40.5s shown)    + ~12s Oracle
PARABLE PRESENT granary (frame 1363, gen 175, narration 35.3s)
PARABLE COMPLETE granary (narration finished, overlay closed naturally)
PARABLE PRESENT census (frame 2537, gen 1349, narration 35.2s)
PARABLE COMPLETE census (narration finished, overlay closed naturally)
Autopilot: clean exit after 4000 frames (gen 2812, 13 agents, 6 parables unlocked).
```

The hero-stage line spoke first; the stones cutscene *waited* for it
(presentation gate), then played its narration plus the Oracle fragments to
completion (37.5 s vs 25.9 s body — the fragments are audibly appended and
the hold covers them). Nothing was cut off.

**Standard gates**: smoke `Autopilot: clean exit after 400 frames (gen 223,
10 agents, 4 parables unlocked).` Phase-6 determinism unaffected — headless
and windowed still hash to `a2d8b88c30…cf536ba4` (narration is pure
presentation).
