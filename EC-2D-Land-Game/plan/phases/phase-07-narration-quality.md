# Phase 7 — Narration coverage + quality pass

**Status: planned — queued after phase 6 (final batch item; batch closed)**

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
- [ ] Coverage table: every enumerated text narrated or explicitly
      out-of-scope with reason.
- [ ] Decode-check output for all narration files pasted.
- [ ] Manual-run note: a hero-stage line and an Oracle line narrate at the
      right moments without cutting off a parable.
- [ ] Standard gates green.
