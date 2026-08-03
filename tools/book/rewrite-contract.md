# Rewrite Contract — Electronic Consciousness, Second Edition

Every chapter rewrite MUST obey all of these rules. They implement the external review's
findings (2026-08) and the author's direction: *a visionary speculative manifesto on
consciousness and reality — a speculative philosophy of AI consciousness informed by
neuroscience, AI, quantum computing, and mythology — never a proven scientific theory.*

## 1. Framing and verb discipline

- The work is **speculative philosophy**. Any sentence that asserts an undemonstrated
  mechanism or benefit in the indicative ("X enhances Y", "X will optimize Y", "X allows
  EC to...") must be rewritten as explicit speculation ("one might design...", "we
  imagine...", "if X were true, then...") or as a falsifiable question (see rule 2).
- Established science may be stated plainly — but only what is actually established, with
  a citation marker (rule 4).
- One epistemic-status blockquote per file, at the top (keep the existing one, tightened).
  DELETE all other repeated disclaimers — the discipline must live in the prose itself,
  not in apologies around it.

## 2. Benefits → falsifiable questions

- Every "Benefits of X" / "Advantages" / "Practical Example" list of hypothetical wins is
  REPLACED by a section titled **"Open Questions and Falsifiable Tests"** (or folded into
  one). Each former benefit becomes:
  a) a question ("Would a φ-weighted allocation outperform a learned allocation?"),
  b) a sketch of the test (system, metric, baseline, controls), and
  c) what result would count as falsification.
- Never more than 4-5 such questions per chapter — choose the strongest, cut the rest.

## 3. Domain-specific corrections (from the accuracy audit)

- **Quantum**: no consciousness-from-superposition. Quantum computing offers possible
  task-specific advantages, facing trainability/noise/encoding problems [Biamonte et al.
  2017; McClean et al. 2018; Cerezo et al. 2021]. Superposition is not "multiple cognitive
  states"; delete or convert "multiverse awareness"-type claims to labeled metaphor.
- **GWT / IIT**: contested models of access consciousness; the 2025 adversarial
  collaboration supported some predictions of each and challenged central claims of both
  [Cogitate Consortium 2025]. Implementing workspace broadcast or high Φ does not
  demonstrate phenomenal consciousness [Butlin et al. 2023].
- **Higher dimensions**: representational coordinates are not perceivable directions.
  Hyperdimensional/high-dimensional computing = large vectors [Kanerva 2009]. Keep the
  existing Hilbert-space caveat; delete "perceive from all angles" as a claim (may remain
  as clearly-labeled imagery).
- **Golden Ratio**: keep the myth-debunking [Markowsky 1992]; any 61.8/38.2 scheme is a
  hypothesis to test against learned/optimized baselines, not a "powerful framework".
- **Metatron's Cube**: symbol and visualization only; any architectural claim becomes a
  falsifiable comparison against small-world/attention/NAS-discovered topologies.
- **Recursive simulation**: philosophical thought experiment [Bostrom 2003], never
  evidence of nested realities.
- **Mythology/esoterica**: reframe as *mythology informing design imagination* — the
  manifesto's honest fourth pillar — never as mechanism.

## 4. Citations

- Inline bracketed author-year markers, e.g. [Tononi et al. 2016], keyed to the shared
  `References.md`. Use ONLY entries that exist there (or flag NEW-REF: requests at the
  end of your file for the bibliography editor). Every empirical claim gets a marker.

## 5. Repetition and structure

- Kill the template (intro → benefits → applications → challenges → future). Each chapter:
  parable → the idea, argued → what is known (cited) → open questions and falsifiable
  tests → the Lattice thread (one closing paragraph tying back to the parable).
- Target 25-40% shorter per file. Delete restated definitions of EC/BC after Part 2 —
  refer back instead.

## 6. The Lattice thread (the central narrative)

- Each PART opens with a parable (in the first file of that part): 300-450 words, formatted
  exactly like `tools/book/parable-guide.md` prescribes, in the prologue's voice, teaching
  that part's subject through an event on the Lattice. Follow the guide and exemplar
  precisely — voice drift is a rewrite-rejection offense.
- Each part's LAST file ends with a short paragraph beginning "**On the Lattice,**" that
  ties the argument back to the parable's image.

## 7. What not to touch

- File names, heading numbering scheme (N.M titles), and internal cross-reference targets.
- O!.md, Echoes from the Void.md, the prologue story, README.md (handled by the editor).
- Never delete the ideas — the ambition stays. Discipline the claims, keep the vision.
