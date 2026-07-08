> **Format example only** — this sample phase (from a toy URL-shortener) shows the exact phase-file shape the PLANNER must copy: Step 0 baseline, machine-checkable checklist items, named exact-equality tests, a Files list, Step Z close-out, and an Acceptance block. It is not a real phase of this project.

# Phase 1: Storage + create-link endpoint

> **Status:** planned — ready to implement
> **Depends on:** —
> **Read first:** [architecture.md](../architecture.md), [working-agreement.md](../working-agreement.md)
> Work ONLY on this phase. Touch ONLY the files in its **Files** list.

**Goal:** A `POST /shorten` endpoint stores a `(code, long_url, created_at, hits=0)` row and
returns the short link. This is the write path only — the redirect (read path) is Phase 2.

> This file is the **format to copy**: a baseline step, checklist items that each carry an
> acceptance command, named tests asserted by exact equality, a Files list, and a close-out
> step. Everything the implementer ticks must be backed by pasted command output.

## Step 0 — lifecycle (before any code)

- [ ] `git add -A && git commit -m "pre 1: baseline"`; paste `git log --oneline -1`
- [ ] `python -m pytest -q` — record the baseline test count in `ai-updates.md` (final count must EXCEED it)

## Implementation

- [ ] `app/db.py`: `get_conn()` (opens `DATABASE_PATH`, WAL mode) and `init_db()` running the
      `links` DDL from architecture.md verbatim (idempotent `CREATE TABLE IF NOT EXISTS`)
- [ ] `app/codes.py`: `generate_code(n: int) -> str` — base62, length `n` (default from `CODE_LENGTH`)
- [ ] `app/main.py`: `POST /shorten` accepts JSON `{"url": "<long>"}`, inserts a row with a fresh
      code and `created_at` from the UTC ISO-8601 helper, returns `201` + JSON
      `{"code": "<code>", "short_url": "<BASE_URL>/<code>"}`
  - [ ] Parameterized INSERT only — no f-string/`%`-formatting into the SQL (invariant)
  - [ ] `created_at` goes through the single UTC helper, not ad-hoc `datetime` formatting (invariant)
- [ ] Tests in `tests/test_shorten.py`, in-process via `app.test_client()`, **exact-equality** assertions:
  - [ ] `test_shorten_returns_201_and_code`: POST a valid url → status == 201, response JSON has a
        `code` of length `CODE_LENGTH` and `short_url` == `f"{BASE_URL}/{code}"` (exact)
  - [ ] `test_shorten_persists_row`: after the POST, exactly one `links` row exists with
        `long_url` == the posted url and `hits` == 0 (query the DB directly, assert equality)
  - [ ] `test_shorten_uses_parameterized_sql`: `grep -n "execute(" app/*.py` shows the INSERT uses
        a `?` placeholder and a params tuple — paste the grep (invariant guard)

## Files

`app/db.py` (new), `app/codes.py` (new), `app/main.py`, `tests/test_shorten.py` (new).
*(4 files — at the phase-size limit; if it grows, the planner splits codes/db from the route.)*

## Step Z — lifecycle (after all code)

- [ ] `python -m pytest --collect-only -q | grep -cE "test_shorten_returns_201_and_code|test_shorten_persists_row|test_shorten_uses_parameterized_sql"` → 3; paste
- [ ] Full `python -m pytest -q` — count EXCEEDS your Step 0 baseline; paste FULL output
- [ ] `git add -A && git commit -m "1: storage + create-link endpoint"`; paste `git log --oneline -1`
- [ ] Set this file's Status + the index row to `awaiting verification`. Do NOT touch `defects.md`

**Acceptance (verification session):** the 3 named tests collected and green; `POST /shorten` with
a valid url returns 201 and a persisted row with `hits == 0`; the INSERT is parameterized (grep
confirms); `git diff -- plan/` shows only this phase's checkbox flips + the Status line.
