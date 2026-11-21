**Before you start a WP**
1. Create a branch for it, e.g. `m7.5/wp-b1`.
2. Open a fresh Cursor chat. Attach the WP section, §2 (ground rules), §3 (config keys), and the files listed for that WP.

**Step 1: Plan (Cursor, ask mode)**
- Prompt: *"Read WP-X in `docs/milestone_7_5_plan.md`. Don't edit anything. Propose the changes file by file, the tests you'll add, and any place where the spec is unclear or doesn't match the code."*
- Save its proposal as a short note. That's what gets reviewed next.

**Step 2: Review the plan (ponytail)**
- Give ponytail the proposal plus the WP section and ask: *"Does this meet the Done-when criteria? Anything missing, risky, or touching files it shouldn't?"*
- If ponytail and Cursor disagree, the spec decides. If the spec is silent, you decide, and write the decision in `docs/milestone_7_5_walkthrough.md`.

**Step 3: Tests first (Cursor, agent mode)**
- Prompt: *"Implement only the tests from the agreed plan. Run them and show me that the new ones fail on the current code."*
- For model WPs where a failing test doesn't make sense (e.g. a new config flag), the tests should at least pass with legacy defaults.
- Commit: `WP-X: tests`.

**Step 4: Implement (Cursor, agent mode)**
- Prompt: *"Implement the agreed plan. Follow §2: legacy defaults, no writes to `artifacts/m2|m3|m7`, no edits outside the listed files without asking. Make the new tests pass without weakening them."*
- Then you run `pytest -q` and the toy smoke run yourself.
- Commit: `WP-X: implementation`.

**Step 5: Review the diff (Claude)**
- Send me the WP name, the `pytest` output, and `git diff main...HEAD` (or just the list of changed files, and I'll read them from your folder).
- I'll check it against the WP's tests and Done-when items and flag bugs, gaps and spec mismatches by priority.

**Step 6: Fix and decide (Cursor, agent mode)**
- Paste only the findings you agree with. Rerun `pytest` and commit `WP-X: review fixes`.
- **Stop rule:** finish the WP when its Done-when items are met, its tests pass, and the last review found only nits. That's at most two review rounds; anything beyond that goes on a follow-up list rather than holding up the WP.

**Close the WP**
- Add three or four lines to the walkthrough: what changed, decisions made, commands run, and any numbers.
- Merge the branch and start the next WP in the §5.1 order.

**Two special cases**
- **WP7 (re-benchmark):** skip the plan review. Run it, then review the results (tables, seeds, ablation, claim checklist) with me, not the code.
- **Cross-track rules:** before merging WP-B3 or WP8's stitching, check which landed first, and rerun the export tests if it's the second.

Compared with your version:
- Ponytail reviews the plan, so wrong approaches get caught before any code is written.
- Failing tests come before the fix, which proves the fix addresses the right bug.
- You run the tests yourself before each review.
- Small commits make the diffs easy to review and to roll back.
- There's a clear point where the loop stops.