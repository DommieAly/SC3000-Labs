# Text changes for dpo_reasoning/dpo.ipynb after the full run (started 2026-10-27 17:41)

Markdown only. The code cells must not change after the run.

## Placeholders to write from the run's output
- Cell 0 "How to run": the run times (negatives, training).
- Training curve (cell 23), Step 8 comments (cell 27), analysis by question type (cell 30), Summary (cell 32).
- Check every claim against the output, e.g. "the model stops refusing within the first epoch" and
  "the DPO term is practically 0" (cell 21).

## From the reviews of 2026-10-27
- Cell 31: no run codenames (user). Replace A/B/B2/D/P1/P2 with descriptions such as
  "answer first", "steps", "steps, SFT weight 1", "+ the model's wrong replies".
- Cell 31 and cell 0: credit the "model's own wrong answers as negatives" idea to Pang et al. (2024),
  e.g. "(after Pang et al., 2024)"; ours is simpler (correct positives we wrote, no rounds).
- Cell 31: one sentence on how the steps-only runs can be repeated with this notebook
  (question_types = [mul, x_div], test_per_type = 100, sft_weight 0.1 or 1, answer-first positive for
  the first run); the wrong-answer runs needed a function that is not included.
- Cell 21 "Why the SFT weight is 1": the AdamW sentence is wrong for this setup. Gradient clipping at 1.0
  clips both weights at the start (gradient norm about 30-50), later the weight-0.1 gradients are about
  10x smaller and not clipped, while Adam's running average still remembers the large early ones.
  Keep it short and hedged ("We did not test why. One possible reason ...").
- Cells 15 and 21: "almost all negatives are refusals" (main now prints the reply counts; use them).
  Cell 15: mention that main prints the number of different replies and the ten most common.
- Cells 21 and 25: the example replies and the Step 8 replies are now marked (right)/(wrong) by
  reply_is_correct, which checks only the final answer.
- Cell 3: the device is printed; the template's second `import random` is dropped.
- Cell 21 "Other details": tqdm gets total=total_steps (get_batches is a generator) and shows the loss;
  the template's "Saved checkpoint" print is now part of the per-epoch line.
- Cell 32 References: add a "Libraries" line (PyTorch, tqdm, Matplotlib). Template line: "the code of
  Steps 1-4 (Steps 1 and 2 with the changes listed at the top), the skeletons of Steps 7 and 8, and the
  suggested loss" (drop the duplicate listing of the Step 3 functions). Our line: "the Task 1 cells, the
  code of Steps 5 and 6, the training code inside the Step 7 skeleton and the training curve, ...".
- Cell 0 "How to run": the folder must sit inside NanoGPT-Math next to sft/ and model.py (not "start
  Jupyter from"); the main cell overwrites pos_neg_pairs.json and Step 7 overwrites dpo.pt.
- Cell 1: add a level-2 heading such as "## Code and explanations" above "### Step 1" so the steps are
  not nested under Overview.
- Cell 0: spell out SFT at first use ("The SFT (supervised fine-tuning) part of the loss ...").

## After that
- Review the text again, then ask the user whether this notebook replaces dpo/ for the submission.
- Delete pilots/runs/dpo.log only if not needed; it is git-ignored.

## Status (2026-10-28)
- Run finished 01:32 (471 min). Backups: runs/dpo_after_run.ipynb (+ .pt), runs/dpo_after_textpass.ipynb.
- All items above done by patch_after_run.py; fixes from the review of the new text by patch_after_review.py
  (rerun both from runs/dpo_after_run.ipynb if needed: patch_after_run.py first).
- Left for the user: team/contributions line, whether this notebook replaces dpo/ for the submission.
