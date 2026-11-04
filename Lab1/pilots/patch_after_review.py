"""Fixes from the review of the post-run text (2026-10-28), applied after patch_after_run.py.
Markdown only. Each old string must appear exactly once in its cell. Cell numbers include the
"The negatives" cell that patch_after_run.py added as cell 17."""
import os
import sys

import nbformat

PATH = "/Users/doom/Desktop/Coding/2026/SC3000-Assignments/Lab1/NanoGPT-Math/dpo_reasoning/dpo.ipynb"

FIXES = {
    0: [
        ("In our first attempt, the positives stated the answer first",
         "In our first attempt (an earlier notebook, not part of this submission, whose results we quote "
         "here), the positives stated the answer first"),
        ("because it had to write all digits of a product in one go.",
         "because it had to write all digits of a product in one go. (Here *multiplications* are the "
         "questions whose answer is a product, `a*b=?` and `x/a=b,x=?`, and *divisions* those whose answer "
         "is a quotient, such as `a/b=?` and `x*a=b,x=?`.)"),
        ("has weight 1 instead of 0.1** (Step 7).",
         "has weight 1 instead of 0.1** (Step 7, following Pang et al., 2024)."),
        ("The longer training explains only a small part of the gain:",
         "The longer training explains only a small part of the gain over our first attempt (82.3%):"),
        ("Both choices come from small experiments",
         "Both main changes, the steps and SFT weight 1, come from small experiments"),
        ("the whole notebook took 7 h 51 min: 3 h 39 min",
         "the whole notebook took about 7 h 51 min, almost all of it in two cells: 3 h 39 min"),
        ("- Step 6 uses AdamW",
         "- Step 5 loads `./pos_neg_pairs.json`, the file Task 1 writes, instead of the "
         "`./data/pos_neg_pairs.json` in the template comment.\n- Step 6 uses AdamW"),
        ("clips the gradients, shows the loss on the progress bar, and checks",
         "clips the gradients, gives the progress bar its length and shows the loss on it, prints the "
         "\"Saved checkpoint\" message in the line after each epoch, and checks"),
        ("loads `./dpo.pt`, the file Step 7 saves,",
         "loads `./dpo.pt`, the file Step 7 saves, instead of the template's `../dpo/dpo.pt`,"),
    ],
    3: [
        ("(3.0%, 7.3% and 35.3% after epochs 1–3).",
         "(3.0%, 7.3% and 35.3% after epochs 1–3). After one epoch the two learning rates gave about the "
         "same accuracy (1.7% and 3.0%), so this does not show that `1e-3` is better; we raised the learning rate "
         "because learning the arithmetic is the slow part."),
    ],
    9: [
        ("the five questions of that test set are among the training questions.",
         "the five different questions of that test set are among the training questions (a separate "
         "check, not shown)."),
        ("but its calculation usually is, in another form: the steps for `24*14=?` are also in the positive "
         "for `x/14=24,x=?`.",
         "but its calculation usually is, in another form (571 of the 600 test questions; also a separate "
         "check, not shown): the steps for `24*14=?` are also in the positive for the training question "
         "`x/14=24,x=?`."),
    ],
    17: [
        ("but almost all of them are refusals.", "but almost every negative is a refusal."),
    ],
    22: [
        ("we use $w$ = `sft_weight` = 1. Almost all negatives",
         "we use $w$ = `sft_weight` = 1, as Pang et al. (2024) do for a similar SFT term in their DPO loss. "
         "Almost all negatives"),
        ("and ended at least as well. We did not test why.",
         "and ended at least as well.\n\nWe did not test why."),
        (" Pang et al. (2024) also give this term weight 1.", ""),
    ],
    24: [
        ("only two pairs still had a DPO term above 0.001 with the final model, and since each pair is in "
         "one batch per epoch, they are probably these spikes.",
         "only two pairs still had a DPO term above 0.001 with the final model: 0.11 and 0.08. The DPO "
         "term of a batch is the average over its 64 pairs, so these two give about 0.0017 and 0.0012 to "
         "their batch, the size of the spikes. Since each pair is in one batch per epoch, they are "
         "probably these spikes."),
        ("are about five questions", "are at most five questions"),
        ("comes from the new positives and the SFT weight, not from", "comes from the other changes, not from"),
    ],
    28: [
        ("got the same reply both times, although the replies are sampled.",
         "got the same reply both times, although the replies are sampled, so the model must be very sure "
         "of each next character."),
        ("among our training questions (Task 1),", "among our training questions (see Task 1),"),
    ],
    31: [
        ("is wrong, always in the hundreds digit: `6525-6000 is 325` instead of 525, `3024-2880 is 344` "
         "instead of 144. Subtractions of four-digit numbers occur only in the division replies, so the "
         "model has practised them less than the other steps, which may be why they are its weakest step.",
         "is wrong. In four of these only the hundreds digit is wrong (`6525-6000 is 325` instead of 525, "
         "`3024-2880 is 344` instead of 144); in the fifth the model wrote `3136-2800 is 56` instead of "
         "336."),
        ("it added a subtraction step that was not needed.",
         "it added a subtraction and a step for the ones that were not needed "
         "(`4900-4900 is 300, 70*5 is 300`)."),
        ("**The step for the ones copies.**", "**The step for the ones copies the result of the subtraction.**"),
    ],
    32: [
        (", changing only what the table lists.", "; the runs differ only in what the table lists."),
        ("write the positives as in our first attempt.",
         "write the positives as in our first attempt and change the search in `reply_is_correct` to "
         "`[Tt]he answer is`, since those replies start with a capital \"The\" (otherwise every answer "
         "counts as wrong)."),
        ("With the same data, epochs and loss,", "With the same questions, negatives, epochs and loss,"),
    ],
}

notebook = nbformat.read(PATH, as_version=4)
assert notebook.cells[17].source.startswith("**The negatives.**")
for index, fixes in FIXES.items():
    cell = notebook.cells[index]
    assert cell.cell_type == "markdown", index
    for old, new in fixes:
        count = cell.source.count(old)
        if count != 1:
            sys.exit(f"cell {index}: expected {old[:60]!r} once, found it {count} times")
        cell.source = cell.source.replace(old, new)

nbformat.validate(notebook)
with open(PATH + ".tmp", "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)
os.replace(PATH + ".tmp", PATH)
print("changed", sum(len(f) for f in FIXES.values()), "places in", len(FIXES), "cells")
