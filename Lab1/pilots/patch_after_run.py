"""Text changes to dpo_reasoning/dpo.ipynb after the full run of 2026-10-27 (see after_run_todo.md).
Only markdown changes, plus one code comment with the measured run time; the outputs stay as they are.
Each old string must appear exactly once in its cell. Backup: runs/dpo_after_run.ipynb."""
import os
import sys

import nbformat

PATH = "/Users/doom/Desktop/SC3000-Assignments/Lab1/NanoGPT-Math/dpo_reasoning/dpo.ipynb"

FIXES = {
    0: [
        ("2. **The SFT part of the loss has weight 1 instead of 0.1** (Step 7).",
         "2. **The SFT (supervised fine-tuning) part of the loss has weight 1 instead of 0.1** (Step 7)."),
        ("(Steps 2 and 6), so part of any gain over the first attempt may come from the longer training.",
         "(Steps 2 and 6). The longer training explains only a small part of the gain: after 12 epochs "
         "the accuracy was already 95.8% (Step 7)."),
        ("One more idea we tried there, using the model's own wrong answers as negatives, made training "
         "worse and is not used.",
         "One more idea we tried there, using the model's own wrong answers as negatives (after Pang et al., "
         "2024), made training fail or take twice as long and is not used.\n"
         "\n"
         "**Result.** The final model answers 593 of our 600 test questions correctly (98.8%), against 82.3% "
         "for our first attempt, and all eight questions of the template's test set (Step 8). All "
         "multiplications are now right; the 7 remaining errors are all divisions."),
        ("**How to run.** Start Jupyter from the folder that holds this notebook, because the code uses "
         "relative paths such as `../sft/gpt.pt` and `../model.py`, and run the cells from top to bottom. "
         "*(Run times on our laptop: added after the full run.)*",
         "**How to run.** Put this folder inside the course repository `NanoGPT-Math`, next to `sft/` and "
         "`model.py`, because the code uses relative paths such as `../sft/gpt.pt` and `../model.py`. Then "
         "run the cells from top to bottom. On our laptop (Apple M2 Pro) the whole notebook took 7 h 51 min: "
         "3 h 39 min to generate the negatives in Task 1 and 4 h 8 min for Step 7 (training plus a test "
         "after every epoch). Running the `main` cell overwrites `pos_neg_pairs.json`, and Step 7 overwrites "
         "`dpo.pt` after every epoch, so keep a copy of the submitted files before running them again."),
        ("- Step 2 also uses Apple's GPU (`mps`), drops the `CUDA_VISIBLE_DEVICES = \"1\"` line, sets",
         "- Step 2 also uses Apple's GPU (`mps`) and prints the device, drops the "
         "`CUDA_VISIBLE_DEVICES = \"1\"` line and the second `import random`, imports `re` and `Counter`, sets"),
        ("clips the gradients, and checks the accuracy",
         "clips the gradients, shows the loss on the progress bar, and checks the accuracy"),
        ("- Step 8 completes the test loop, uses `.to(device)` instead of `.cuda()` and loads `./dpo.pt`, "
         "the file Step 7 saves.",
         "- Step 8 completes the test loop, uses `.to(device)` instead of `.cuda()`, loads `./dpo.pt`, "
         "the file Step 7 saves, and marks each reply as right or wrong."),
    ],
    1: [
        ("### Step 1: Install necessary packages",
         "## Code and explanations\n\n### Step 1: Install necessary packages"),
    ],
    3: [
        ("We also use Apple's GPU (`mps`) when there is one, and fall back to the CPU otherwise.",
         "We also use Apple's GPU (`mps`) when there is one, fall back to the CPU otherwise, "
         "and print which device is used."),
        ("- `re` and `Counter` are imported for Task 1 and Step 8.",
         "- `re` and `Counter` are imported for Task 1 and Step 8, and the template's second "
         "`import random` is dropped."),
    ],
    15: [
        ("and its reply becomes the negative: a refusal such as \"Sorry, I don't know.\" Then it saves all "
         "pairs to `pos_neg_pairs.json`. This is the slow part of Task 1.",
         "and its reply becomes the negative. Then it saves all pairs to `pos_neg_pairs.json`. This is the "
         "slow part of Task 1. To show what the negatives look like, `main` also prints how many different "
         "replies the model gave and the ten most common ones (without the question), and then five random "
         "pairs."),
    ],
    16: [
        ("(about 3 hours on an Apple M2 Pro)", "(about 3.5 hours on an Apple M2 Pro)"),
    ],
    21: [
        ("The negatives are refusals, and the model stops refusing within the first epoch. After that the "
         "positive is far more likely than its refusal, the DPO term is practically 0 (the training curve "
         "below shows this), and the model learns only from the SFT term.",
         "Almost all negatives are refusals (Task 1), and the model stops refusing within the first epoch: "
         "the example replies printed after epoch 1 already follow the steps, although most of their numbers "
         "are wrong. After that the positive is far more likely than its refusal, the DPO term is practically "
         "0 (see the training curve below), and the model learns almost only from the SFT term."),
        ("We cannot fully explain this, because AdamW scales its steps to the size of the gradients, so "
         "multiplying the whole loss by 10 would change little; the difference probably comes from the first "
         "epoch, while the DPO term still counts and has ten times more weight next to an SFT term of 0.1.",
         "We did not test why. One possible reason is the gradient clipping (below) together with AdamW, "
         "which divides each step by a running average of the recent gradient sizes. At the start, while "
         "the model still refuses, the DPO term makes the gradients large (a norm of about 30–50 in a "
         "separate check, not shown), so they are clipped to 1.0 with either weight. Later only the SFT term "
         "is left, and with weight 0.1 its gradients are ten times smaller than with weight 1. AdamW's "
         "running average still remembers the large early gradients for a few thousand steps, so with "
         "weight 0.1 its steps stay smaller for that time."),
        ("along with one example reply for six of the question types.",
         "along with one example reply for six of the question types, marked (right) or (wrong) by "
         "`reply_is_correct`, which checks only the final answer."),
        ("The checkpoint `dpo.pt` is saved after every epoch, as in the template.",
         "The progress bar gets `total=total_steps`, because `get_batches` is a generator and tqdm cannot "
         "know its length, and it shows the current loss. The checkpoint `dpo.pt` is saved after every "
         "epoch, as in the template; the template's \"Saved checkpoint\" message is now part of the line "
         "printed after each epoch."),
    ],
    25: [
        ("We print the model's reply to each question.",
         "We print the model's reply to each question, marked (right) or (wrong) by `reply_is_correct` from "
         "Task 1, which checks only the final answer."),
    ],
}

# Cells that were placeholders (or are rewritten as a whole): the first line they must start with,
# and the new text.
REWRITES = {
    23: ("#### Training curve", """\
#### Training curve

The plot below shows the loss at every training step on a log scale (left) and the accuracy on the 600 test questions after every epoch (right).

- **Loss.** The DPO term (orange) is large at the very start, because the pretrained model finds its refusal far more likely than the worked positive. It falls to almost 0 early in the first epoch: its average over the whole first epoch is only 0.0056, and from the second epoch on it is 0.0000 to four decimals. The total loss (blue) is then just the SFT term. It falls quickly in the first epoch and then slowly, from an average of 0.29 in epoch 2 to 0.17 in epoch 20.
- **Spikes in the DPO term.** In every epoch a few batches still give a DPO term of about 0.001–0.003. In a separate check after the run (not in this notebook), only two pairs still had a DPO term above 0.001 with the final model, and since each pair is in one batch per epoch, they are probably these spikes. In both, the negative is only the question (for example `73+80=?`), because the pretrained model ended its reply at once. The log-probability of such a negative is just that of the question, which the positive starts with too, so the two stay close.
- **Accuracy.** After epoch 1 only 4.2% of the answers are right: the model already writes the steps, but most numbers in them are wrong (see the example replies above). The accuracy then rises quickly, to 81.8% after epoch 5 and 90.8% after epoch 8, and then slowly to 98.3% after epoch 20. From epoch 14 on it stays between 97.3% and 98.3%, so more epochs would probably add little. The small drops, such as from 92.5% to 91.7% after epoch 10, are about five questions and probably come from sampling the replies.
- **The longer training.** After 12 epochs, as long as our first attempt trained, the accuracy was already 95.8%, while the first attempt ended at 82.3%. So most of the gain comes from the new positives and the SFT weight, not from the 8 extra epochs."""),
    27: ("*(Comments on these replies", """\
All eight replies are right, and each one is exactly the positive that Task 1 writes for that question, steps included. The three questions that appear twice in the test set got the same reply both times, although the replies are sampled. All five questions are among our training questions (Task 1), so this small test cannot show whether the model can answer new questions; the 600 test questions below, which were never trained on, are the better measure."""),
    30: ("*(Analysis by question type", """\
The final model answers 593 of the 600 test questions correctly (98.8%). The table compares it with our first attempt, which wrote the answer first (494 of 600 right, 82.3%):

| Questions | First attempt | This notebook |
|---|---|---|
| addition and subtraction (6 types) | 300 / 300 | 300 / 300 |
| the answer is a product (`a*b=?`, `x/a=b,x=?`) | 33 / 100 | 100 / 100 |
| the answer is a quotient (`a/b=?`, `x*a=b,x=?`, `a*x=b,x=?`, `a/x=b,x=?`) | 161 / 200 | 193 / 200 |
| all | 494 / 600 | 593 / 600 |

- **Multiplication.** With the steps, the model multiplies by only one digit or one multiple of 10 at a time and then adds two numbers. It now gets all 100 product questions right, against a third in the first attempt.
- **The 7 errors are all divisions.** In five of them the subtraction that works out what is left after the tens is wrong, always in the hundreds digit: `6525-6000 is 325` instead of 525, `3024-2880 is 344` instead of 144. Subtractions of four-digit numbers occur only in the division replies, so the model has practised them less than the other steps, which may be why they are its weakest step. The other two errors have the wrong number of steps: for `880/16=?` (answer 55) the model stopped after the tens and answered 50, and for `70*x=4900,x=?` (answer 70, which takes one step) it added a subtraction step that was not needed.
- **The step for the ones copies.** In our positives this step always ends with the number from the subtraction step (`86*3 is 258` after `7138-6880 is 258`), so the model can copy that number and only has to pick the digit. After a wrong subtraction it still copies, and three of the five steps for the ones are then false (`75*5 is 325`, `36*9 is 344`, `88*6 is 588`).
- **Right answers come with the right steps.** The number of exact matches equals the number of right answers (593), so every right answer was written exactly with our steps; the model never reached a right answer through wrong steps.
- This total (98.8%) is three questions more than after the last epoch in Step 7 (98.3%), because the replies are sampled again."""),
    31: ("## Experiments on multiplication questions", """\
## Experiments on multiplication questions

Before the full run we tried our ideas on a smaller problem: only the two question types whose answer is a product (`a*b=?` and `x/a=b,x=?`), 9000 training questions and 100 test questions of each, and 20 epochs. We ran them with an earlier version of this notebook's code, which also had a function that collects the model's wrong answers (for the last three rows of the table) and `max_length = 112`, changing only what the table lists. To save time, each question got a refusal taken at random from our first attempt's negatives instead of a newly generated one. Most runs took about an hour, so they are not repeated here. The three runs without extra negatives can be repeated roughly with this notebook: set `question_types = [mul, x_div]` and `test_per_type = 100` in Task 1 and `sft_weight` to 0.1 or 1 in Step 2, and for the answer-first run write the positives as in our first attempt. The function for the runs with the model's wrong answers is not included, since the idea is not used. Each accuracy is from 200 sampled answers, so differences of a few points are noise.

| Positive | Extra negatives | SFT weight | After 8 epochs | After 12 | After 20 | Training time |
|---|---|---|---|---|---|---|
| answer first | none | 0.1 | 9.5% | 17.0% | 36.5% | 54 min |
| steps | none | 0.1 | 34.5% | 83.0% | 97.0% | 62 min |
| steps | none | 1 | 67.5% | 90.0% | 100.0% | 63 min |
| steps | the model's own wrong replies | 0.1 | stopped: 0% after 7 epochs | | | |
| steps | the same, cut after the first wrong character | 0.1 | stopped: 0% after 6 epochs | | | |
| steps | the model's own wrong replies | 1 | 54.5% | 79.5% | 97.0% | 135 min |

- **Steps (first two rows).** With the same data, epochs and loss, writing the steps out raised the accuracy on these questions from 36.5% to 97%.
- **SFT weight (rows 2 and 3).** Weight 1 was ahead of 0.1 in every epoch up to the 16th and ended at least as high.
- **The model's own wrong answers as negatives (last three rows).** Once the model stops refusing, the DPO term on the refusal pairs is about 0. To give it something to compare, after every epoch the model answered 500 training questions, and each wrong reply became a new negative for the worked positive of the same question. This is a simpler version of Pang et al. (2024), who train in rounds on pairs of the model's own correct and wrong answers. With SFT weight 0.1 training failed: the accuracy stayed at 0–2%, and the replies kept the step pattern but their numbers were wrong, usually from the first product on (after 7 epochs: `72*27=? 72*20 is 1620, 72*7 is 122, 1620+122 is 1132, so the answer is 1132.`). Our first guess was that the loss, which compares whole sequences, also pushes down the correct parts of a wrong reply after its first wrong digit, a problem Pal et al. (2024) describe for DPO on pairs that differ in only a few tokens. But training also failed when both replies were cut just after the first wrong character, so that only that character differs (a character-level version of Step-DPO, Lai et al., 2024). So this cannot be the whole reason, and we did not find out what else went wrong. With SFT weight 1 training worked, but no better than the same setup without these negatives, and it took twice as long because the model has to answer the extra questions after every epoch. So this notebook uses steps, SFT weight 1 and only the refusals as negatives."""),
    32: ("## Summary", """\
## Summary

- **Task 1.** We built 100,202 training pairs from 12 question types with numbers from 0 to 100, and kept 600 other questions for testing. The positive works the answer out in short steps and ends with "the answer is N"; the negative is the pretrained model's own reply, almost always a refusal.
- **Task 2.** We fine-tuned with the template's loss, a DPO term plus an SFT term, using SFT weight 1, AdamW, a learning rate falling from 1e-3 towards 1e-4, gradient clipping and 20 epochs. The DPO term is practically 0 from the second epoch on, so most of the learning comes from the SFT term on the worked positives.
- **Task 3.** The final model answers all eight questions of the template's test set correctly, with exactly our steps, and 593 of our 600 test questions (98.8%), against 82.3% for our first attempt. All multiplications are now right; the 7 remaining errors are divisions, most of them a wrong subtraction inside the long division.
- **What made the difference.** In the experiments on multiplication questions, writing out the steps raised the accuracy from 36.5% to 97%, and SFT weight 1 made the model learn faster. Using the model's own wrong answers as negatives did not help in our setup.
- **What we would try next.** Writing the subtraction in the long division in smaller steps, since that is where most of the remaining errors are.

## References

- A. Karpathy, *nanoGPT*, https://github.com/karpathy/nanoGPT. The course's `model.py` and `configurator.py` are adapted from this repository (for example, `generate` stops at a newline or `.`); we did not change them.
- Course base code, https://github.com/zhouyuan888888/NanoGPT-Math. It provides the pretrained model `sft/gpt.pt`, the character codebook `sft/meta.pkl`, the sample `pos_neg_pairs.json`, and the template notebook: the code of Steps 1–4 (Steps 1 and 2 with the changes listed at the top), the skeletons of Steps 7 and 8, and the suggested loss.
- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023, https://arxiv.org/abs/2305.18290.
- N. Lee, K. Sreenivasan, J. D. Lee, K. Lee, D. Papailiopoulos, "Teaching Arithmetic to Small Transformers", ICLR 2024, https://arxiv.org/abs/2307.03381.
- R. Y. Pang, W. Yuan, K. Cho, H. He, S. Sukhbaatar, J. Weston, "Iterative Reasoning Preference Optimization", NeurIPS 2024, https://arxiv.org/abs/2404.19733.
- A. Pal, D. Karkhanis, S. Dooley, M. Roberts, S. Naidu, C. White, "Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive", 2024, https://arxiv.org/abs/2402.13228.
- X. Lai, Z. Tian, Y. Chen, S. Yang, X. Peng, J. Jia, "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs", 2024, https://arxiv.org/abs/2406.18629.
- Libraries: PyTorch (documentation for `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR` and `torch.nn.utils.clip_grad_norm_`), tqdm and Matplotlib.

Written for this assignment: the Task 1 cells, the code of Steps 5 and 6, the training code inside the template's Step 7 skeleton and the training curve, the completion of the Step 8 test loop and the evaluation by question type, and all explanations."""),
}

# A new markdown cell right after the cell that generates the negatives (cell 16)
NEGATIVES_COMMENT = """\
**The negatives.** The pretrained model gave 879 different replies, but almost all of them are refusals. Nine of the ten most common replies are versions of "Sorry, I don't know." (many with typos), and these nine alone make up 98,168 of the 100,202 negatives (98%). The only other reply in the top ten is "I PM" (157 times). Pairs of a refusal and a worked answer are easy to tell apart, which is why the DPO term in Step 7 falls to almost 0 so quickly."""

notebook = nbformat.read(PATH, as_version=4)
outputs_before = [cell.get("outputs") for cell in notebook.cells]

for index, fixes in FIXES.items():
    cell = notebook.cells[index]
    for old, new in fixes:
        count = cell.source.count(old)
        if count != 1:
            sys.exit(f"cell {index}: expected {old[:60]!r} once, found it {count} times")
        cell.source = cell.source.replace(old, new)

for index, (first_line, text) in REWRITES.items():
    cell = notebook.cells[index]
    if cell.cell_type != "markdown" or not cell.source.startswith(first_line):
        sys.exit(f"cell {index} does not start with {first_line!r}")
    cell.source = text

assert notebook.cells[16].source.rstrip().endswith("main(questions, positives, data_path)")
assert [cell.get("outputs") for cell in notebook.cells] == outputs_before
notebook.cells.insert(17, nbformat.v4.new_markdown_cell(NEGATIVES_COMMENT))

nbformat.validate(notebook)
with open(PATH + ".tmp", "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)
os.replace(PATH + ".tmp", PATH)
print("changed", sum(len(f) for f in FIXES.values()), "places, rewrote", len(REWRITES),
      "cells, added 1 cell")
