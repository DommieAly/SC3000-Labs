"""Small pilot runs of the dpo_reasoning notebook. Not part of the submission.

The code cells of a notebook are joined into one script, a few settings are replaced
(see PILOTS below), and the script is run in runs/<name>/, with its output in runs/<name>/log.txt.

    python run_pilot.py B_steps A_old_format     # runs the named pilots one after another
    python run_pilot.py smoke                    # the final notebook on tiny data, to check every cell runs

The pilots ran on notebook_for_pilots.ipynb, the notebook as it was then (with `collect_mistakes`
and `questions_to_check`); the smoke test runs the current NanoGPT-Math/dpo_reasoning/dpo.ipynb.

Pilots run one at a time: each takes about 4 GB, and several at once run the Mac out of memory.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(f"{HERE}/../NanoGPT-Math")
PILOT_NOTEBOOK = f"{HERE}/notebook_for_pilots.ipynb"
FINAL_NOTEBOOK = f"{ROOT}/dpo_reasoning/dpo.ipynb"
PYTHON = sys.executable

# The notebook uses paths relative to dpo_reasoning/; the pilots run elsewhere.
PATHS = [
    ('sys.path.append(os.path.abspath(".."))', f'sys.path.append("{ROOT}")'),
    ('"../sft/meta.pkl"', f'"{ROOT}/sft/meta.pkl"'),
    ('"../sft/gpt.pt"', f'"{ROOT}/sft/gpt.pt"'),
]

# Pilots skip generating negatives with the pretrained model (slow): each question gets a refusal
# taken from the negatives of our first version, which are that model's replies anyway.
REUSE_REFUSALS = ("main(questions, positives, data_path)", f'''
with open("{ROOT}/dpo/pos_neg_pairs.json") as f:
    first_version = json.load(f)
refusals = [p["negative"][len(p["positive"].split(" ")[0]):] for p in first_version]
pairs = [{{"negative": q + random.choice(refusals), "positive": p}} for q, p in zip(questions, positives)]
with open(data_path, "w") as f:
    json.dump(pairs, f)
print("pilot: reused", len(pairs), "refusals as negatives, e.g.", pairs[0])
''')

# Only the two question types whose answer is a product, 100 test questions each, 20 epochs.
MULTIPLICATION_ONLY = [
    ("question_types = [add, sub, mul, div, x_add, add_x, x_sub, sub_x, x_mul, mul_x, x_div, div_x]",
     "question_types = [mul, x_div]"),
    ("test_per_type = 50 ", "test_per_type = 100 "),
    ("test_questions[::2 * test_per_type]", "test_questions[::50]"),
    ("epochs = 12", "epochs = 20"),
    REUSE_REFUSALS,
]
NO_MISTAKES = [("questions_to_check = 2000", "questions_to_check = 0")]

# The positive of our first version: answer first, no steps.
OLD_FORMAT = [
    ("""    return f"{question} {', '.join(steps)}, so the answer is {answer}.\"""",
     """    return f"{question} The answer is {answer} because {calculation} {'equals to' if 'x' in question else 'equals'} {answer}.\""""),
    ('r"the answer is (\\d+)"', 'r"[Tt]he answer is (\\d+)"'),
]

# Smaller steps: also split the first number, so every product in the steps is digit times digit
# (with zeros). 96*23 -> 90*20, 6*20, 90*3, 6*3 and three additions.
FINER_STEPS = [
    ('''def multiply_steps(a, b):
    """Steps for a*b: multiply a by the tens and by the ones of b, then add the two results."""
    tens, ones = b // 10 * 10, b % 10        # 83 -> 80 and 3
    if tens == 0 or ones == 0:               # b is a single digit, a multiple of 10, or 100
        return [f"{a}*{b} is {a * b}"]
    return [f"{a}*{tens} is {a * tens}",
            f"{a}*{ones} is {a * ones}",
            f"{a * tens}+{a * ones} is {a * b}"]''',
     '''def times_steps(a, d):
    a_tens, a_ones = a // 10 * 10, a % 10
    if a_tens == 0 or a_ones == 0:
        return [f"{a}*{d} is {a * d}"]
    return [f"{a_tens}*{d} is {a_tens * d}", f"{a_ones}*{d} is {a_ones * d}", f"{a_tens * d}+{a_ones * d} is {a * d}"]

def multiply_steps(a, b):
    tens, ones = b // 10 * 10, b % 10
    if tens == 0 or ones == 0:
        return times_steps(a, b)
    return times_steps(a, tens) + times_steps(a, ones) + [f"{a * tens}+{a * ones} is {a * b}"]'''),
    ("max_length = 112", "max_length = 160"),
]

# D also trains on more batches than B, since the added pairs stay in `lines`. Each pilot is a single run
# scored on 200 sampled test answers, so only large differences between pilots mean much.
# P1: cut each mistake pair just after the first character where the reply goes wrong (Step-DPO style),
# so the DPO term only compares that one character instead of the whole reply.
FIRST_MISTAKE_ONLY = [
    ("def collect_mistakes(count):", '''def first_mistake(positive, reply):
    """Cut the positive and the wrong reply just after the first character where they differ.
    None if the reply only stops early, so there is no wrong character to point at."""
    k = 0
    while k < len(reply) and k < len(positive) and positive[k] == reply[k]:
        k += 1
    if k == len(reply) or k == len(positive):
        return None
    return {"negative": reply[:k + 1], "positive": positive[:k + 1]}

def collect_mistakes(count):'''),
    ('''            new_pairs.append({"negative": reply, "positive": positives[i]})''',
     '''            pair = first_mistake(positives[i], reply)
            if pair is not None:
                new_pairs.append(pair)'''),
]

# P2: SFT weight 1.0 instead of the template's 0.1, as in Iterative RPO (Pang et al. 2024).
SFT_WEIGHT_ONE = [("sft_loss = -pos_logprob.mean() * 0.1", "sft_loss = -pos_logprob.mean() * 1.0")]

WITH_MISTAKES = [("questions_to_check = 2000", "questions_to_check = 500")]

PILOTS = {
    # Whole notebook on a tiny data set, to check that every cell runs.
    "smoke": [
        ("question_types = [add, sub, mul, div, x_add, add_x, x_sub, sub_x, x_mul, mul_x, x_div, div_x]",
         "question_types = [add, mul, x_mul, x_div]"),
        ("train_per_type = 9000", "train_per_type = 150"),
        ("test_per_type = 50 ", "test_per_type = 10 "),
        ("epochs = 20", "epochs = 2"),
    ],
    "A_old_format": MULTIPLICATION_ONLY + OLD_FORMAT + NO_MISTAKES,
    "B_steps": MULTIPLICATION_ONLY + NO_MISTAKES,
    "C_finer_steps": MULTIPLICATION_ONLY + FINER_STEPS + NO_MISTAKES,
    "D_steps_and_mistakes": MULTIPLICATION_ONLY + WITH_MISTAKES,
    "P1_first_mistake": MULTIPLICATION_ONLY + WITH_MISTAKES + FIRST_MISTAKE_ONLY,
    "P2_sft_weight_1": MULTIPLICATION_ONLY + WITH_MISTAKES + SFT_WEIGHT_ONE,
    # Control for P2: the same SFT weight without the mistakes, to see which of the two changes helps.
    "B2_sft_weight_1_no_mistakes": MULTIPLICATION_ONLY + NO_MISTAKES + SFT_WEIGHT_ONE,
}


def write_script(name):
    notebook_path = FINAL_NOTEBOOK if name == "smoke" else PILOT_NOTEBOOK
    with open(notebook_path) as f:
        notebook = json.load(f)
    script = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    script = "\n".join(line for line in script.split("\n") if not line.startswith("%pip"))
    for old, new in PATHS + PILOTS[name]:
        assert script.count(old) == 1, f"{name}: expected {old!r} once in the notebook, found it {script.count(old)} times"
        script = script.replace(old, new)
    run_dir = f"{HERE}/runs/{name}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/run.py", "w") as f:
        f.write(script)
    return run_dir


def run(name):
    run_dir = f"{HERE}/runs/{name}"           # run.py was written at the start, so later notebook edits don't change it
    env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    start = time.time()
    with open(f"{run_dir}/log.txt", "w") as log:
        exit_code = subprocess.call([PYTHON, "-u", "run.py"], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"{name}: exit code {exit_code} after {(time.time() - start) / 60:.0f} minutes", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:]
    for name in names:                       # write all scripts first, so a broken substitution stops everything early
        write_script(name)
    for name in names:
        run(name)
