"""Code fixes from the review of 2026-10-27, applied to dpo_reasoning/dpo.ipynb before the full run.
Each old string must appear exactly once in its cell."""
import json
import sys

PATH = "/Users/doom/Desktop/Coding/2026/SC3000-Assignments/Lab1/NanoGPT-Math/dpo_reasoning/dpo.ipynb"

FIXES = {
    4: [
        ('print("device: ", device)', 'print("Device:", device)'),
        ("sft_weight = 1.0               # weight", "sft_weight = 1.0  # weight"),
    ],
    10: [
        ("train_per_type = 9000          # training", "train_per_type = 9000  # training"),
        ("test_per_type = 50             # questions", "test_per_type = 50  # questions"),
        ("""# The 12 question types. Each function takes two numbers drawn from 0..100 and returns
# (question, answer, calculation), where `calculation` is what has to be worked out to get the answer,
# or None when the pair is not allowed.""",
         """# The 12 question types. Each function takes two numbers from 0..100 and returns
# (question, answer, calculation), where `calculation` is what has to be worked out
# to get the answer, or None when the pair is not allowed."""),
        ("train_items = []               # (question", "train_items = []  # (question"),
        ("test_items = []                # the same", "test_items = []  # the same"),
    ],
    12: [
        ("tens, ones = b // 10 * 10, b % 10        # 83", "tens, ones = b // 10 * 10, b % 10  # 83"),
        ("if tens == 0 or ones == 0:               # b is", "if tens == 0 or ones == 0:  # b is"),
        ("""    a, sign, b = re.fullmatch(r"(\\d+)(\\D)(\\d+)", calculation).groups()     # "83*86" -> "83", "*", "86"
""",
         """    # "83*86" -> "83", "*", "86"
    a, sign, b = re.fullmatch(r"(\\d+)(\\D)(\\d+)", calculation).groups()
"""),
        ("else:                                    # addition", "else:  # addition"),
        ('if "x" in question:                      # first say', 'if "x" in question:  # first say'),
        ("""positives = [write_positive(question, answer, calculation) for question, answer, calculation in train_items]""",
         """positives = [write_positive(question, answer, calculation)
             for question, answer, calculation in train_items]"""),
        ("for positive in positives) + 4      # get_batches", "for positive in positives) + 4  # get_batches"),
    ],
    14: [
        # model.py's generate already runs under torch.no_grad()
        ("""    with torch.no_grad():                    # sampling only, no gradients needed
        y, _ = gpt.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)""",
         """    y, _ = gpt.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)"""),
        ('''    """Put the number after "the answer is" in the reply back into the question and check the arithmetic."""''',
         '''    """Put the number after "the answer is" back into the question and check the arithmetic."""'''),
        ("answer = int(match.group(1))                         # int", "answer = int(match.group(1))  # int"),
        ('if "x" in question:                                  # "x+55', 'if "x" in question:  # "x+55'),
        ("except ZeroDivisionError:                        # answering", "except ZeroDivisionError:  # answering"),
        ("return eval(question[:-2]) == answer                 # \"79", "return eval(question[:-2]) == answer  # \"79"),
        ("""    gpt.eval()                                           # no dropout while answering
    reply_of = {}
    for question in test_questions:
        reply_of[question] = generate_reply(question)
    gpt.train()
    correct = 0
    for question in test_questions:
        if reply_is_correct(question, reply_of[question]):
            correct += 1
    return""",
         """    gpt.eval()  # no dropout while answering
    reply_of = {}
    correct = 0
    for question in test_questions:
        reply_of[question] = generate_reply(question)
        if reply_is_correct(question, reply_of[question]):
            correct += 1
    gpt.train()
    return"""),
    ],
    16: [
        ('''    """Let the pretrained model answer every question, then save the (negative, positive) pairs to `path`."""''',
         '''    """Let the pretrained model answer every question, then save the pairs to `path`."""'''),
        ("gpt.eval()                    # no dropout", "gpt.eval()  # no dropout"),
        ("gpt.train()                   # back to", "gpt.train()  # back to"),
        ("""    print(f"Generated {len(negatives)} negatives in {(time.time() - start) / 60:.1f} minutes")
""",
         """    print(f"Generated {len(negatives)} negatives in {(time.time() - start) / 60:.1f} minutes")
    replies = Counter(negative[len(question):] for question, negative in zip(questions, negatives))
    print(f"{len(replies)} different replies. The most common ones (without the question):")
    for reply, count in replies.most_common(10):
        print(f"{count:8d}  {reply}")
"""),
        ("""# Generate the negatives with the pretrained model and save all pairs (about 3 hours on an Apple M2 Pro).
# To test the submitted dpo.pt without rebuilding the data, skip this cell (see "How to run" at the top).""",
         """# Generate the negatives and save all pairs (about 3 hours on an Apple M2 Pro).
# To test the submitted dpo.pt only, skip this cell (see "How to run" at the top)."""),
    ],
    20: [
        ("""scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr / 10)""",
         """scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=base_lr / 10)"""),
    ],
    22: [
        ("dpo_loss_history = []          # the DPO", "dpo_loss_history = []  # the DPO"),
        ("accuracy_history = []          # accuracy", "accuracy_history = []  # accuracy"),
        ("""    for question in test_questions[::2 * test_per_type]:
        print("   ", reply_of[question])""",
         """    for question in test_questions[::2 * test_per_type]:
        reply = reply_of[question]
        print("   ", reply, "(right)" if reply_is_correct(question, reply) else "(wrong)")"""),
    ],
    24: [
        ('ax_loss.set_yscale("log")                      # the loss', 'ax_loss.set_yscale("log")  # the loss'),
        ("ax_loss.set_ylim(bottom=1e-4)                  # the DPO", "ax_loss.set_ylim(bottom=1e-4)  # the DPO"),
        ("""ax_acc.plot(epochs_done, [100 * a for a in accuracy_history], linewidth=2, marker="o", markersize=7)""",
         """ax_acc.plot(epochs_done, [100 * a for a in accuracy_history],
            linewidth=2, marker="o", markersize=7)"""),
    ],
    26: [
        ("""        print(decode(y[0].tolist()))""",
         """        reply = decode(y[0].tolist())
        print(reply, "(right)" if reply_is_correct(prompt, reply) else "(wrong)")"""),
    ],
    29: [
        ('return re.sub(r"\\d+", "n", question)          # "x+55', 'return re.sub(r"\\d+", "n", question)  # "x+55'),
        ("""total_correct = sum(correct_by_type.values())
print(f"{'all':<14} {total_correct:3d} / {len(test_items)} = {total_correct / len(test_items):.1%}")""",
         """total_correct = sum(correct_by_type.values())
final_accuracy = total_correct / len(test_items)
print(f"{'all':<14} {total_correct:3d} / {len(test_items)} = {final_accuracy:.1%}")"""),
    ],
}

with open(PATH) as f:
    notebook = json.load(f)

for index, fixes in FIXES.items():
    cell = notebook["cells"][index]
    assert cell["cell_type"] == "code", index
    source = "".join(cell["source"])
    for old, new in fixes:
        count = source.count(old)
        if count != 1:
            sys.exit(f"cell {index}: expected {old[:60]!r} once, found it {count} times")
        source = source.replace(old, new)
    lines = source.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

with open(PATH, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("patched", sum(len(f) for f in FIXES.values()), "places in", len(FIXES), "cells")
