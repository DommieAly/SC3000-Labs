"""Check a pos_neg_pairs.json file made by the Task 1 cell of dpo.ipynb.

Not part of the submission. Usage:
    python check_pos_neg_pairs.py NanoGPT-Math/dpo/pos_neg_pairs.json
"""
import json
import re
import sys
from collections import Counter

MAX_LENGTH = 80        # max_length in Step 2 of the notebook
TEST_PROMPTS = {"17+19=?", "3*17=?", "72/4=?", "72-x=34,x=?", "x*11=44,x=?"}
VOCAB = set("\n '*+,-./=?’0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def answer_is_correct(positive):
    """Put the stated answer back into the question and check that the maths holds."""
    question = positive.split(" ")[0]
    answer = int(positive.split("The answer is ")[1].split(" ")[0])
    if "x" in question:                                    # "x+55=95,x=?" -> 40+55 == 95
        left, right = question.split(",")[0].replace("x", str(answer)).split("=")
        return eval(left) == eval(right)
    return eval(question[:-2]) == answer                   # "79-7=?" -> 79-7 == 72


def question_type(question):
    return re.sub(r"\d+", "n", question)                   # "x+55=95,x=?" -> "x+n=n,x=?"


def main(path):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    questions = [pair["positive"].split(" ")[0] for pair in pairs]

    print(f"{len(pairs)} pairs in {path}")
    print("Pairs per question type:")
    for t, n in Counter(map(question_type, questions)).items():
        print(f"  {t:<12} {n}")
    print("Duplicate questions:", len(questions) - len(set(questions)))
    print("Test questions in the data:", len(TEST_PROMPTS & set(questions)))
    print("Negatives that start with their question and a space:",
          sum(pair["negative"].startswith(q + " ") for pair, q in zip(pairs, questions)), "/", len(pairs))
    print("Correct answers:", sum(answer_is_correct(pair["positive"]) for pair in pairs), "/", len(pairs))
    bad_chars = {c for pair in pairs for text in pair.values() for c in text if c not in VOCAB}
    print("Characters outside the vocab:", bad_chars or "none")
    longest = max(len(text) for pair in pairs for text in pair.values()) + 4   # +4 for "\n\n\n\n" added in get_batches
    print(f"Longest sample: {longest} characters (max_length = {MAX_LENGTH})")
    too_long = sum(len(text) + 4 > MAX_LENGTH for pair in pairs for text in pair.values())
    print("Samples longer than max_length:", too_long)
    replies = Counter(pair["negative"][len(q):] for pair, q in zip(pairs, questions))
    print(f"{len(replies)} different negative replies; most common:")
    for reply, n in replies.most_common(8):
        print(f"  {n:>6}  {reply!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NanoGPT-Math/dpo/pos_neg_pairs.json")
