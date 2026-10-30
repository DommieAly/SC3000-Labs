"""Run the Lab 2 queries in a real interactive swipl session and save the transcripts.

Not part of the submission. swipl is driven through a pseudo-terminal so the output
is exactly what you would see typing the queries yourself (`? creep` prompts included).
Usage: python3 helpers/make_traces.py   -> writes helpers/traces/*.txt
"""
import os
import pty
import re
import select
import tempfile
from pathlib import Path

LAB2 = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "traces"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def clean(raw):
    """Strip colours and replay backspaces/carriage returns like a terminal would."""
    lines = []
    for line in ANSI.sub("", raw).replace("\r\n", "\n").split("\n"):
        out, col = [], 0
        for ch in line:
            if ch == "\b":
                col = max(col - 1, 0)
            elif ch == "\r":
                col = 0
            else:
                out[col:col + 1] = [ch]
                col += 1
        lines.append("".join(out).rstrip())
    return "\n".join(lines)


def read_until_prompt(fd, buf):
    """Read until swipl waits for input; return (buf, kind) with kind in top/port/answer."""
    while True:
        plain = ANSI.sub("", buf)
        tail = clean(plain).split("\n")[-1] + (" " if plain.endswith(" ") else "")
        if select.select([fd], [], [], 0.3)[0]:
            try:
                chunk = os.read(fd, 65536).decode(errors="replace")
            except OSError:
                return buf, "eof"
            if not chunk:
                return buf, "eof"
            buf += chunk
        elif tail.endswith("?- "):
            return buf, "top"
        elif tail.endswith("? "):
            return buf, "port"
        elif re.fullmatch(r"(true|false|\S.* = .*) ", tail):
            return buf, "answer"


def session(pl_file, goal, trace=False, more=0, max_steps=None, stack_limit="1g"):
    """Consult pl_file, run goal; press ';' `more` times; abort the trace after max_steps."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("swipl", ["swipl", "-q", f"--stack-limit={stack_limit}",
                            "-g", "set_prolog_flag(toplevel_prompt, '~m~d~l?- ')", str(pl_file)])
    buf, kind = read_until_prompt(fd, "")
    for cmd in (["trace."] if trace else []) + [goal]:
        os.write(fd, (cmd + "\n").encode())
        buf, kind = read_until_prompt(fd, buf)
        steps = 0
        while kind in ("port", "answer"):
            if kind == "port":
                steps += 1
                creep = trace and not (max_steps and steps > max_steps)
                key = "\n" if creep else "a"  # untraced port = debugger after an error: abort
            else:
                key, more = (";", more - 1) if more > 0 else ("\n", more)
            os.write(fd, key.encode())
            buf, kind = read_until_prompt(fd, buf)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    text = clean(buf)
    return text[:text.rfind("\n")]  # drop the trailing empty prompt


def save(name, *parts):
    (OUT / name).write_text("\n\n".join(p.rstrip() for p in parts) + "\n")
    print(f"--- {name}\n" + (OUT / name).read_text())


def main():
    OUT.mkdir(exist_ok=True)
    q12, q21, q22 = (LAB2 / f"team_qn_{q}.pl" for q in ("1_2", "2_1", "2_2"))
    cyclic = Path(tempfile.mkdtemp()) / "team_qn_2_2_cyclic.pl"
    cyclic.write_text(q22.read_text().replace("% depends_on(driver, pytorch).", "depends_on(driver, pytorch)."))

    save("q1_trace.txt", session(q12, "violates_policy(anna).", trace=True))
    save("q2_1_all.txt", session(q21, "requires(visionapp, P).", more=10))
    save("q2_1_trace.txt", session(q21, "requires(visionapp, driver).", trace=True))
    save("q2_2_trace_a.txt", session(q22, "requires_a(visionapp, driver).", trace=True))
    save("q2_2_trace_b.txt", session(q22, "requires_b(visionapp, driver).", trace=True))
    save("q2_2_all_ab.txt",
         session(q22, "requires_a(visionapp, P).", more=10),
         session(q22, "requires_b(visionapp, P).", more=10))
    # Version A keeps answering forever; a false query loops in constant space (never overflows)
    save("q2_2_cycle_a.txt",
         session(cyclic, "requires_a(visionapp, P).", more=12),
         session(cyclic, "call_with_time_limit(10, requires_a(cuda, opencv))."))
    # Version B never reaches the base clause; choice points pile up until the stack overflows
    # (64 MB so it fails in seconds -- the default 1 GB takes many minutes)
    save("q2_2_cycle_b.txt",
         session(cyclic, "requires_b(visionapp, driver).", trace=True, max_steps=24),
         session(cyclic, "requires_b(visionapp, driver).", stack_limit="64m"))


if __name__ == "__main__":
    main()
