"""Run a notebook from top to bottom in a Jupyter kernel and save the outputs into it.
Not part of the submission.

    python run_notebook.py ../NanoGPT-Math/dpo_reasoning/dpo.ipynb
    python run_notebook.py --smoke      # the same notebook on tiny data, in a temporary folder

The notebook is saved after every cell, so a crash keeps the outputs so far. Everything the cells
print also goes to runs/<notebook name>.log as it happens, so the progress bars can be followed
with tail -f. Run only one of these at a time: the training takes about 4 GB.
"""
import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from run_pilot import FINAL_NOTEBOOK, HERE, PILOTS, ROOT


class LoggingClient(NotebookClient):
    """A NotebookClient that also writes every printed text to a log file."""

    def __init__(self, notebook, log, **kwargs):
        super().__init__(notebook, **kwargs)
        self.log_file = log

    def output(self, outs, msg, display_id, cell_index):
        if msg["msg_type"] == "stream":
            self.log_file.write(msg["content"]["text"])
        elif msg["msg_type"] == "error":
            self.log_file.write("\n".join(msg["content"]["traceback"]) + "\n")
        self.log_file.flush()
        return super().output(outs, msg, display_id, cell_index)


def tidy(outputs):
    """Join neighbouring pieces of the same stream, as Jupyter shows them, and keep only the last
    state of each progress bar (tqdm redraws its line with a carriage return)."""
    joined = []
    for output in outputs:
        if (output.output_type == "stream" and joined and joined[-1].output_type == "stream"
                and joined[-1].name == output.name):
            joined[-1].text += output.text
        else:
            joined.append(output)
    for output in joined:
        if output.output_type == "stream" and "\r" in output.text:
            lines = []
            for line in output.text.split("\n"):
                parts = [part for part in line.split("\r") if part]
                lines.append(parts[-1] if parts else "")
            output.text = "\n".join(lines)
    # tqdm's last newline can arrive on its own after the cell's prints; drop such empty pieces
    return [output for output in joined if output.output_type != "stream" or output.text.strip()]


def save(notebook, path):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    os.replace(path + ".tmp", path)


def run(path):
    path = os.path.abspath(path)
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    env = {k: v for k, v in os.environ.items() if k not in ("MPLBACKEND", "TQDM_DISABLE")}
    start = time.time()
    log_path = f"{HERE}/runs/" + os.path.basename(path).replace(".ipynb", ".log")
    with open(log_path, "w") as log:
        client = LoggingClient(notebook, log, timeout=None, kernel_name="python3",
                               resources={"metadata": {"path": os.path.dirname(path)}})
        with client.setup_kernel(env=env):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                log.write(f"\n===== cell {index} at {time.strftime('%H:%M:%S')} =====\n")
                log.flush()
                try:
                    client.execute_cell(cell, index)
                except CellExecutionError:
                    minutes = (time.time() - start) / 60
                    print(f"cell {index} failed after {minutes:.0f} minutes", flush=True)
                    return 1
                finally:  # also after Ctrl-C or a dead kernel, so the outputs so far are kept
                    cell.outputs = tidy(cell.outputs)
                    save(notebook, path)
        log.write(f"\n===== done after {(time.time() - start) / 60:.0f} minutes =====\n")
    print(f"{path}: done after {(time.time() - start) / 60:.0f} minutes", flush=True)
    return 0


def smoke():
    """Copy the final notebook into a temporary folder, make the data tiny, and run it."""
    folder = f"{ROOT}/smoke_run"
    os.makedirs(folder, exist_ok=True)
    notebook = nbformat.read(FINAL_NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for old, new in PILOTS["smoke"]:
                cell.source = cell.source.replace(old, new)
    nbformat.write(notebook, f"{folder}/dpo.ipynb")
    exit_code = run(f"{folder}/dpo.ipynb")
    print(f"look at {folder}/dpo.ipynb, then delete {folder}")
    return exit_code


if __name__ == "__main__":
    sys.exit(smoke() if sys.argv[1] == "--smoke" else run(sys.argv[1]))
