"""Fill helpers/report.html with the .pl files and traces, then print it to PDF with Chrome.

Not part of the submission. Run make_traces.py first.
Usage: python3 helpers/build_report.py   -> writes team_Assignment3.pdf
"""
import html
import re
import subprocess
import tempfile
from pathlib import Path

HELPERS = Path(__file__).resolve().parent
LAB2 = HELPERS.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def fill(match):
    kind, name = match.groups()
    path = (HELPERS / "traces" if kind == "TRACE" else LAB2) / name
    return html.escape(path.read_text().rstrip())


def main():
    page = re.sub(r"\{\{(TRACE|PL):([\w.]+)\}\}", fill, (HELPERS / "report.html").read_text())
    src = Path(tempfile.mkdtemp()) / "report.html"
    src.write_text(page)
    pdf = LAB2 / "team_Assignment3.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", src.as_uri()], check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main()
