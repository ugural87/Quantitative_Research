from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
MODEL_NOTEBOOKS = [
    ROOT / "01_four_factor_svensson_irrbb_engine.ipynb",
    ROOT / "02_three_factor_nelson_siegel_irrbb_engine.ipynb",
]


def execute_notebook(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute both IRRBB engines and regenerate the Markdown-only case study.")
    parser.add_argument("--data-mode", choices=["AUTO", "LIVE", "OFFLINE"], default="AUTO")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--report-only", action="store_true", help="Skip model execution and rebuild from existing artifacts.")
    args = parser.parse_args()

    os.environ["IRRBB_DATA_MODE"] = args.data_mode
    if not args.report_only:
        for path in MODEL_NOTEBOOKS:
            execute_notebook(path, args.timeout)

    from build_report import main as build_report
    build_report()


if __name__ == "__main__":
    main()
