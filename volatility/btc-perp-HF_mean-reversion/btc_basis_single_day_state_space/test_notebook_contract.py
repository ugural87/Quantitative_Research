from __future__ import annotations

import ast
import json
from pathlib import Path


def test_workbench_is_outputs_cleared_and_syntax_valid():
    notebook = json.loads(Path("btc_basis_realdata_workbench.ipynb").read_text())
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]))


def test_workbench_declares_single_day_scope_and_isolates_holdout():
    notebook = json.loads(Path("btc_basis_realdata_workbench.ipynb").read_text())
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "intentionally a single-day methodological research project" in markdown
    assert "separate follow-on project" in markdown
    assert "threshold-development interval" in markdown
    assert "final holdout" in markdown.lower()
    assert "make_intraday_research_split" in code
    assert "development_end_exclusive" in code
    assert "holdout_end_exclusive" in code
    assert "run_proxy_backtest_detailed" in code
    assert "write_research_manifest" in code
    assert "to_availability_bars" in code
    assert "2.0 * (EXECUTION_DELAY_SECONDS + MAX_FILL_DELAY_SECONDS)" in code
    assert "summarize_proxy_result" in code
    assert "THRESHOLD_GRID = np.arange(0.75, STOP_Z, 0.25)" in code
    assert "stop_z=STOP_Z" in code
    assert "max(STOP_Z" not in code
    assert "research_status" in code
    assert "ASSUMED_FEE_BP_PER_FILL" in code
    assert "short-basis exit boundary" in code
