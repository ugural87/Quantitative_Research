from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
FIG = ART / "figures"
OUT = ROOT / "03_model_risk_business_architecture_case_study.ipynb"

REQUIRED_MODEL_FIGURES = [
    "01_curve_fit.png",
    "02_factor_alignment.png",
    "03_identification.png",
    "05_nmd_profile.png",
    "06_repricing_gap.png",
    "07_eve_scenarios.png",
    "08_nii.png",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(key: str) -> tuple[dict, dict]:
    model_dir = ART / key
    metrics = load_json(model_dir / "metrics.json")
    manifest = load_json(model_dir / "run_manifest.json")
    if metrics.get("model_key") != key or manifest.get("model_key") != key:
        raise ValueError(f"Model-key mismatch in {model_dir}")
    for name, expected_hash in manifest.get("tables", {}).items():
        path = model_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Stale or modified table artifact: {path}")
    for name, expected_hash in manifest.get("figures", {}).items():
        path = FIG / name
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ValueError(f"Stale or modified figure artifact: {path}")
    return metrics, manifest


def validate_cross_model_contract(nss: dict, ns: dict, nss_manifest: dict, ns_manifest: dict) -> None:
    required = [
        "artifact_schema_version", "model_key", "model_name", "data_source",
        "dataset_fingerprint", "report_group_id", "reference_date", "observations",
        "train_observations", "validation_observations", "test_observations",
        "test_mean_rmse_bp", "test_p95_rmse_bp", "design_condition_number",
        "maximum_loading_correlation", "level_factor_abs_correlation",
        "matched_factor_abs_correlation_mean", "tier1_capital_mm", "base_eve_mm",
        "worst_eve_scenario", "maximum_adverse_delta_eve_mm",
        "maximum_adverse_delta_eve_to_tier1_pct", "base_nii_mm",
        "delta_nii_parallel_up_mm", "delta_nii_parallel_down_mm",
    ]
    for model in [nss, ns]:
        missing = [key for key in required if key not in model]
        if missing:
            raise KeyError(f"Missing metrics in {model.get('model_key')}: {missing}")

    exact_common = [
        "data_source", "dataset_fingerprint", "report_group_id", "reference_date",
        "observations", "train_observations", "validation_observations",
        "test_observations", "tier1_capital_mm",
    ]
    mismatches = {key: (nss[key], ns[key]) for key in exact_common if nss[key] != ns[key]}
    if mismatches:
        raise ValueError(f"The two model runs are not comparable: {mismatches}")
    if nss_manifest["dataset_fingerprint"] != ns_manifest["dataset_fingerprint"]:
        raise ValueError("Manifest dataset fingerprints differ")
    if nss_manifest["report_group_id"] != ns_manifest["report_group_id"]:
        raise ValueError("Manifest report-group IDs differ")

    for key in ["nss4", "ns3"]:
        for suffix in REQUIRED_MODEL_FIGURES:
            path = FIG / f"{key}_{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)


def read_csv(key: str, name: str) -> pd.DataFrame:
    path = ART / key / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def create_comparison_figures(nss: dict, ns: dict) -> None:
    labels = ["Four-factor NSS", "Three-factor NS"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(labels, [nss["test_mean_rmse_bp"], ns["test_mean_rmse_bp"]])
    axes[0].set_title("Held-out curve reconstruction")
    axes[0].set_ylabel("Mean test RMSE (bp)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, [nss["design_condition_number"], ns["design_condition_number"]])
    axes[1].set_title("Numerical conditioning")
    axes[1].set_ylabel("Design condition number")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(labels, [nss["level_factor_abs_correlation"], ns["level_factor_abs_correlation"]])
    axes[2].set_title("Direct PC1–Δβ0 alignment")
    axes[2].set_ylabel("Absolute correlation")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=15)
    fig.suptitle("Model trade-off: fit, identification and empirical alignment")
    fig.tight_layout()
    fig.savefig(FIG / "comparison_01_model_tradeoff.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    nss_eve = read_csv("nss4", "eve_scenarios.csv")
    ns_eve = read_csv("ns3", "eve_scenarios.csv")
    scenario_col = "scenario" if "scenario" in nss_eve.columns else nss_eve.columns[0]
    nss_eve = nss_eve.set_index(scenario_col)
    scenario_col_ns = "scenario" if "scenario" in ns_eve.columns else ns_eve.columns[0]
    ns_eve = ns_eve.set_index(scenario_col_ns)
    scenario_order = [x for x in ["parallel_up", "parallel_down", "short_up", "short_down", "steepener", "flattener"] if x in nss_eve.index and x in ns_eve.index]
    x = np.arange(len(scenario_order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width/2, nss_eve.loc[scenario_order, "delta_EVE_mm"].astype(float), width, label="Four-factor NSS")
    ax.bar(x + width/2, ns_eve.loc[scenario_order, "delta_EVE_mm"].astype(float), width, label="Three-factor NS")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x, scenario_order, rotation=25)
    ax.set_ylabel("ΔEVE (mm)")
    ax.set_title("EVE scenario comparison from current model artifacts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "comparison_02_eve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    nss_nmd = read_csv("nss4", "nmd_sensitivity.csv")
    ns_nmd = read_csv("ns3", "nmd_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(nss_nmd["core_scale"].astype(float), nss_nmd["worst_loss_to_Tier1_pct"].astype(float), marker="o", label="Four-factor NSS")
    ax.plot(ns_nmd["core_scale"].astype(float), ns_nmd["worst_loss_to_Tier1_pct"].astype(float), marker="o", label="Three-factor NS")
    ax.axhline(15.0, linestyle="--", linewidth=1.0, label="15% Tier 1 threshold")
    ax.set_xlabel("Core NMD assumption multiplier")
    ax.set_ylabel("Maximum adverse ΔEVE / Tier 1 (%)")
    ax.set_title("NMD behavioural-assumption sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "comparison_03_nmd_sensitivity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def factor_narrative(nss: dict, ns: dict) -> str:
    level_diff = ns["level_factor_abs_correlation"] - nss["level_factor_abs_correlation"]
    matched_diff = ns["matched_factor_abs_correlation_mean"] - nss["matched_factor_abs_correlation_mean"]
    if level_diff > 0.15 and matched_diff > 0.15:
        return (
            "On this run, the three-factor model has materially stronger empirical factor alignment on both the direct "
            "PC1–Δβ0 diagnostic and the matched-factor summary."
        )
    if abs(level_diff) < 0.10:
        return (
            "The direct PC1–Δβ0 diagnostic does not identify a decisive winner on this run. The matched-factor score must "
            "also be interpreted cautiously because the four-factor model can match three PCs against the best three of four betas, "
            "whereas all three NS betas participate in the three-by-three assignment."
        )
    if matched_diff < -0.05:
        return (
            "The unconstrained matched-factor summary is higher for the four-factor model on this run, but that comparison is not "
            "dimension-neutral: three PCs are matched against four candidate betas. The result therefore does not overturn the separate "
            "conditioning and VIF evidence."
        )
    return (
        "Empirical factor alignment is mixed on this run. It should be treated as supporting evidence rather than the sole model-selection criterion."
    )


def recommendation_narrative(nss: dict, ns: dict) -> str:
    fit_gain = ns["test_mean_rmse_bp"] - nss["test_mean_rmse_bp"]
    fit_gain_pct = 100 * fit_gain / ns["test_mean_rmse_bp"]
    condition_ratio = nss["design_condition_number"] / ns["design_condition_number"]
    return (
        f"The four-factor NSS model reduces mean test RMSE by **{fmt(fit_gain)} bp** "
        f"(**{fmt(fit_gain_pct, 1)}%** relative to NS), but its design matrix is **{fmt(condition_ratio, 1)}×** more ill-conditioned. "
        "The defensible architecture is therefore use-case specific: retain NSS as a flexible reconstruction/valuation challenger, and use the "
        "three-factor NS model as the parsimonious structural benchmark for governance and attribution unless additional evidence demonstrates "
        "stable incremental value from the fourth factor."
    )


def attachment_cell(markdown: str, figure_names: list[str] | None = None):
    cell = nbf.v4.new_markdown_cell(markdown)
    if figure_names:
        attachments = {}
        for name in figure_names:
            path = FIG / name
            if not path.exists():
                raise FileNotFoundError(path)
            attachments[name] = {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        cell["attachments"] = attachments
    return cell


def build_notebook(nss: dict, ns: dict, nss_manifest: dict, ns_manifest: dict) -> None:
    factor_text = factor_narrative(nss, ns)
    recommendation = recommendation_narrative(nss, ns)
    condition_ratio = nss["design_condition_number"] / ns["design_condition_number"]
    eve_difference = abs(nss["maximum_adverse_delta_eve_mm"] - ns["maximum_adverse_delta_eve_mm"])
    tier1 = nss["tier1_capital_mm"]

    nss_nmd = read_csv("nss4", "nmd_sensitivity.csv")
    ns_nmd = read_csv("ns3", "nmd_sensitivity.csv")
    base_nss = nss_nmd.loc[np.isclose(nss_nmd["core_scale"].astype(float), 1.0)].iloc[0]
    low_nss = nss_nmd.loc[nss_nmd["core_scale"].astype(float).idxmin()]
    high_nss = nss_nmd.loc[nss_nmd["core_scale"].astype(float).idxmax()]

    cells = []
    cells.append(attachment_cell(f'''# IRRBB model-risk and business-architecture case study

**Automatically generated from the current Notebook 1 and Notebook 2 artifact contracts.**

| Run field | Value |
|---|---|
| Data source | {nss["data_source"]} |
| Reference date | {nss["reference_date"]} |
| Observations | {int(nss["observations"]):,} |
| Train / validation / test | {int(nss["train_observations"]):,} / {int(nss["validation_observations"]):,} / {int(nss["test_observations"]):,} |
| Dataset fingerprint | `{nss["dataset_fingerprint"][:16]}…` |
| Report group ID | `{nss["report_group_id"][:16]}…` |

This report is intentionally Markdown-only. It is not edited manually after model execution; `build_report.py` regenerates its tables, interpretations and embedded figures from versioned artifacts.'''))

    cells.append(attachment_cell('''## 1. Why IRRBB is a business problem before it is a curve-fitting problem

IRRBB is the risk to a bank's **capital and earnings** from adverse changes in interest rates. The architecture must cover gap risk, basis risk and option risk across complete contractual and behavioural cash flows. Curve representation is therefore only one controlled component inside governance, data, behavioural modelling, measurement, limits, management action and independent validation.

![Target IRRBB architecture](attachment:business_architecture.png)''', ["business_architecture.png"]))

    cells.append(attachment_cell('''## 2. Target operating and model architecture

The project separates four uses that should not automatically be forced into one model:

1. **Approved valuation curve:** present-value and cash-flow discounting.
2. **Structural factor model:** level, slope and curvature attribution used in risk explanation.
3. **Challenger model:** additional flexibility and model-form challenge.
4. **FTP architecture:** benchmark curve plus funding, liquidity, basis, optionality and regulatory transfer-pricing adjustments.

The two notebooks deliberately feed the same behavioural NMD model, reconciled synthetic banking book, EVE engine, NII engine and prescribed scenario set. This isolates curve-model form as the principal experimental difference.'''))

    cells.append(attachment_cell('''## 3. Automated analytical workflow and artifact contract

```python
# Notebook 1 and Notebook 2 both write the same contract
artifacts/<model_key>/metrics.json
artifacts/<model_key>/run_manifest.json
artifacts/<model_key>/eve_scenarios.csv
artifacts/<model_key>/nmd_sensitivity.csv
artifacts/figures/<model_key>_*.png
```

The report builder performs four controls before creating this notebook:

- validates required metrics and files;
- verifies SHA-256 hashes in each run manifest;
- requires matching data fingerprints, report-group IDs, dates, sample sizes and Tier 1 inputs;
- regenerates all comparison figures and embeds every PNG as a notebook attachment.

A stale report can therefore no longer be silently combined with newer model outputs.'''))

    cells.append(attachment_cell(f'''## 4. Temporal validation design

The common sample is split chronologically:

- **Train:** {int(nss["train_observations"]):,} observations — decay hyperparameter calibration;
- **Validation:** {int(nss["validation_observations"]):,} observations — candidate selection;
- **Test:** {int(nss["test_observations"]):,} observations — untouched final evaluation.

PCA is fitted on training-period yield changes and the held-out period is transformed into that frozen PCA space. The daily beta estimates are cross-sectional decompositions of each observed curve after the global decay parameters have been selected; they are not forecasts of future rates.'''))

    cells.append(attachment_cell(f'''## 5. Current model results

| Metric | Four-factor Svensson | Three-factor Nelson–Siegel | Current interpretation |
|---|---:|---:|---|
| Test mean RMSE | {fmt(nss["test_mean_rmse_bp"])} bp | {fmt(ns["test_mean_rmse_bp"])} bp | Lower is better for reconstruction. |
| Test 95th-percentile RMSE | {fmt(nss["test_p95_rmse_bp"])} bp | {fmt(ns["test_p95_rmse_bp"])} bp | Held-out tail fit. |
| Design condition number | {fmt(nss["design_condition_number"], 1)} | {fmt(ns["design_condition_number"], 1)} | NSS is {fmt(condition_ratio, 1)}× more ill-conditioned. |
| Maximum loading correlation | {fmt(nss["maximum_loading_correlation"])} | {fmt(ns["maximum_loading_correlation"])} | Higher values indicate stronger loading competition. |
| abs(corr(PC1, Δβ0)) | {fmt(nss["level_factor_abs_correlation"])} | {fmt(ns["level_factor_abs_correlation"])} | Direct level-factor diagnostic. |
| Mean matched PC–beta correlation | {fmt(nss["matched_factor_abs_correlation_mean"])} | {fmt(ns["matched_factor_abs_correlation_mean"])} | Interpret with dimensionality caveat. |
| Worst ΔEVE / Tier 1 | {fmt(nss["maximum_adverse_delta_eve_to_tier1_pct"])}% | {fmt(ns["maximum_adverse_delta_eve_to_tier1_pct"])}% | Current capital sensitivity. |
| ΔNII parallel up | {fmt(nss["delta_nii_parallel_up_mm"])} mm | {fmt(ns["delta_nii_parallel_up_mm"])} mm | Earnings response. |
| ΔNII parallel down | {fmt(nss["delta_nii_parallel_down_mm"])} mm | {fmt(ns["delta_nii_parallel_down_mm"])} mm | Earnings response. |

{factor_text}

![Model trade-off](attachment:comparison_01_model_tradeoff.png)''', ["comparison_01_model_tradeoff.png"]))

    cells.append(attachment_cell(f'''## 6. Diagnostic evidence from each engine

### Four-factor Nelson–Siegel–Svensson

![Svensson curve fit](attachment:nss4_01_curve_fit.png)

![Svensson factor alignment](attachment:nss4_02_factor_alignment.png)

![Svensson identification](attachment:nss4_03_identification.png)

The current NSS run obtains the better held-out curve fit, but the loading matrix is materially less stable. Its reported condition number is **{fmt(nss["design_condition_number"], 1)}**, versus **{fmt(ns["design_condition_number"], 1)}** for NS. The individual beta coefficients should therefore not be treated as automatically reliable economic risk factors merely because the reconstructed curve is accurate.

### Three-factor Nelson–Siegel

![NS curve fit](attachment:ns3_01_curve_fit.png)

![NS factor alignment](attachment:ns3_02_factor_alignment.png)

![NS identification](attachment:ns3_03_identification.png)

The NS specification accepts a reconstruction penalty in exchange for a much more parsimonious and numerically stable factor basis. {factor_text}
''', [
        "nss4_01_curve_fit.png", "nss4_02_factor_alignment.png", "nss4_03_identification.png",
        "ns3_01_curve_fit.png", "ns3_02_factor_alignment.png", "ns3_03_identification.png",
    ]))

    cells.append(attachment_cell(f'''## 7. EVE, NMD and NII implications

![EVE comparison](attachment:comparison_02_eve.png)

The current worst prescribed scenario is **{nss["worst_eve_scenario"]}** for NSS and **{ns["worst_eve_scenario"]}** for NS. Maximum adverse losses are **{fmt(nss["maximum_adverse_delta_eve_mm"])} mm** and **{fmt(ns["maximum_adverse_delta_eve_mm"])} mm**, against Tier 1 capital of **{fmt(tier1, 1)} mm**. Their absolute worst-loss difference is only **{fmt(eve_difference)} mm**.

This does not validate unstable factor coefficients. EVE is a portfolio-level functional of the fitted curve; offsetting coefficients can preserve valuation while weakening attribution and management explanation.

### Behavioural-assumption sensitivity

![NMD sensitivity](attachment:comparison_03_nmd_sensitivity.png)

For the NSS run, the {float(low_nss["core_scale"]):.1f}× core-NMD assumption produces **{fmt(low_nss["worst_loss_to_Tier1_pct"])}%**, the base assumption produces **{fmt(base_nss["worst_loss_to_Tier1_pct"])}%**, and the {float(high_nss["core_scale"]):.1f}× assumption produces **{fmt(high_nss["worst_loss_to_Tier1_pct"])}%**. The behavioural NMD assumption therefore has a much larger capital effect than the difference between the two curve representations.

### NMD and earnings architecture

![NMD profile](attachment:ns3_05_nmd_profile.png)

![NII results](attachment:ns3_08_nii.png)

The parallel-shock ΔNII figures are driven principally by contractual repricing, deposit betas, lags and hedging. The small difference in base NII across curve models is distinct from the shocked-minus-base earnings sensitivity.''', [
        "comparison_02_eve.png", "comparison_03_nmd_sensitivity.png", "ns3_05_nmd_profile.png", "ns3_08_nii.png",
    ]))

    cells.append(attachment_cell(f'''## 8. Model-risk decision

{recommendation}

### Governance assignment

| Use case | Recommended role |
|---|---|
| Curve reconstruction / valuation challenge | Four-factor NSS challenger |
| Structural benchmark and management attribution | Three-factor NS benchmark |
| Final discounting curve | Separate approved market-data/curve service |
| FTP | Separate curve stack with liquidity, funding, basis and optionality components |

The recommendation is conditional on the current artifacts. If later runs change the fit, conditioning, factor-alignment or downstream materiality evidence, this section is regenerated automatically rather than preserving a stale conclusion.'''))

    cells.append(attachment_cell('''## 9. Limitations and production requirements

This remains a portfolio-grade research prototype. A bank production implementation would require approved instrument-level curve construction, full product cash-flow conventions, account-level NMD and option models, basis-risk architecture, dynamic-balance-sheet NII, multi-currency/legal-entity aggregation, independent implementation validation, data lineage, access controls, monitoring and formal model-governance approval.'''))

    cells.append(attachment_cell(f'''## 10. Reproducibility, lineage and references

### Source artifacts

- `artifacts/nss4/metrics.json` and `run_manifest.json`
- `artifacts/ns3/metrics.json` and `run_manifest.json`
- model-specific CSV tables and PNG diagnostics
- comparison figures regenerated by `build_report.py`

The report was built only after the two manifests matched on dataset fingerprint `{nss["dataset_fingerprint"][:16]}…`, report-group ID `{nss["report_group_id"][:16]}…`, reference date **{nss["reference_date"]}**, sample sizes and Tier 1 capital.

### References

1. Basel Committee on Banking Supervision, **SRP31 — Interest rate risk in the banking book**, effective 1 January 2026.
2. Basel Committee on Banking Supervision, **Recalibration of shocks for IRRBB**, 16 July 2024.
3. Basel Committee on Banking Supervision, **SRP98 — Application guidance on IRRBB**.
4. Basel Committee on Banking Supervision, **DIS70 — IRRBB disclosure requirements**.
5. Federal Reserve Board, **Gürkaynak–Sack–Wright nominal yield-curve data**.
'''))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "report_generation": {
            "builder": "build_report.py",
            "dataset_fingerprint": nss["dataset_fingerprint"],
            "report_group_id": nss["report_group_id"],
            "reference_date": nss["reference_date"],
        },
    })
    if any(cell.cell_type != "markdown" for cell in nb.cells):
        raise AssertionError("The report notebook must remain Markdown-only")
    nbf.write(nb, OUT)


def main() -> None:
    nss, nss_manifest = load_model("nss4")
    ns, ns_manifest = load_model("ns3")
    validate_cross_model_contract(nss, ns, nss_manifest, ns_manifest)
    create_comparison_figures(nss, ns)
    # Comparison figures are generated after model-manifest validation and embedded directly.
    build_notebook(nss, ns, nss_manifest, ns_manifest)
    print(f"Generated Markdown-only report: {OUT}")
    print(f"Data fingerprint: {nss['dataset_fingerprint']}")
    print(f"Reference date: {nss['reference_date']}")


if __name__ == "__main__":
    main()
