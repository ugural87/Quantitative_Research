from __future__ import annotations

import base64
import json
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path('/mnt/data/irrbb_portfolio_project')
DATA = ROOT / 'data'
ART = ROOT / 'artifacts'
FIG = ART / 'figures'
for p in [ROOT, DATA, ART, FIG]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Deterministic offline reference panel. It is deliberately and visibly labelled
# synthetic. The notebooks default to AUTO and use the official Fed endpoint when
# available; this file only makes the delivered project executable without network.
# -----------------------------------------------------------------------------
TENORS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30.0])
LABELS = [f'{int(t)}Y' for t in TENORS]

def ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)
    x = np.maximum(tau, 1e-8) / lam
    slope = -np.expm1(-x) / x
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curvature])

def nss_loadings(tau, lam1, lam2):
    tau = np.asarray(tau, dtype=float)
    x1 = np.maximum(tau, 1e-8) / lam1
    x2 = np.maximum(tau, 1e-8) / lam2
    slope = -np.expm1(-x1) / x1
    c1 = slope - np.exp(-x1)
    c2 = -np.expm1(-x2) / x2 - np.exp(-x2)
    return np.column_stack([np.ones_like(tau), slope, c1, c2])

def ou_path(rng, n, x0, mu, theta, sigma):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu - x[i-1]) + sigma * rng.standard_normal()
    return x

rng = np.random.default_rng(20260724)
n = 1000
b0 = ou_path(rng, n, 4.20, 4.30, 0.010, 0.045)
b1 = ou_path(rng, n, -1.10, -0.80, 0.015, 0.035)
b2 = ou_path(rng, n, 0.80, 0.60, 0.020, 0.030)
b3 = ou_path(rng, n, 0.05, 0.00, 0.080, 0.040)
Y = np.column_stack([b0, b1, b2]) @ ns_loadings(TENORS, 2.20).T
Y += np.outer(b3, nss_loadings(TENORS, 2.20, 14.0)[:, 3])
for start, amp in [(220, 0.12), (600, -0.10), (820, 0.08)]:
    pulse = amp * np.exp(-0.5 * ((np.arange(n) - start) / 40.0) ** 2)
    Y += np.outer(pulse, nss_loadings(TENORS, 2.20, 16.0)[:, 3])
Y += rng.normal(0.0, 0.007, size=Y.shape)
Y = np.clip(Y, 0.05, None)
dates = pd.bdate_range(end='2026-07-17', periods=n)
offline = pd.DataFrame(Y, index=dates, columns=LABELS)
offline.index.name = 'Date'
offline.to_csv(DATA / 'offline_reference_zero_curve.csv')

# -----------------------------------------------------------------------------
# Notebook cell builders
# -----------------------------------------------------------------------------
def md(s: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())

def code(s: str):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())

COMMON_IMPORTS = r'''
from __future__ import annotations

import io
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.decomposition import PCA
from IPython.display import display

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 115,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class ProjectConfig:
    currency: str = "USD"
    lookback_years: int = 4
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    tier1_capital_mm: float = 170.0
    accounting_equity_mm: float = 200.0
    outlier_limit: float = 0.15
    shock_decay_years: float = 4.0
    post_shock_floor_pct: float | None = None
    nii_horizon_months: int = 12
    data_mode: str = os.getenv("IRRBB_DATA_MODE", "AUTO").upper()

CFG = ProjectConfig()
ZERO_TENORS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
ZERO_LABELS = [f"{int(t)}Y" for t in ZERO_TENORS]
ZERO_COLUMNS = [f"SVENY{int(t):02d}" for t in ZERO_TENORS]
FED_ZERO_CURVE_CSV = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

# BCBS shock table effective 1 January 2026 (basis points).
SHOCK_CONFIG_BP = {
    "USD": {"parallel": 200.0, "short": 300.0, "long": 225.0},
    "TRY": {"parallel": 400.0, "short": 500.0, "long": 300.0},
}
if CFG.currency not in SHOCK_CONFIG_BP:
    raise ValueError(f"Unsupported shock currency: {CFG.currency}")
'''

DATA_CODE = r'''
def load_fed_gsw_zero_curve(url=FED_ZERO_CURVE_CSV, lookback_years=CFG.lookback_years):
    """Load official continuously compounded SVENY zero-coupon yields."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.federalreserve.gov/data/nominal-yield-curve.htm",
    })
    response = session.get(url, timeout=30)
    response.raise_for_status()
    lines = response.content.decode("utf-8-sig").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    raw = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    missing = sorted(set(ZERO_COLUMNS) - set(raw.columns))
    if missing:
        raise KeyError(f"Missing official SVENY columns: {missing}")
    panel = (
        raw.set_index("Date")[ZERO_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-999.99, np.nan)
    )
    panel.columns = ZERO_LABELS
    cutoff = panel.index.max() - pd.DateOffset(years=lookback_years)
    panel = panel.loc[panel.index >= cutoff].sort_index().dropna(how="all")
    panel = panel.interpolate(limit=3, limit_direction="both").dropna()
    if len(panel) < 400:
        raise ValueError(f"Curve panel too short after cleaning: {len(panel)} rows")
    return panel


def load_zero_curve_panel():
    offline_path = DATA_DIR / "offline_reference_zero_curve.csv"
    if CFG.data_mode == "OFFLINE":
        panel = pd.read_csv(offline_path, index_col="Date", parse_dates=True)
        return panel[ZERO_LABELS], "OFFLINE synthetic reference panel — not market data"
    try:
        panel = load_fed_gsw_zero_curve()
        panel.to_csv(DATA_DIR / "latest_fed_gsw_zero_curve.csv")
        return panel, "Federal Reserve GSW SVENY zero-coupon curve (live)"
    except Exception as exc:
        if CFG.data_mode == "LIVE":
            raise
        panel = pd.read_csv(offline_path, index_col="Date", parse_dates=True)
        note = f"OFFLINE synthetic reference panel — live pull failed: {type(exc).__name__}"
        return panel[ZERO_LABELS], note


zero_yields, DATA_SOURCE = load_zero_curve_panel()
zero_yields = zero_yields.sort_index().astype(float)

n_obs = len(zero_yields)
train_end = int(n_obs * CFG.train_fraction)
validation_end = int(n_obs * (CFG.train_fraction + CFG.validation_fraction))
if not (200 <= train_end < validation_end < n_obs):
    raise ValueError("Invalid train/validation/test split")

train_panel = zero_yields.iloc[:train_end]
validation_panel = zero_yields.iloc[train_end:validation_end]
test_panel = zero_yields.iloc[validation_end:]

print(f"Data source       : {DATA_SOURCE}")
print(f"Observations      : {n_obs:,}")
print(f"Date range        : {zero_yields.index.min().date()} -> {zero_yields.index.max().date()}")
print(f"Train / val / test: {len(train_panel)} / {len(validation_panel)} / {len(test_panel)}")
display(zero_yields.tail())
'''

MODEL_DIAGNOSTIC_CODE = r'''
def fit_panel(panel_values, design):
    betas = np.linalg.lstsq(design, panel_values.T, rcond=None)[0].T
    fitted = betas @ design.T
    return betas, fitted


def rmse_by_date_bp(actual, fitted):
    return np.sqrt(np.mean((fitted - actual) ** 2, axis=1)) * 100.0


def variance_inflation_factors(design):
    values = {}
    for j in range(design.shape[1]):
        y = design[:, j]
        X = np.delete(design, j, axis=1)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ coef
        sse = float(residual @ residual)
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
        values[f"loading_{j}"] = 1.0 / max(1.0 - r2, 1e-12)
    return values


def max_off_diagonal_abs_corr(matrix):
    corr = np.corrcoef(matrix.T)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.max(np.abs(corr[mask])))


def build_factor_alignment(params, beta_columns):
    train_changes = train_panel.diff().dropna()
    test_changes = test_panel.diff().dropna()
    pca = PCA(n_components=3).fit(train_changes.values)
    test_scores = pca.transform(test_changes.values)
    beta_changes = params[beta_columns].diff().reindex(test_changes.index).dropna()
    common = beta_changes.index.intersection(test_changes.index)
    scores = pd.DataFrame(
        pca.transform(test_changes.loc[common].values),
        index=common,
        columns=["PC1", "PC2", "PC3"],
    )
    beta_delta = beta_changes.loc[common]
    corr = pd.DataFrame(index=scores.columns, columns=beta_columns, dtype=float)
    for pc in scores.columns:
        for beta in beta_columns:
            corr.loc[pc, beta] = np.corrcoef(scores[pc], beta_delta[beta])[0, 1]
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr.values))
    matched = [abs(float(corr.values[r, c])) for r, c in zip(row_idx, col_idx)]
    return pca, corr, matched


def curve_shape_checks(params, beta_columns, rate_function):
    grid = np.linspace(0.05, 30.0, 600)
    sample_dates = list(params.index[-min(40, len(params)):])
    monotonicity_violations = 0
    min_forward, max_forward = np.inf, -np.inf
    for date in sample_dates:
        beta = params.loc[date, beta_columns].values
        z_pct = rate_function(beta, grid)
        discount = np.exp(-(z_pct / 100.0) * grid)
        monotonicity_violations += int(np.any(np.diff(discount) > 1e-10))
        dz_dt = np.gradient(z_pct / 100.0, grid)
        forward = z_pct / 100.0 + grid * dz_dt
        min_forward = min(min_forward, float(np.min(forward)))
        max_forward = max(max_forward, float(np.max(forward)))
    return {
        "sampled_curve_dates": len(sample_dates),
        "discount_monotonicity_violations": monotonicity_violations,
        "minimum_instantaneous_forward_pct": 100.0 * min_forward,
        "maximum_instantaneous_forward_pct": 100.0 * max_forward,
    }
'''

NMD_CODE = r'''
NMD_SEGMENTS = pd.DataFrame([
    {
        "segment": "Retail transactional",
        "target_current_mm": 360.0,
        "core_cap": 0.90,
        "average_maturity_cap_y": 5.0,
        "beta_up": 0.15,
        "beta_down": 0.08,
        "repricing_lag_months": 1,
        "ladder_tenors_y": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        "ladder_weights": [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
    },
    {
        "segment": "Retail non-transactional",
        "target_current_mm": 190.0,
        "core_cap": 0.70,
        "average_maturity_cap_y": 4.5,
        "beta_up": 0.35,
        "beta_down": 0.22,
        "repricing_lag_months": 1,
        "ladder_tenors_y": [0.25, 0.5, 1.0, 2.0, 3.0, 4.5],
        "ladder_weights": [0.10, 0.10, 0.15, 0.25, 0.25, 0.15],
    },
    {
        "segment": "Wholesale",
        "target_current_mm": 100.0,
        "core_cap": 0.50,
        "average_maturity_cap_y": 4.0,
        "beta_up": 0.65,
        "beta_down": 0.50,
        "repricing_lag_months": 0,
        "ladder_tenors_y": [1/12, 0.25, 0.5, 1.0, 2.0, 4.0],
        "ladder_weights": [0.20, 0.15, 0.15, 0.20, 0.20, 0.10],
    },
])


def simulate_segment_balance(target_current_mm, seed, n_days=2520):
    """Ten-year illustrative history; production calibration requires account data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    trend = 0.010 * t
    seasonal = 9.0 * np.sin(2.0 * np.pi * t / 252.0)
    innovations = rng.normal(0.0, 1.0, n_days)
    persistent = np.empty(n_days)
    persistent[0] = 0.0
    for i in range(1, n_days):
        persistent[i] = 0.985 * persistent[i - 1] + innovations[i]
    raw = 430.0 + trend + seasonal + 2.0 * persistent
    raw = np.maximum(raw, 50.0)
    raw *= target_current_mm / raw[-1]
    dates = pd.bdate_range(end=zero_yields.index[-1], periods=n_days)
    return pd.Series(raw, index=dates)


def calibrate_nmd_segments():
    summaries, histories, ladder_rows = [], {}, []
    for i, cfg in NMD_SEGMENTS.iterrows():
        hist = simulate_segment_balance(cfg.target_current_mm, seed=500 + i)
        histories[cfg.segment] = hist
        current = float(hist.iloc[-1])
        stable_volume = float(hist.quantile(0.05))
        # Volume stability and rate stickiness are separated. This is a transparent
        # illustrative rule, not a claim of regulatory approval.
        rate_stickiness = 1.0 - max(cfg.beta_up, cfg.beta_down)
        estimated_core = stable_volume * rate_stickiness
        cap_amount = cfg.core_cap * current
        core = min(estimated_core, cap_amount)
        noncore = current - core
        tenors = np.asarray(cfg.ladder_tenors_y, dtype=float)
        weights = np.asarray(cfg.ladder_weights, dtype=float)
        weights = weights / weights.sum()
        avg_maturity = float(np.sum(tenors * weights))
        if avg_maturity > cfg.average_maturity_cap_y + 1e-12:
            raise ValueError(f"NMD ladder exceeds Basel average-maturity cap for {cfg.segment}")
        summaries.append({
            "segment": cfg.segment,
            "current_mm": current,
            "stable_volume_mm": stable_volume,
            "estimated_core_mm": estimated_core,
            "governance_cap_mm": cap_amount,
            "core_mm": core,
            "noncore_mm": noncore,
            "core_ratio": core / current,
            "weighted_core_maturity_y": avg_maturity,
            "maturity_cap_y": cfg.average_maturity_cap_y,
            "beta_up": cfg.beta_up,
            "beta_down": cfg.beta_down,
            "repricing_lag_months": int(cfg.repricing_lag_months),
        })
        ladder_rows.append({
            "segment": cfg.segment,
            "bucket": "overnight non-core",
            "notional_mm": noncore,
            "behavioural_maturity_y": 1.0 / 365.0,
            "beta_up": cfg.beta_up,
            "beta_down": cfg.beta_down,
            "repricing_lag_months": int(cfg.repricing_lag_months),
            "core_flag": False,
        })
        for tenor, weight in zip(tenors, weights):
            ladder_rows.append({
                "segment": cfg.segment,
                "bucket": f"core {tenor:g}Y",
                "notional_mm": core * weight,
                "behavioural_maturity_y": float(tenor),
                "beta_up": cfg.beta_up,
                "beta_down": cfg.beta_down,
                "repricing_lag_months": int(cfg.repricing_lag_months),
                "core_flag": True,
            })
    return pd.DataFrame(summaries), histories, pd.DataFrame(ladder_rows)


nmd_summary, nmd_histories, nmd_ladder = calibrate_nmd_segments()
CURRENT_NMD_MM = float(nmd_summary.current_mm.sum())
NMD_CORE_MM = float(nmd_summary.core_mm.sum())
NMD_NONCORE_MM = float(nmd_summary.noncore_mm.sum())
NMD_CORE_RATIO = NMD_CORE_MM / CURRENT_NMD_MM
NMD_WEIGHTED_MATURITY = float(
    np.average(nmd_ladder.behavioural_maturity_y, weights=nmd_ladder.notional_mm)
)

display(nmd_summary)
print(f"Total NMD              : {CURRENT_NMD_MM:,.2f} mm")
print(f"Core NMD ratio         : {NMD_CORE_RATIO:.1%}")
print(f"Weighted EVE maturity  : {NMD_WEIGHTED_MATURITY:.2f} years")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for segment, history in nmd_histories.items():
    axes[0].plot(history.index, history, label=segment)
axes[0].set_title("Illustrative ten-year NMD balance histories")
axes[0].set_ylabel("Balance (mm)")
axes[0].legend(fontsize=8)

ladder_plot = nmd_ladder.groupby(["segment", "behavioural_maturity_y"], as_index=False).notional_mm.sum()
for segment, group in ladder_plot.groupby("segment"):
    axes[1].plot(group.behavioural_maturity_y, group.notional_mm, marker="o", label=segment)
axes[1].set_title("Governed NMD behavioural repricing ladder")
axes[1].set_xlabel("Behavioural maturity (years)")
axes[1].set_ylabel("Notional (mm)")
axes[1].legend(fontsize=8)
plt.tight_layout()
NMD_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_05_nmd_profile.png"
plt.savefig(NMD_FIGURE, bbox_inches="tight")
plt.show()
'''

BOOK_CODE = r'''
def row(
    instrument_id, side, product, notional_mm, coupon_rate, legal_maturity_y,
    next_repricing_y, rate_type, coupon_frequency=2, amortization="bullet",
    prepayment_rate=0.0, early_redemption_rate=0.0, reference_index="fixed",
    nii_beta_up=1.0, nii_beta_down=1.0, repricing_lag_months=0,
    instrument_type="cash", fixed_leg_sign=0.0,
):
    return {
        "instrument_id": instrument_id,
        "side": side,
        "product": product,
        "currency": CFG.currency,
        "instrument_type": instrument_type,
        "notional_mm": float(notional_mm),
        "coupon_rate": float(coupon_rate),
        "legal_maturity_y": float(legal_maturity_y),
        "next_repricing_y": float(next_repricing_y),
        "rate_type": rate_type,
        "coupon_frequency": int(coupon_frequency),
        "amortization": amortization,
        "prepayment_rate": float(prepayment_rate),
        "early_redemption_rate": float(early_redemption_rate),
        "reference_index": reference_index,
        "nii_beta_up": float(nii_beta_up),
        "nii_beta_down": float(nii_beta_down),
        "repricing_lag_months": int(repricing_lag_months),
        "fixed_leg_sign": float(fixed_leg_sign),
    }


def build_banking_book(core_scale=1.0):
    assets = [
        row("A01", "asset", "Cash and reserves",       100, 0.040, 0.01, 0.01, "floating", 12, reference_index="OIS"),
        row("A02", "asset", "HQLA securities fixed", 250, 0.046, 2.00, 2.00, "fixed", 2),
        row("A03", "asset", "Corporate floating",    300, 0.055, 4.00, 0.25, "floating", 4, reference_index="SOFR"),
        row("A04", "asset", "SME floating",          250, 0.060, 5.00, 0.50, "floating", 2, reference_index="SOFR"),
        row("A05", "asset", "Mortgage fixed",        450, 0.052, 12.0, 12.0, "fixed", 12, "linear", prepayment_rate=0.04),
        row("A06", "asset", "Consumer fixed",        180, 0.061, 4.00, 4.00, "fixed", 12, "linear", prepayment_rate=0.07),
        row("A07", "asset", "Corporate fixed",       220, 0.057, 7.00, 7.00, "fixed", 2),
        row("A08", "asset", "Mortgage floating",     250, 0.054, 8.00, 0.50, "floating", 2, reference_index="SOFR"),
    ]
    liabilities = [
        row("L01", "liability", "Wholesale overnight",     200, 0.038, 0.08, 0.08, "floating", 12, reference_index="SOFR"),
        row("L02", "liability", "Wholesale floating",      250, 0.041, 2.00, 0.25, "floating", 4, reference_index="SOFR"),
        row("L03", "liability", "Term funding fixed",      300, 0.044, 3.00, 3.00, "fixed", 2),
        row("L04", "liability", "Retail term deposits",    200, 0.032, 2.00, 2.00, "fixed", 12, early_redemption_rate=0.03),
        row("L05", "liability", "Secured funding floating",200, 0.039, 1.00, 0.25, "floating", 4, reference_index="SOFR"),
    ]

    # Total NMD remains fixed; core_scale shifts balances between core and non-core.
    nmd_rows = []
    for segment, group in nmd_ladder.groupby("segment"):
        total = float(group.notional_mm.sum())
        core_base = float(group.loc[group.core_flag, "notional_mm"].sum())
        cfg_row = NMD_SEGMENTS.loc[NMD_SEGMENTS.segment == segment].iloc[0]
        core_cap = float(cfg_row.core_cap * total)
        core = min(core_base * core_scale, core_cap, total)
        noncore = total - core
        nmd_rows.append(row(
            f"NMD-{segment[:3]}-NC", "liability", f"NMD {segment} non-core",
            noncore, 0.018, 1/365, 0.0, "behavioural", 12,
            reference_index="administered", nii_beta_up=cfg_row.beta_up,
            nii_beta_down=cfg_row.beta_down, repricing_lag_months=cfg_row.repricing_lag_months,
        ))
        core_group = group[group.core_flag].copy()
        weights = core_group.notional_mm / core_group.notional_mm.sum()
        for j, ((_, ladder_row), weight) in enumerate(zip(core_group.iterrows(), weights), start=1):
            tenor = float(ladder_row.behavioural_maturity_y)
            nmd_rows.append(row(
                f"NMD-{segment[:3]}-{j:02d}", "liability", f"NMD {segment} core",
                core * float(weight), 0.018, tenor, tenor, "behavioural", 12,
                reference_index="administered", nii_beta_up=cfg_row.beta_up,
                nii_beta_down=cfg_row.beta_down, repricing_lag_months=cfg_row.repricing_lag_months,
            ))

    # Pay-fixed / receive-floating swap: an off-balance-sheet hedge.
    derivative = row(
        "D01", "derivative", "Pay-fixed receive-floating IRS", 150, 0.045,
        5.0, 0.25, "swap", 2, reference_index="SOFR", instrument_type="swap",
        fixed_leg_sign=-1.0,
    )
    return pd.DataFrame(assets + liabilities + nmd_rows + [derivative])


banking_book = build_banking_book()
on_balance = banking_book[banking_book.side.isin(["asset", "liability"])]
asset_total = float(on_balance.loc[on_balance.side == "asset", "notional_mm"].sum())
liability_total = float(on_balance.loc[on_balance.side == "liability", "notional_mm"].sum())
accounting_equity = asset_total - liability_total

reconciliation = pd.Series({
    "Assets": asset_total,
    "Liabilities": liability_total,
    "Accounting equity": accounting_equity,
    "Tier 1 capital (regulatory input)": CFG.tier1_capital_mm,
    "Off-balance-sheet IRS notional": float(banking_book.loc[banking_book.instrument_type == "swap", "notional_mm"].sum()),
}, name="mm")
display(reconciliation.to_frame())
assert np.isclose(accounting_equity, CFG.accounting_equity_mm)
assert not np.isclose(CFG.tier1_capital_mm, accounting_equity), "Tier 1 should be distinct from accounting equity in this case study"
assert np.isclose(
    on_balance.loc[on_balance["product"].str.startswith("NMD"), "notional_mm"].sum(),
    CURRENT_NMD_MM,
)

BASEL_BUCKETS = [
    (0.0028, "ON"), (1/12, "ON-1M"), (0.25, "1M-3M"), (0.50, "3M-6M"),
    (0.75, "6M-9M"), (1.00, "9M-1Y"), (1.50, "1Y-1.5Y"), (2.00, "1.5Y-2Y"),
    (3.00, "2Y-3Y"), (4.00, "3Y-4Y"), (5.00, "4Y-5Y"), (6.00, "5Y-6Y"),
    (7.00, "6Y-7Y"), (8.00, "7Y-8Y"), (9.00, "8Y-9Y"), (10.0, "9Y-10Y"),
    (15.0, "10Y-15Y"), (20.0, "15Y-20Y"), (np.inf, ">20Y"),
]

def bucket_name(t):
    t = float(t)
    for upper_bound, name in BASEL_BUCKETS:
        if t <= upper_bound + 1e-12:
            return name
    raise RuntimeError("Unreachable bucket mapping")

repricing_rows = on_balance.copy()
repricing_rows["bucket"] = repricing_rows.next_repricing_y.map(bucket_name)
repricing_rows["signed_notional_mm"] = np.where(
    repricing_rows.side == "asset", repricing_rows.notional_mm, -repricing_rows.notional_mm
)
# Add the two repricing legs of the pay-fixed/receive-floating IRS to the gap view.
swap = banking_book.loc[banking_book["instrument_type"] == "swap"].iloc[0]
derivative_gap = pd.DataFrame([
    {"bucket": bucket_name(swap.next_repricing_y), "signed_notional_mm": swap.notional_mm},
    {"bucket": bucket_name(swap.legal_maturity_y), "signed_notional_mm": -swap.notional_mm},
])
gap_source = pd.concat([repricing_rows[["bucket", "signed_notional_mm"]], derivative_gap], ignore_index=True)
gap_table = (
    gap_source.groupby("bucket", as_index=False).signed_notional_mm.sum()
    .set_index("bucket").reindex([x[1] for x in BASEL_BUCKETS], fill_value=0.0)
)
gap_table["cumulative_gap_mm"] = gap_table.signed_notional_mm.cumsum()
display(gap_table.T)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(gap_table.index, gap_table.signed_notional_mm)
ax.axhline(0.0, linewidth=0.8)
ax.set_title("Basel-style repricing gap by 19 time buckets")
ax.set_ylabel("Assets minus liabilities (mm)")
ax.tick_params(axis="x", rotation=70)
plt.tight_layout()
GAP_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_06_repricing_gap.png"
plt.savefig(GAP_FIGURE, bbox_inches="tight")
plt.show()
'''

VALUATION_CODE = r'''
def prepayment_scalar(scenario):
    return {
        "base": 1.00,
        "parallel_up": 0.70,
        "parallel_down": 1.60,
        "short_up": 0.80,
        "short_down": 1.35,
        "steepener": 1.15,
        "flattener": 0.90,
    }[scenario]


def early_redemption_scalar(scenario):
    return {
        "base": 1.00,
        "parallel_up": 1.50,
        "parallel_down": 0.75,
        "short_up": 1.35,
        "short_down": 0.85,
        "steepener": 1.10,
        "flattener": 1.20,
    }[scenario]


def contractual_cashflows(row, scenario="base"):
    horizon = float(row.legal_maturity_y)
    if row.rate_type in {"floating", "behavioural"}:
        horizon = max(float(row.next_repricing_y), 1.0 / 365.0)
    frequency = max(int(row.coupon_frequency), 1)
    step = 1.0 / frequency
    times = np.arange(step, horizon + 1e-12, step)
    if len(times) == 0 or not np.isclose(times[-1], horizon):
        times = np.append(times, horizon)
    accruals = np.diff(np.insert(times, 0, 0.0))
    notional = float(row.notional_mm)
    coupon = float(row.coupon_rate)

    if row.amortization == "linear" and row.rate_type == "fixed":
        scheduled_principal = notional / len(times)
        outstanding = notional
        cfs = []
        cpr = min(float(row.prepayment_rate) * prepayment_scalar(scenario), 0.60)
        for accrual in accruals:
            interest = outstanding * coupon * accrual
            prepay = min(outstanding - scheduled_principal, outstanding * cpr * accrual)
            principal = min(outstanding, scheduled_principal + max(prepay, 0.0))
            cfs.append(interest + principal)
            outstanding -= principal
        if outstanding > 1e-10:
            cfs[-1] += outstanding
        return times, np.asarray(cfs)

    # Fixed-term deposits: the early-redeemed portion stops accruing after withdrawal.
    if row.early_redemption_rate > 0 and row.side == "liability":
        er = min(float(row.early_redemption_rate) * early_redemption_scalar(scenario), 0.50)
        early_t = min(0.5, horizon)
        regular_notional = notional * (1.0 - er)
        early_notional = notional * er
        cashflow_map = {}
        for t, accrual in zip(times, accruals):
            cashflow_map[t] = cashflow_map.get(t, 0.0) + regular_notional * coupon * accrual
        cashflow_map[times[-1]] += regular_notional
        early_interest = early_notional * coupon * early_t
        cashflow_map[early_t] = cashflow_map.get(early_t, 0.0) + early_notional + early_interest
        out_times = np.array(sorted(cashflow_map), dtype=float)
        out_cfs = np.array([cashflow_map[t] for t in out_times], dtype=float)
        return out_times, out_cfs

    cfs = notional * coupon * accruals
    cfs[-1] += notional
    return times, cfs


def discount_factor(times, scenario="base"):
    times = np.asarray(times, dtype=float)
    rates_pct = fitted_zero_rate_pct(times)
    if scenario != "base":
        rates_pct = rates_pct + irrbb_shift_bp(times, scenario) / 100.0
    if CFG.post_shock_floor_pct is not None:
        rates_pct = np.maximum(rates_pct, CFG.post_shock_floor_pct)
    return np.exp(-(rates_pct / 100.0) * times)


def pv_swap(row, scenario="base"):
    T = float(row.legal_maturity_y)
    freq = int(row.coupon_frequency)
    times = np.arange(1.0 / freq, T + 1e-12, 1.0 / freq)
    dfs = discount_factor(times, scenario)
    fixed_annuity = np.sum((1.0 / freq) * dfs)
    pv_fixed = float(row.notional_mm) * float(row.coupon_rate) * fixed_annuity
    pv_float = float(row.notional_mm) * (1.0 - float(discount_factor(np.array([T]), scenario)[0]))
    # fixed_leg_sign=-1 means pay fixed / receive floating.
    return float(row.fixed_leg_sign) * pv_fixed - float(row.fixed_leg_sign) * pv_float


def pv_instrument(row, scenario="base"):
    if row.instrument_type == "swap":
        return pv_swap(row, scenario)
    times, cfs = contractual_cashflows(row, scenario)
    pv = float(np.sum(cfs * discount_factor(times, scenario)))
    return pv if row.side == "asset" else -pv


def value_book(book, scenario="base"):
    details = book.copy()
    details["signed_pv_mm"] = [pv_instrument(r, scenario) for _, r in details.iterrows()]
    return float(details.signed_pv_mm.sum()), details


EVE_BASE_MM, base_details = value_book(banking_book, "base")
print(f"Base EVE (IMS full-cash-flow proxy): {EVE_BASE_MM:,.3f} mm")
product_pv = base_details.groupby(["side", "product"], as_index=False).signed_pv_mm.sum()
display(product_pv)
'''

SCENARIO_CODE = r'''
def irrbb_shift_bp(times, scenario, currency=CFG.currency):
    times = np.asarray(times, dtype=float)
    cfg = SHOCK_CONFIG_BP[currency]
    short = cfg["short"] * np.exp(-times / CFG.shock_decay_years)
    long = cfg["long"] * (1.0 - np.exp(-times / CFG.shock_decay_years))
    if scenario == "parallel_up":
        return np.full_like(times, cfg["parallel"])
    if scenario == "parallel_down":
        return np.full_like(times, -cfg["parallel"])
    if scenario == "short_up":
        return short
    if scenario == "short_down":
        return -short
    if scenario == "steepener":
        return -0.65 * short + 0.90 * long
    if scenario == "flattener":
        return 0.80 * short - 0.60 * long
    raise ValueError(f"Unknown scenario: {scenario}")

SCENARIOS = ["parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"]
scenario_rows = []
for scenario in SCENARIOS:
    shocked_eve, _ = value_book(banking_book, scenario)
    delta = shocked_eve - EVE_BASE_MM
    adverse = max(-delta, 0.0)
    scenario_rows.append({
        "scenario": scenario,
        "shocked_EVE_mm": shocked_eve,
        "delta_EVE_mm": delta,
        "adverse_loss_mm": adverse,
        "adverse_delta_EVE_to_Tier1_pct": 100.0 * adverse / CFG.tier1_capital_mm,
        "limit_utilisation_pct": 100.0 * adverse / (CFG.outlier_limit * CFG.tier1_capital_mm),
        "outlier_breach": adverse / CFG.tier1_capital_mm >= CFG.outlier_limit,
    })
irrbb_table = pd.DataFrame(scenario_rows).set_index("scenario").sort_values("delta_EVE_mm")
worst_scenario = str(irrbb_table.index[0])
worst_loss_mm = float(irrbb_table.iloc[0].adverse_loss_mm)
worst_ratio = worst_loss_mm / CFG.tier1_capital_mm

display(irrbb_table)
print(f"Worst scenario                    : {worst_scenario}")
print(f"Maximum adverse ΔEVE / Tier 1     : {worst_ratio:.2%}")
print(f"15% supervisory outlier threshold : {CFG.outlier_limit:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_table = irrbb_table.sort_values("delta_EVE_mm")
axes[0].barh(plot_table.index, plot_table.delta_EVE_mm)
axes[0].axvline(0.0, linewidth=0.8)
axes[0].set_title("ΔEVE under six prescribed scenarios")
axes[0].set_xlabel("ΔEVE (mm)")
axes[1].barh(plot_table.index, plot_table.adverse_delta_EVE_to_Tier1_pct)
axes[1].axvline(CFG.outlier_limit * 100.0, linestyle="--", linewidth=1.0, label="15% Tier 1")
axes[1].set_title("Adverse ΔEVE / Tier 1")
axes[1].set_xlabel("Percent")
axes[1].legend(fontsize=8)
plt.tight_layout()
EVE_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_07_eve_scenarios.png"
plt.savefig(EVE_FIGURE, bbox_inches="tight")
plt.show()

# Key behavioural-assumption sensitivity: change core NMD allocation while total NMD is fixed.
sensitivity_rows = []
for scale in [0.80, 1.00, 1.20]:
    b = build_banking_book(core_scale=scale)
    base, _ = value_book(b, "base")
    losses = {}
    for scenario in SCENARIOS:
        shocked, _ = value_book(b, scenario)
        losses[scenario] = max(base - shocked, 0.0)
    worst = max(losses, key=losses.get)
    sensitivity_rows.append({
        "core_scale": scale,
        "worst_scenario": worst,
        "worst_loss_mm": losses[worst],
        "worst_loss_to_Tier1_pct": 100.0 * losses[worst] / CFG.tier1_capital_mm,
    })
nmd_sensitivity = pd.DataFrame(sensitivity_rows)
display(nmd_sensitivity)
'''

NII_CODE = r'''
def monthly_nii(book, parallel_shock_bp=0.0):
    """Constant-balance-sheet 12-month NII path with discrete repricing."""
    shock = parallel_shock_bp / 10_000.0
    months = np.arange(1, CFG.nii_horizon_months + 1)
    records = []
    for _, r in book.iterrows():
        if r.instrument_type == "swap":
            fixed_monthly = float(r.notional_mm) * float(r.coupon_rate) / 12.0
            reset_month = max(1, int(np.ceil(float(r.next_repricing_y) * 12)))
            float_base = float(fitted_zero_rate_pct(np.array([max(r.next_repricing_y, 0.25)]))[0]) / 100.0
            for m in months:
                float_rate = float_base + (shock if m >= reset_month else 0.0)
                # pay fixed / receive floating when fixed_leg_sign=-1
                income = float(r.notional_mm) * float_rate / 12.0 - fixed_monthly
                records.append({"month": m, "instrument_id": r.instrument_id, "product": r["product"], "nii_mm": income})
            continue

        base_rate = float(r.coupon_rate)
        reset_month = int(np.ceil(float(r.next_repricing_y) * 12))
        reset_month += int(r.repricing_lag_months)
        beta = float(r.nii_beta_up if parallel_shock_bp >= 0 else r.nii_beta_down)
        sign = 1.0 if r.side == "asset" else -1.0
        for m in months:
            repriced = (r.rate_type in {"floating", "behavioural"} and m >= max(reset_month, 1))
            if r.rate_type == "fixed" and reset_month <= CFG.nii_horizon_months:
                repriced = m >= max(reset_month, 1)
            rate = base_rate + (beta * shock if repriced else 0.0)
            rate = max(rate, 0.0)
            nii = sign * float(r.notional_mm) * rate / 12.0
            records.append({"month": m, "instrument_id": r.instrument_id, "product": r["product"], "nii_mm": nii})
    return pd.DataFrame(records)

base_nii_path = monthly_nii(banking_book, 0.0)
up_nii_path = monthly_nii(banking_book, SHOCK_CONFIG_BP[CFG.currency]["parallel"])
down_nii_path = monthly_nii(banking_book, -SHOCK_CONFIG_BP[CFG.currency]["parallel"])

base_nii = float(base_nii_path.nii_mm.sum())
nii_up = float(up_nii_path.nii_mm.sum() - base_nii)
nii_down = float(down_nii_path.nii_mm.sum() - base_nii)
print(f"Base 12-month NII : {base_nii:,.3f} mm")
print(f"ΔNII parallel up  : {nii_up:+,.3f} mm")
print(f"ΔNII parallel down: {nii_down:+,.3f} mm")

up_attr = up_nii_path.groupby("product").nii_mm.sum() - base_nii_path.groupby("product").nii_mm.sum()
down_attr = down_nii_path.groupby("product").nii_mm.sum() - base_nii_path.groupby("product").nii_mm.sum()
nii_attribution = pd.concat([up_attr.rename("parallel_up"), down_attr.rename("parallel_down")], axis=1).fillna(0.0)
display(nii_attribution.sort_values("parallel_up"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
monthly = pd.DataFrame({
    "base": base_nii_path.groupby("month").nii_mm.sum(),
    "parallel_up": up_nii_path.groupby("month").nii_mm.sum(),
    "parallel_down": down_nii_path.groupby("month").nii_mm.sum(),
})
monthly.plot(ax=axes[0], marker="o")
axes[0].set_title("Monthly NII under constant-balance-sheet scenarios")
axes[0].set_ylabel("NII (mm)")
nii_attribution.parallel_up.sort_values().tail(8).plot(kind="barh", ax=axes[1])
axes[1].set_title("Largest positive-shock ΔNII contributions")
axes[1].set_xlabel("ΔNII (mm)")
plt.tight_layout()
NII_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_08_nii.png"
plt.savefig(NII_FIGURE, bbox_inches="tight")
plt.show()
'''

EXPORT_CODE = r'''
# Integration, unit and governance checks.
assert zero_yields.index.is_monotonic_increasing
assert zero_yields.notna().all().all()
assert np.isfinite(fitted_values).all()
assert design_condition < 1_000.0, "Loading matrix exceeds hard governance limit"
assert max_loading_corr < 0.995, "Loading correlation exceeds hard governance limit"
assert np.isclose(asset_total, liability_total + CFG.accounting_equity_mm)
assert np.isclose(CURRENT_NMD_MM, 650.0)
assert (nmd_summary.weighted_core_maturity_y <= nmd_summary.maturity_cap_y + 1e-12).all()
assert np.isfinite(irrbb_table.select_dtypes(include=[np.number]).values).all()
assert np.isclose(irrbb_shift_bp(np.array([1.0, 10.0]), "parallel_up"), SHOCK_CONFIG_BP[CFG.currency]["parallel"]).all()
assert worst_ratio >= 0.0
assert np.isfinite([base_nii, nii_up, nii_down]).all()

metrics = {
    "model_key": MODEL_KEY,
    "model_name": MODEL_NAME,
    "data_source": DATA_SOURCE,
    "reference_date": str(zero_yields.index.max().date()),
    "observations": int(n_obs),
    "train_observations": int(len(train_panel)),
    "validation_observations": int(len(validation_panel)),
    "test_observations": int(len(test_panel)),
    "lambda1_years": float(lambda1_hat),
    "lambda2_years": float(lambda2_hat) if lambda2_hat is not None else None,
    "design_condition_number": float(design_condition),
    "maximum_loading_correlation": float(max_loading_corr),
    "vif": {k: float(v) for k, v in vif.items()},
    "train_mean_rmse_bp": float(train_rmse.mean()),
    "validation_mean_rmse_bp": float(validation_rmse.mean()),
    "test_mean_rmse_bp": float(test_rmse.mean()),
    "test_p95_rmse_bp": float(np.quantile(test_rmse, 0.95)),
    "test_max_point_error_bp": float(np.max(np.abs(test_errors_bp))),
    "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
    "level_factor_abs_correlation": float(abs(factor_corr.loc["PC1", "beta0"])),
    "matched_factor_abs_correlation_mean": float(np.mean(matched_factor_corr)),
    "factor_correlation_matrix": factor_corr.astype(float).to_dict(),
    "curve_shape_checks": curve_checks,
    "nmd_core_ratio": float(NMD_CORE_RATIO),
    "nmd_weighted_maturity_years": float(NMD_WEIGHTED_MATURITY),
    "accounting_equity_mm": float(accounting_equity),
    "tier1_capital_mm": float(CFG.tier1_capital_mm),
    "base_eve_mm": float(EVE_BASE_MM),
    "worst_eve_scenario": worst_scenario,
    "maximum_adverse_delta_eve_mm": float(worst_loss_mm),
    "maximum_adverse_delta_eve_to_tier1_pct": float(100.0 * worst_ratio),
    "outlier_breach": bool(worst_ratio >= CFG.outlier_limit),
    "base_nii_mm": float(base_nii),
    "delta_nii_parallel_up_mm": float(nii_up),
    "delta_nii_parallel_down_mm": float(nii_down),
}

model_artifact_dir = ARTIFACT_DIR / MODEL_KEY
model_artifact_dir.mkdir(exist_ok=True)
with open(model_artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
params.to_csv(model_artifact_dir / "curve_parameters.csv")
fit_diagnostics.to_csv(model_artifact_dir / "fit_diagnostics.csv")
factor_corr.to_csv(model_artifact_dir / "factor_correlations.csv")
nmd_summary.to_csv(model_artifact_dir / "nmd_summary.csv", index=False)
nmd_ladder.to_csv(model_artifact_dir / "nmd_ladder.csv", index=False)
gap_table.to_csv(model_artifact_dir / "repricing_gap.csv")
irrbb_table.to_csv(model_artifact_dir / "eve_scenarios.csv")
nii_attribution.to_csv(model_artifact_dir / "nii_attribution.csv")
nmd_sensitivity.to_csv(model_artifact_dir / "nmd_sensitivity.csv", index=False)

print("All validation checks passed.")
print(f"Artifacts written to: {model_artifact_dir}")
display(pd.Series(metrics).to_frame("value"))
'''


def model_specific(model: str):
    if model == 'nss4':
        title = 'Four-Factor Nelson–Siegel–Svensson IRRBB Engine'
        key = 'nss4'
        formula = r'''
## 3. Four-factor Svensson representation

The representation is

$$z(\tau)=\beta_0+\beta_1L_1(\tau;\lambda_1)+\beta_2L_2(\tau;\lambda_1)+\beta_3L_3(\tau;\lambda_2).$$

This notebook uses **four beta factors and two decay hyperparameters**. The decay parameters are calibrated only on the training sample; daily beta coefficients are then obtained by linear least squares on train, validation and test dates. A hard governance screen rejects loading matrices with condition number above 1,000 or absolute pairwise loading correlation above 0.995. This prevents a numerically singular fit from silently entering the risk engine, while still allowing the notebook to measure whether the fourth factor is economically identified.
'''
        model_code = r'''
MODEL_KEY = "nss4"
MODEL_NAME = "Four-factor Nelson–Siegel–Svensson"
BETA_COLUMNS = ["beta0", "beta1", "beta2", "beta3"]


def model_loadings(tau, lambda1, lambda2):
    tau = np.asarray(tau, dtype=float)
    x1 = np.maximum(tau, 1e-8) / lambda1
    x2 = np.maximum(tau, 1e-8) / lambda2
    slope = -np.expm1(-x1) / x1
    curvature1 = slope - np.exp(-x1)
    curvature2 = -np.expm1(-x2) / x2 - np.exp(-x2)
    return np.column_stack([np.ones_like(tau), slope, curvature1, curvature2])


def unpack_lambdas(x):
    lambda1 = np.exp(x[0])
    lambda2 = lambda1 + np.exp(x[1])
    return lambda1, lambda2


def calibrate_lambdas():
    train_values = train_panel.iloc[::3].values
    candidates = []
    starts = [(0.5, 1.0), (1.0, 3.0), (2.0, 5.0), (2.0, 12.0), (4.0, 12.0)]
    bounds = [(np.log(0.10), np.log(8.0)), (np.log(0.20), np.log(30.0))]

    def objective(x):
        l1, l2 = unpack_lambdas(x)
        design = model_loadings(ZERO_TENORS, l1, l2)
        _, fitted = fit_panel(train_values, design)
        mse = np.mean((fitted - train_values) ** 2)
        condition = np.linalg.cond(design)
        loading_corr = max_off_diagonal_abs_corr(design[:, 1:])
        # Soft penalties make the numerical risk visible in model selection rather
        # than applying a penalty that only activates after the failure has occurred.
        penalty = 2e-7 * max(np.log(condition / 500.0), 0.0) ** 2
        penalty += 2e-7 * max((loading_corr - 0.997) / 0.003, 0.0) ** 2
        return mse + penalty

    for l1_start, gap_start in starts:
        result = minimize(
            objective,
            x0=np.log([l1_start, gap_start]),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not (result.success and np.isfinite(result.fun)):
            continue
        l1, l2 = unpack_lambdas(result.x)
        design = model_loadings(ZERO_TENORS, l1, l2)
        condition = float(np.linalg.cond(design))
        loading_corr = max_off_diagonal_abs_corr(design[:, 1:])
        if condition >= 1_000 or loading_corr >= 0.995:
            continue
        _, fitted_val = fit_panel(validation_panel.values, design)
        validation_rmse = float(rmse_by_date_bp(validation_panel.values, fitted_val).mean())
        candidates.append((validation_rmse, result.fun, l1, l2, result))
    if not candidates:
        raise RuntimeError("No governed Svensson calibration candidate survived")
    return min(candidates, key=lambda x: (x[0], x[1]))


validation_score, _, lambda1_hat, lambda2_hat, optimizer_result = calibrate_lambdas()
design = model_loadings(ZERO_TENORS, lambda1_hat, lambda2_hat)
'''
        rate_code = 'return model_loadings(np.atleast_1d(tau), lambda1_hat, lambda2_hat) @ np.asarray(beta, dtype=float)'
    else:
        title = 'Three-Factor Nelson–Siegel IRRBB Engine'
        key = 'ns3'
        formula = r'''
## 3. Three-factor Nelson–Siegel representation

The representation is

$$z(\tau)=\beta_0+\beta_1L_1(\tau;\lambda_1)+\beta_2L_2(\tau;\lambda_1).$$

This notebook uses **three beta factors and one decay hyperparameter**. The decay parameter is calibrated only on the training sample and selected using validation error. Daily beta coefficients are fitted linearly with the decay fixed. This creates a parsimonious structural factor model whose level, slope and curvature interpretation can be benchmarked against out-of-sample PCA factors.
'''
        model_code = r'''
MODEL_KEY = "ns3"
MODEL_NAME = "Three-factor Nelson–Siegel"
BETA_COLUMNS = ["beta0", "beta1", "beta2"]


def model_loadings(tau, lambda1):
    tau = np.asarray(tau, dtype=float)
    x = np.maximum(tau, 1e-8) / lambda1
    slope = -np.expm1(-x) / x
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curvature])


def calibrate_lambda():
    train_values = train_panel.iloc[::3].values
    candidates = []
    bounds = [(np.log(0.10), np.log(10.0))]

    def objective(x):
        lambda1 = np.exp(x[0])
        design = model_loadings(ZERO_TENORS, lambda1)
        _, fitted = fit_panel(train_values, design)
        return np.mean((fitted - train_values) ** 2)

    for start in [0.5, 1.0, 1.5, 2.5, 4.0, 6.0]:
        result = minimize(objective, np.log([start]), method="L-BFGS-B", bounds=bounds)
        if not (result.success and np.isfinite(result.fun)):
            continue
        lambda1 = float(np.exp(result.x[0]))
        design = model_loadings(ZERO_TENORS, lambda1)
        _, fitted_val = fit_panel(validation_panel.values, design)
        validation_rmse = float(rmse_by_date_bp(validation_panel.values, fitted_val).mean())
        candidates.append((validation_rmse, result.fun, lambda1, result))
    if not candidates:
        raise RuntimeError("Nelson–Siegel calibration failed")
    return min(candidates, key=lambda x: (x[0], x[1]))


validation_score, _, lambda1_hat, optimizer_result = calibrate_lambda()
lambda2_hat = None
design = model_loadings(ZERO_TENORS, lambda1_hat)
'''
        rate_code = 'return model_loadings(np.atleast_1d(tau), lambda1_hat) @ np.asarray(beta, dtype=float)'
    post_model = f'''
def zero_rate_pct(beta, tau):
    {rate_code}

all_betas, fitted_values = fit_panel(zero_yields.values, design)
params = pd.DataFrame(all_betas, index=zero_yields.index, columns=BETA_COLUMNS)
params["lambda1"] = lambda1_hat
if lambda2_hat is not None:
    params["lambda2"] = lambda2_hat

current_beta = params.iloc[-1][BETA_COLUMNS].values
def fitted_zero_rate_pct(tau):
    return zero_rate_pct(current_beta, tau)

errors_bp = (fitted_values - zero_yields.values) * 100.0
train_rmse = rmse_by_date_bp(zero_yields.iloc[:train_end].values, fitted_values[:train_end])
validation_rmse = rmse_by_date_bp(zero_yields.iloc[train_end:validation_end].values, fitted_values[train_end:validation_end])
test_rmse = rmse_by_date_bp(zero_yields.iloc[validation_end:].values, fitted_values[validation_end:])
test_errors_bp = errors_bp[validation_end:]
design_condition = float(np.linalg.cond(design))
max_loading_corr = max_off_diagonal_abs_corr(design[:, 1:])
vif = variance_inflation_factors(design)

fit_diagnostics = pd.DataFrame({{
    "date": zero_yields.index,
    "sample": np.where(np.arange(n_obs) < train_end, "train", np.where(np.arange(n_obs) < validation_end, "validation", "test")),
    "rmse_bp": rmse_by_date_bp(zero_yields.values, fitted_values),
    "max_abs_error_bp": np.max(np.abs(errors_bp), axis=1),
}}).set_index("date")

print(f"Model                     : {{MODEL_NAME}}")
print(f"Global lambda1            : {{lambda1_hat:.4f}} years")
if lambda2_hat is not None:
    print(f"Global lambda2            : {{lambda2_hat:.4f}} years")
print(f"Design condition number   : {{design_condition:,.1f}}")
print(f"Max loading correlation   : {{max_loading_corr:.4f}}")
print(f"Train mean RMSE           : {{train_rmse.mean():.3f}} bp")
print(f"Validation mean RMSE      : {{validation_rmse.mean():.3f}} bp")
print(f"Test mean RMSE            : {{test_rmse.mean():.3f}} bp")
print(f"Test 95th percentile RMSE : {{np.quantile(test_rmse, 0.95):.3f}} bp")
display(params.tail())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for beta in BETA_COLUMNS:
    axes[0].plot(params.index, params[beta], label=beta)
axes[0].axvline(zero_yields.index[train_end], linestyle="--", linewidth=0.8, label="validation start")
axes[0].axvline(zero_yields.index[validation_end], linestyle=":", linewidth=0.8, label="test start")
axes[0].set_title("Parameter evolution with decay fixed from training")
axes[0].legend(fontsize=8)

fine_tau = np.linspace(0.05, 30.0, 500)
for pos in [-1, -63, -252]:
    if abs(pos) <= len(zero_yields):
        date = zero_yields.index[pos]
        beta = params.loc[date, BETA_COLUMNS].values
        axes[1].plot(fine_tau, zero_rate_pct(beta, fine_tau), label=f"fit {{date.date()}}")
        axes[1].scatter(ZERO_TENORS, zero_yields.loc[date], s=18)
axes[1].set_title("Parametric fit versus zero-coupon observations")
axes[1].set_xlabel("Maturity (years)")
axes[1].set_ylabel("Continuously compounded zero rate (%)")
axes[1].legend(fontsize=8)
plt.tight_layout()
CURVE_FIGURE = FIGURE_DIR / f"{{MODEL_KEY}}_01_curve_fit.png"
plt.savefig(CURVE_FIGURE, bbox_inches="tight")
plt.show()
'''
    return title, key, formula, model_code, post_model


def make_model_notebook(model: str):
    title, key, formula, model_code, post_model = model_specific(model)
    intro = f'''
# {title}

## Purpose

This notebook is an auditable **IRRBB internal-measurement-system prototype**. It is designed to demonstrate model architecture, risk measurement and model governance; it is not represented as a production regulatory calculation.

The notebook connects:

1. Official Federal Reserve `SVENY` continuously compounded zero-coupon yields when network access is available, with an explicitly labelled offline demonstration panel for reproducibility.
2. A train/validation/test curve representation with no look-ahead in decay calibration.
3. Numerical identification, PCA factor-alignment, out-of-sample and curve-shape diagnostics.
4. Basel-segmented NMDs with core caps and average-maturity controls.
5. A reconciled banking book, distinct accounting equity and Tier 1 capital, and an off-balance-sheet interest-rate swap.
6. Basel-prescribed six-scenario ΔEVE, a 15% Tier 1 outlier diagnostic and 12-month constant-balance-sheet ΔNII.
7. Behavioural-assumption sensitivity, validation assertions and versioned machine-readable artifacts.

### Important boundary

The Fed series are already smoothed zero-coupon estimates. This notebook performs **parametric representation and model-risk analysis**, not primary-market bond bootstrapping. In a bank, an approved curve service, behavioural model inventory and independent validation process would sit upstream of this engine.

### Primary references

- [Basel Framework SRP31 — IRRBB](https://www.bis.org/basel_framework/chapter/SRP/31.htm?inforce=20260101&published=20240716)
- [BCBS d578 — recalibrated shocks effective 1 January 2026](https://www.bis.org/bcbs/publ/d578.htm)
- [Basel Framework DIS70 — IRRBB disclosure](https://www.bis.org/basel_framework/chapter/DIS/70.htm)
- [Federal Reserve Gürkaynak–Sack–Wright nominal yield curve](https://www.federalreserve.gov/data/nominal-yield-curve.htm)
'''
    cells = [
        md(intro),
        md('''## 1. Configuration, controls and data contracts\n\nUnits are explicit: curve observations are percentage points, shocks are basis points, and monetary amounts are millions. The delivered outputs were produced in an offline execution environment; the source code defaults to `AUTO` and pulls the official Fed series when available.'''),
        code(COMMON_IMPORTS),
        md('''## 2. Market-data acquisition and temporal split\n\nThe loader treats data provenance as a first-class output. Decay parameters are calibrated on the training sample, selected using validation error and then frozen before the untouched test sample is evaluated.'''),
        code(DATA_CODE),
        md(formula),
        code(MODEL_DIAGNOSTIC_CODE),
        code(model_code),
        code(post_model),
        md('''## 4. Identification, factor alignment and out-of-sample diagnostics\n\nLow curve RMSE is not sufficient evidence of a sound factor model. The diagnostics below measure loading-matrix conditioning, pairwise loading correlation, variance inflation, out-of-sample error and correspondence with PCA factors estimated only from training-period changes. PCA signs are arbitrary, so the decision statistics use absolute correlations.'''),
        code(r'''
pca, factor_corr, matched_factor_corr = build_factor_alignment(params, BETA_COLUMNS)
curve_checks = curve_shape_checks(params, BETA_COLUMNS, zero_rate_pct)

print("Training PCA explained variance:", np.round(pca.explained_variance_ratio_, 4))
print(f"PC1–Δbeta0 absolute correlation: {abs(factor_corr.loc['PC1', 'beta0']):.3f}")
print(f"Mean matched factor correlation : {np.mean(matched_factor_corr):.3f}")
display(pd.DataFrame({"VIF": vif}))
display(factor_corr)
display(pd.Series(curve_checks).to_frame("value"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for i, pc in enumerate(["PC1", "PC2", "PC3"]):
    axes[0].plot(ZERO_LABELS, pca.components_[i], marker="o", label=pc)
axes[0].axhline(0.0, linewidth=0.8)
axes[0].set_title("Training-sample PCA loadings")
axes[0].legend()
image = axes[1].imshow(factor_corr.astype(float).values, aspect="auto", vmin=-1, vmax=1)
axes[1].set_xticks(range(len(BETA_COLUMNS)), BETA_COLUMNS)
axes[1].set_yticks(range(3), ["PC1", "PC2", "PC3"])
axes[1].set_title("Test-sample PC versus Δbeta correlations")
for i in range(3):
    for j in range(len(BETA_COLUMNS)):
        axes[1].text(j, i, f"{factor_corr.iloc[i, j]:.2f}", ha="center", va="center")
fig.colorbar(image, ax=axes[1], shrink=0.8)
plt.tight_layout()
FACTOR_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_02_factor_alignment.png"
plt.savefig(FACTOR_FIGURE, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(fit_diagnostics.index, fit_diagnostics.rmse_bp)
axes[0].axvline(zero_yields.index[train_end], linestyle="--", linewidth=0.8)
axes[0].axvline(zero_yields.index[validation_end], linestyle=":", linewidth=0.8)
axes[0].set_title("Daily fit RMSE with untouched test period")
axes[0].set_ylabel("RMSE (bp)")
axes[1].bar(range(design.shape[1]), [vif[f"loading_{j}"] for j in range(design.shape[1])])
axes[1].set_xticks(range(design.shape[1]), BETA_COLUMNS)
axes[1].set_title("Variance inflation by loading")
axes[1].set_ylabel("VIF")
plt.tight_layout()
IDENT_FIGURE = FIGURE_DIR / f"{MODEL_KEY}_03_identification.png"
plt.savefig(IDENT_FIGURE, bbox_inches="tight")
plt.show()
'''),
        md('''## 5. Behavioural NMD segmentation and replicating portfolio\n\nBasel requires NMD segmentation and imposes caps on core proportions and average repricing maturities. The illustrative calibration separates volume stability from rate stickiness and applies the Table 4 limits: 90%/5 years for retail transactional, 70%/4.5 years for retail non-transactional and 50%/4 years for wholesale. This remains a transparent demonstration because no account-level deposit data are supplied.'''),
        code(NMD_CODE),
        md('''## 6. Banking-book data model and repricing architecture\n\nThe book distinguishes contractual maturity, next repricing, behavioural maturity, product optionality, reference index and rate pass-through. Accounting equity reconciles the balance sheet, while Tier 1 capital remains a separate regulatory input. The 19-bucket view is a repricing-gap diagnostic; it is not substituted for full valuation.'''),
        code(BOOK_CODE),
        md('''## 7. Cash-flow engine and EVE valuation boundary\n\nThis is an **IMS-style full-cash-flow proxy**, not a claim that the standardised framework has been replicated in every detail. Fixed-rate amortising assets include transparent scenario-dependent prepayment scalars; term deposits include early-redemption scalars; floating instruments are valued to next reset; the swap is valued as fixed-leg annuity versus floating leg. Commercial margins remain in contractual coupons and the risk-free fitted curve is used for discounting. A production methodology would require explicit governance over commercial-margin inclusion.'''),
        code(VALUATION_CODE),
        md('''## 8. Prescribed ΔEVE scenarios, capital test and assumption sensitivity\n\nThe six Basel scenarios are applied per currency. The supervisory diagnostic is the maximum adverse ΔEVE divided by Tier 1 capital, not ΔEVE divided by base EVE. Because behavioural assumptions can dominate IRRBB, the engine also reruns the scenario set after scaling the core NMD allocation while total NMD remains fixed.'''),
        code(SCENARIO_CODE),
        md('''## 9. Twelve-month constant-balance-sheet ΔNII\n\nThe earnings simulation is monthly rather than a single exposure-period approximation. Positions remain at constant balances; rates change only after their contractual or behavioural repricing point; product-specific asymmetric deposit betas and lags are applied; the swap contributes fixed-versus-floating carry. The two prescribed parallel NII shocks are compared with the unshocked 12-month path.'''),
        code(NII_CODE),
        md('''## 10. Validation, artifact contract and reproducibility\n\nAssertions cover units, curve conditioning, NMD caps, reconciliation, shock signs and aggregation. Each model writes the same result contract (`metrics.json`, scenario tables, diagnostics and figures), allowing the markdown-only case-study notebook to compare the two engines without hidden manual calculations.'''),
        code(EXPORT_CODE),
        md('''## 11. Scope limitations and production migration\n\n**Implemented well enough for a portfolio case:** data provenance, temporal validation, model identification, behavioural segmentation, EVE/NII integration, prescribed shocks, capital normalization, key-assumption sensitivity, artifact versioning and reproducible controls.\n\n**Still illustrative:** the balance sheet and deposit histories are synthetic; no account-level NMD, prepayment or early-redemption estimation is performed; basis risk and CSRBB are not quantified; multicurrency aggregation and FX translation are absent; automatic options are not priced with dedicated option models; legal-entity consolidation, hedge accounting, FTP, market-data lineage, access control and independent model validation remain outside the notebook.\n\nIn production, the approved discount/projection curve service should be separate from the structural factor model. A curve representation that is useful for attribution is not automatically the correct pricing curve, and a low-RMSE pricing representation is not automatically an identified risk-factor model.'''),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3'},
        'project': {'model_key': key, 'artifact_contract': 'v1.0'},
    })
    return nb

# Build and execute the two model notebooks.
notebook_paths = []
for model, filename in [
    ('nss4', '01_four_factor_svensson_irrbb_engine.ipynb'),
    ('ns3', '02_three_factor_nelson_siegel_irrbb_engine.ipynb'),
]:
    path = ROOT / filename
    nb = make_model_notebook(model)
    nbf.write(nb, path)
    os.environ['IRRBB_DATA_MODE'] = 'OFFLINE'
    client = NotebookClient(nb, timeout=240, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    executed = client.execute()
    nbf.write(executed, path)
    notebook_paths.append(path)
    print('executed', path)

# -----------------------------------------------------------------------------
# Read artifacts and create comparison figures + markdown-only report.
# -----------------------------------------------------------------------------
with open(ART / 'nss4' / 'metrics.json', encoding='utf-8') as f:
    m4 = json.load(f)
with open(ART / 'ns3' / 'metrics.json', encoding='utf-8') as f:
    m3 = json.load(f)

# Comparison figure from machine-readable outputs.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
labels = ['4-factor\nSvensson', '3-factor\nNS']
axes[0].bar(labels, [m4['test_mean_rmse_bp'], m3['test_mean_rmse_bp']])
axes[0].set_title('Out-of-sample curve fit')
axes[0].set_ylabel('Test mean RMSE (bp)')
axes[1].bar(labels, [m4['design_condition_number'], m3['design_condition_number']])
axes[1].set_title('Loading-matrix conditioning')
axes[1].set_ylabel('Condition number')
axes[2].bar(labels, [m4['level_factor_abs_correlation'], m3['level_factor_abs_correlation']])
axes[2].set_title('Structural level-factor alignment')
axes[2].set_ylabel('|corr(PC1, Δβ0)|')
axes[2].set_ylim(0, 1)
plt.tight_layout()
comparison_figure = FIG / 'comparison_01_model_tradeoff.png'
plt.savefig(comparison_figure, bbox_inches='tight')
plt.close(fig)

# EVE comparison.
eve4 = pd.read_csv(ART / 'nss4' / 'eve_scenarios.csv', index_col=0)
eve3 = pd.read_csv(ART / 'ns3' / 'eve_scenarios.csv', index_col=0)
scenarios = ['parallel_up', 'parallel_down', 'steepener', 'flattener', 'short_up', 'short_down']
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(scenarios)); width = 0.38
ax.bar(x-width/2, eve4.loc[scenarios, 'delta_EVE_mm'], width, label='4-factor Svensson')
ax.bar(x+width/2, eve3.loc[scenarios, 'delta_EVE_mm'], width, label='3-factor NS')
ax.axhline(0.0, linewidth=0.8)
ax.set_xticks(x, scenarios, rotation=35, ha='right')
ax.set_ylabel('ΔEVE (mm)')
ax.set_title('Downstream EVE impact of curve representation choice')
ax.legend()
plt.tight_layout()
eve_comparison = FIG / 'comparison_02_eve.png'
plt.savefig(eve_comparison, bbox_inches='tight')
plt.close(fig)

# NMD assumption sensitivity comparison.
nmd_sens = pd.read_csv(ART / 'ns3' / 'nmd_sensitivity.csv')
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(nmd_sens.core_scale, nmd_sens.worst_loss_to_Tier1_pct, marker='o')
ax.axhline(15.0, linestyle='--', linewidth=1.0, label='15% Tier 1 threshold')
ax.set_xticks(nmd_sens.core_scale, ['80% of base core', 'Base core', '120% of base core'])
ax.set_ylabel('Worst adverse ΔEVE / Tier 1 (%)')
ax.set_title('Behavioural NMD assumption can dominate curve-model choice')
ax.legend()
plt.tight_layout()
nmd_sensitivity_figure = FIG / 'comparison_03_nmd_sensitivity.png'
plt.savefig(nmd_sensitivity_figure, bbox_inches='tight')
plt.close(fig)

# Business architecture diagram.
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_axis_off()
boxes = {
    "gov": (0.03, 0.80, 0.19, 0.11, "Governance & risk appetite\nBoard / ALCO / limits / ICAAP"),
    "data": (0.28, 0.80, 0.19, 0.11, "Data & curve services\nLineage / approved curves"),
    "beh": (0.53, 0.80, 0.19, 0.11, "Behavioural models\nNMD / prepayment / redemption"),
    "val": (0.78, 0.80, 0.19, 0.11, "Independent validation\nConcept / monitoring / outcomes"),
    "pos": (0.11, 0.53, 0.23, 0.12, "Position & cash-flow layer\nAssets / liabilities / OBS / currencies"),
    "eng": (0.39, 0.53, 0.22, 0.12, "IRRBB engine\nGap / basis / option risk\nEVE + NII"),
    "agg": (0.66, 0.53, 0.23, 0.12, "Aggregation & capital\nCurrency / Tier 1 / limits"),
    "tre": (0.20, 0.27, 0.23, 0.12, "Treasury actions\nHedge / funding / pricing"),
    "rep": (0.57, 0.27, 0.23, 0.12, "Management reporting\nALCO / exceptions / attribution"),
    "dis": (0.385, 0.05, 0.23, 0.11, "Disclosure & supervision\nIRRBB1 / changes / audit trail"),
}
for x0, y0, w, h, label in boxes.values():
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.5))
    ax.text(x0+w/2, y0+h/2, label, ha='center', va='center', fontsize=10)

def centre(key):
    x0,y0,w,h,_=boxes[key]; return x0+w/2,y0+h/2
def edge(key, side):
    x0,y0,w,h,_=boxes[key]
    return {"left":(x0,y0+h/2),"right":(x0+w,y0+h/2),"top":(x0+w/2,y0+h),"bottom":(x0+w/2,y0)}[side]
def arrow(a, aside, b, bside):
    ax.annotate('', xy=edge(b,bside), xytext=edge(a,aside), arrowprops=dict(arrowstyle='->', linewidth=1.2))

arrow('gov','right','data','left')
arrow('data','right','beh','left')
arrow('beh','right','val','left')
arrow('data','bottom','pos','top')
arrow('beh','bottom','eng','top')
arrow('val','bottom','agg','top')
arrow('pos','right','eng','left')
arrow('eng','right','agg','left')
arrow('eng','bottom','tre','top')
arrow('agg','bottom','rep','top')
arrow('tre','right','rep','left')
arrow('rep','bottom','dis','top')
ax.set_title('Target banking architecture for IRRBB management', fontsize=15, pad=18)
architecture_figure = FIG / 'business_architecture.png'
plt.savefig(architecture_figure, bbox_inches='tight')
plt.close(fig)

# Helper for markdown formatting.
def fnum(x, n=3):
    return f'{x:,.{n}f}'

def pct(x, n=2):
    return f'{x:.{n}f}%'

report_cells = []
report_cells.append(md(f'''
# IRRBB Model-Risk Case Study and Banking Architecture

## Executive decision

Two self-contained IRRBB prototypes were run through the same data contract, NMD calibration, banking book, cash-flow engine, Basel scenarios and NII simulation. The only intentional model change is the term-structure representation.

- **Four-factor Svensson:** superior out-of-sample reconstruction, but materially weaker factor identification.
- **Three-factor Nelson–Siegel:** slightly higher curve error, but much stronger numerical conditioning and structural level/slope/curvature alignment.
- **Recommended production architecture:** do not force one parametric model to serve every purpose. Use an approved market curve service for valuation, the three-factor model as the primary structural attribution model, and the four-factor model as a challenger/reconstruction benchmark subject to identification controls.
- **Business-critical finding:** reducing the assumed core NMD allocation to 80% of the base estimate pushes the illustrative adverse ΔEVE above the 15% Tier 1 threshold, while changing NS to Svensson barely moves the capital result. Behavioural-model governance dominates the curve-model choice in this book.

The delivered reference execution used the explicitly labelled offline panel because the build environment had no network. Both model notebooks default to `AUTO` and pull the official Federal Reserve `SVENY` data when available. No empirical market claim is made from the offline run.
'''))
report_cells.append(md('''
## 1. Why IRRBB is a business problem before it is a curve-fitting problem

IRRBB is the risk to a bank's **capital and earnings** from adverse changes in interest rates. Basel treats it under Pillar 2 and distinguishes three sources:

1. **Gap risk:** timing mismatch in repricing and maturity, including parallel and non-parallel term-structure changes.
2. **Basis risk:** imperfect co-movement of different reference rates used to price economically similar positions.
3. **Option risk:** explicit or behavioural options, including loan prepayment, term-deposit early redemption and NMD stability/repricing.

An IRRBB framework therefore cannot begin with `fit_curve()`. It begins with governance, risk appetite, complete position data, approved behavioural assumptions, scenario design, capital and earnings measures, limit monitoring and independent validation. The curve model is one controlled component inside that architecture.

![Target IRRBB architecture](attachment:business_architecture.png)
'''))
report_cells.append(md('''
## 2. Target bank operating model

### 2.1 Governance and risk appetite

The governing body approves the IRRBB framework and risk appetite. ALCO or delegated senior management sets operational limits, reviews breaches, approves material assumptions and decides hedging, funding and pricing actions. Risk appetite must cover both economic value and earnings because EVE and NII reveal different horizons and different failure modes.

### 2.2 First line, second line and validation

- **Treasury/ALM:** owns balance-sheet strategy, funding, hedging and pricing actions; produces or operates the measurement process under approved policy.
- **Independent risk control:** challenges exposures, assumptions, limits and scenario adequacy; monitors model use and exceptions.
- **Independent model validation/internal audit:** assesses conceptual soundness, implementation integrity, ongoing performance, benchmarking, change control and governance evidence.

### 2.3 Data and position architecture

The production data layer must reconcile general ledger, product systems and off-balance-sheet instruments; retain currency, legal entity, contractual maturity, repricing date, index, caps/floors, behavioural attributes and optionality; and preserve lineage from source to ALCO report. The notebook's DataFrame is a compact data contract, not a substitute for those controls.

### 2.4 Behavioural-model inventory

NMD, prepayment and early-redemption models are separate governed models. Basel requires NMD segmentation and caps on core proportions and average maturities. Their outputs modify cash-flow timing and therefore feed both EVE and NII. They must be versioned, independently validated, backtested and subjected to assumption sensitivity.

### 2.5 Measurement and action loop

Approved market curves and behavioural cash flows feed EVE and NII engines. Results are aggregated by currency and legal entity, normalized against Tier 1 and earnings limits, attributed to business units and risk factors, and converted into hedge/funding/pricing actions. Reporting must preserve the bridge from exposure to management decision.
'''))
report_cells.append(md('''
## 3. What the two model notebooks implement

Both notebooks implement the same controlled pipeline:

```text
configuration
    -> data provenance and temporal split
    -> curve calibration on train / selection on validation
    -> untouched test diagnostics
    -> NMD segmentation and governed ladder
    -> reconciled banking book + off-balance-sheet IRS
    -> cash-flow and valuation engine
    -> six prescribed EVE shocks + Tier 1 test
    -> monthly 12-month NII simulation
    -> key-assumption sensitivity
    -> assertions + versioned JSON/CSV/PNG artifacts
```

The notebooks deliberately remove the original trading appendix. A trading signal is a different use case, requires walk-forward signal estimation, transaction-cost and liquidity modelling, and should not be mixed into a regulatory/ALM engine merely because the same curve factors are available.
'''))
report_cells.append(md('''
## 4. Code architecture and controls

### 4.1 Data provenance and no look-ahead

```python
zero_yields, DATA_SOURCE = load_zero_curve_panel()
train_panel = zero_yields.iloc[:train_end]
validation_panel = zero_yields.iloc[train_end:validation_end]
test_panel = zero_yields.iloc[validation_end:]
```

Decay parameters are estimated only from training data. Candidate calibrations are selected on validation RMSE. Test results are reported once, after hyperparameters are frozen. This removes the original full-sample leakage.

### 4.2 Linear-conditional curve fitting

```python
def fit_panel(panel_values, design):
    betas = np.linalg.lstsq(design, panel_values.T, rcond=None)[0].T
    fitted = betas @ design.T
    return betas, fitted
```

Conditional on the decay parameters, NS/NSS is linear in beta. Separating nonlinear decay calibration from linear daily beta fitting makes the numerical problem auditable and allows the loading matrix to be tested directly.

### 4.3 Identification is a model requirement

```python
design_condition = np.linalg.cond(design)
max_loading_corr = max_off_diagonal_abs_corr(design[:, 1:])
vif = variance_inflation_factors(design)
assert design_condition < 1_000
assert max_loading_corr < 0.995
```

The hard screens prevent a singular model from entering downstream risk calculations. They do not prove economic identification; PCA alignment, parameter stability and outcome analysis remain necessary.

### 4.4 Basel scenario implementation

```python
short = short_shock * np.exp(-times / 4.0)
long = long_shock * (1.0 - np.exp(-times / 4.0))
steepener = -0.65 * short + 0.90 * long
flattener =  0.80 * short - 0.60 * long
```

The engine calculates six EVE scenarios and compares the maximum adverse loss with 15% of Tier 1. It does not mislabel `ΔEVE / base EVE` as the supervisory test.

### 4.5 Monthly NII rather than a one-line approximation

```python
repriced = m >= reset_month
rate = base_rate + (beta * shock if repriced else 0.0)
monthly_nii = sign * notional * max(rate, 0.0) / 12.0
```

The NII path observes repricing dates, product betas and lags month by month under a constant-balance-sheet assumption. This remains an illustrative static simulation, but it is materially closer to an ALM earnings engine than multiplying a shock by a single exposure-period scalar.
'''))
report_cells.append(md(f'''
## 5. Model results

| Metric | Four-factor Svensson | Three-factor Nelson–Siegel | Interpretation |
|---|---:|---:|---|
| Test mean RMSE | {fnum(m4['test_mean_rmse_bp'])} bp | {fnum(m3['test_mean_rmse_bp'])} bp | Svensson reconstructs the curve more closely. |
| Test 95th-percentile RMSE | {fnum(m4['test_p95_rmse_bp'])} bp | {fnum(m3['test_p95_rmse_bp'])} bp | Both are stable on the held-out sample. |
| Design condition number | {fnum(m4['design_condition_number'],1)} | {fnum(m3['design_condition_number'],1)} | Svensson is far more sensitive to small data perturbations. |
| Maximum loading correlation | {fnum(m4['maximum_loading_correlation'],3)} | {fnum(m3['maximum_loading_correlation'],3)} | The extra Svensson curvature competes with existing loadings. |
| \\|corr(PC1, Δβ0)\\| | {fnum(m4['level_factor_abs_correlation'],3)} | {fnum(m3['level_factor_abs_correlation'],3)} | NS recovers the empirical level factor; Svensson beta0 does not. |
| Mean matched PC–beta correlation | {fnum(m4['matched_factor_abs_correlation_mean'],3)} | {fnum(m3['matched_factor_abs_correlation_mean'],3)} | Structural interpretability is materially stronger in NS. |
| Worst ΔEVE / Tier 1 | {pct(m4['maximum_adverse_delta_eve_to_tier1_pct'])} | {pct(m3['maximum_adverse_delta_eve_to_tier1_pct'])} | Downstream capital results are close relative to identification differences. |
| ΔNII parallel up | {fnum(m4['delta_nii_parallel_up_mm'])} mm | {fnum(m3['delta_nii_parallel_up_mm'])} mm | NII is driven mainly by repricing architecture and deposit beta. |
| ΔNII parallel down | {fnum(m4['delta_nii_parallel_down_mm'])} mm | {fnum(m3['delta_nii_parallel_down_mm'])} mm | Asymmetric deposit pass-through creates asymmetric earnings risk. |

![Model trade-off](artifacts/figures/comparison_01_model_tradeoff.png)
'''))
report_cells.append(md('''
## 6. Diagnostic evidence from each engine

### Four-factor Svensson

![Svensson curve fit](artifacts/figures/nss4_01_curve_fit.png)

![Svensson factor alignment](artifacts/figures/nss4_02_factor_alignment.png)

![Svensson identification](artifacts/figures/nss4_03_identification.png)

The fourth factor reduces residual curve error, but the improvement is purchased with a much more ill-conditioned loading matrix. Test-period PCA changes cannot be mapped cleanly to the daily beta changes. This is a parameter-identification problem, not merely a plotting anomaly.

### Three-factor Nelson–Siegel

![NS curve fit](artifacts/figures/ns3_01_curve_fit.png)

![NS factor alignment](artifacts/figures/ns3_02_factor_alignment.png)

![NS identification](artifacts/figures/ns3_03_identification.png)

The NS model accepts a modest reconstruction penalty and produces a substantially better-conditioned, economically interpretable factor system. The held-out PCA level, slope and curvature factors map strongly to the structural beta changes.
'''))
report_cells.append(md(f'''
## 7. EVE and NII implications

![EVE comparison](artifacts/figures/comparison_02_eve.png)

The worst prescribed EVE scenario is **{m3['worst_eve_scenario']}** for the three-factor model and **{m4['worst_eve_scenario']}** for the four-factor model. The maximum adverse losses are {fnum(m3['maximum_adverse_delta_eve_mm'])} mm and {fnum(m4['maximum_adverse_delta_eve_mm'])} mm respectively, versus Tier 1 capital of {fnum(m3['tier1_capital_mm'],1)} mm.

The central model-risk result is not that the two representations produce identical valuations. It is that the downstream EVE/NII difference is small relative to the difference in parameter identification. This weakens the business case for using the extra Svensson factor as the primary structural risk-factor model.

### Why downstream similarity does not validate unstable factors

EVE is a portfolio-level functional of the fitted curve. Collinear beta coefficients can offset each other and produce a stable fitted curve even while individual factors are unstable. Therefore:

- low valuation difference does **not** prove the beta decomposition is sound;
- unstable factors can still corrupt attribution, limits, hedging narratives and management explanations;
- valuation and structural-factor models should be governed as distinct use cases.

### Behavioural-assumption sensitivity

![NMD sensitivity](artifacts/figures/comparison_03_nmd_sensitivity.png)

At 80% of the base core-deposit allocation, the worst adverse ΔEVE rises above the 15% Tier 1 outlier threshold; at 120%, it falls materially below the base case. The book is therefore more sensitive to the NMD behavioural assumption than to the difference between the two parametric curve representations.

### NMD and earnings architecture

![NMD profile](artifacts/figures/ns3_05_nmd_profile.png)

![NII results](artifacts/figures/ns3_08_nii.png)

The NII result is determined principally by asset/liability repricing gaps, deposit betas, lags and the hedge. Curve representation matters less over a 12-month parallel-shock horizon than it does for long-horizon EVE. This is precisely why Basel requires both economic-value and earnings measures.
'''))
report_cells.append(md('''
## 8. Model-risk assessment

### 8.1 Conceptual soundness

- **Svensson:** appropriate as a flexible curve representation and challenger; weak as a structural four-factor attribution model on this tenor grid unless identification improves materially.
- **Nelson–Siegel:** appropriate as a parsimonious level/slope/curvature benchmark; residual long-end shape error must be monitored and should not be hidden.
- **Valuation boundary:** neither parametric factor model should automatically replace the bank's approved discount/projection curve infrastructure.

### 8.2 Implementation verification

The two notebooks share identical input contracts and downstream engines. Assertions cover reconciliation, units, shock signs, NMD caps, condition limits and aggregation. Results are saved to a common artifact schema, reducing narrative/output drift.

### 8.3 Ongoing monitoring

Production monitoring should include:

- train/validation/test or rolling-origin reconstruction error;
- condition number, VIF and loading correlations;
- beta jump and drift thresholds;
- PCA/challenger factor alignment;
- valuation and hedge sensitivity to model choice;
- NMD, prepayment and early-redemption backtests;
- EVE/NII sensitivity to behavioural assumptions;
- data-quality, reconciliation and stale-market-data controls.

### 8.4 Change and exception governance

A model change should document purpose, affected reports, validation evidence, parallel run, threshold impacts, user acceptance, approvals and rollback plan. Exceptions to condition/factor-alignment thresholds should be time-bounded and visible to ALCO and independent risk control.
'''))
report_cells.append(md('''
## 9. Final recommendation

### Primary architecture

1. **Approved market curve service for valuation and scenario discounting.** Build or source risk-free and projection curves using the bank's approved market-data and instrument conventions. Do not describe a Fed-smoothed curve refit as primary curve construction.
2. **Three-factor Nelson–Siegel as the primary structural attribution benchmark.** It gives a stable level/slope/curvature language for risk explanation, limit attribution and challenger analysis.
3. **Four-factor Svensson as a controlled challenger/reconstruction model.** Retain it to measure long-end shape residuals and model uncertainty, not as the sole source of economic factor labels.
4. **Separate behavioural-model inventory.** NMD, prepayment and early-redemption outputs should be approved inputs with their own validation, monitoring and overlays.
5. **EVE and NII as complementary decision measures.** EVE captures full-life economic-value sensitivity; NII captures shorter-horizon earnings sensitivity. Neither substitutes for the other.
6. **ALCO decision bridge.** Translate scenario losses into hedge, funding, product-pricing and capital actions with attribution and limit utilization.

### Portfolio-project verdict

The revised project is no longer a curve-fitting demo attached to a synthetic balance sheet. It is a coherent **IRRBB model-risk case study** with temporal validation, numerical identification, behavioural modelling, capital normalization, EVE/NII integration, assumption sensitivity, audit-friendly artifacts and a defensible production migration path. It remains honestly labelled as a prototype because real bank data, local supervisory implementation, production controls and independent validation are not available.
'''))
report_cells.append(md('''
## 10. References

1. Basel Committee on Banking Supervision, **SRP31 — Interest rate risk in the banking book**, version effective 1 January 2026: https://www.bis.org/basel_framework/chapter/SRP/31.htm?inforce=20260101&published=20240716
2. Basel Committee on Banking Supervision, **Recalibration of shocks for interest rate risk in the banking book (d578)**, 16 July 2024: https://www.bis.org/bcbs/publ/d578.htm
3. Basel Committee on Banking Supervision, **SRP98 — Application guidance on IRRBB**: https://www.bis.org/basel_framework/chapter/SRP/98.htm?inforce=20260101&published=20240716
4. Basel Committee on Banking Supervision, **DIS70 — IRRBB disclosure requirements**: https://www.bis.org/basel_framework/chapter/DIS/70.htm
5. Federal Reserve Board, **Nominal Yield Curve — Gürkaynak, Sack and Wright data**: https://www.federalreserve.gov/data/nominal-yield-curve.htm

### Artifact lineage

- `artifacts/nss4/metrics.json` and associated CSV/PNG files are produced by Notebook 1.
- `artifacts/ns3/metrics.json` and associated CSV/PNG files are produced by Notebook 2.
- Every number in this report's comparison tables was read from those machine-readable result contracts when this markdown-only notebook was generated.
'''))

# Embed every PNG referenced by the markdown-only report so the notebook is
# self-contained when copied outside the repository. External PNG copies remain
# under artifacts/figures for repository browsing, README rendering and lineage.
image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+\.png)\)')
embedded_report_images = []
for cell in report_cells:
    if cell.cell_type != 'markdown':
        continue
    attachments = dict(cell.get('attachments', {}))
    source = cell.source
    for _alt, image_ref in image_pattern.findall(source):
        if image_ref.startswith('attachment:'):
            attachment_name = image_ref.split(':', 1)[1]
            candidates = [FIG / attachment_name]
        else:
            relative_path = Path(image_ref)
            attachment_name = relative_path.name
            candidates = [ROOT / relative_path, FIG / attachment_name]
            source = source.replace(
                f']({image_ref})',
                f'](attachment:{attachment_name})',
            )

        image_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if image_path is None:
            raise FileNotFoundError(
                f'Report image is missing: {image_ref}; searched {candidates}'
            )

        attachments[attachment_name] = {
            'image/png': base64.b64encode(image_path.read_bytes()).decode('ascii')
        }
        embedded_report_images.append(attachment_name)

    cell.source = source
    if attachments:
        cell['attachments'] = attachments

assert embedded_report_images, 'No report figures were embedded'
assert len(set(embedded_report_images)) == 12, embedded_report_images

report_nb = nbf.v4.new_notebook(cells=report_cells)
report_nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'markdown'},
    'project': {'report_type': 'markdown-only model-risk case study', 'artifact_contract': 'v1.0'},
})
report_path = ROOT / '03_model_risk_business_architecture_case_study.ipynb'
nbf.write(report_nb, report_path)
notebook_paths.append(report_path)

# README and requirements make the project usable without weakening the three-notebook request.
README_TEXT = "# IRRBB Term-Structure Model Risk and Banking-Book Measurement Engine\n\n## Executive summary\n\nThis project is a portfolio-grade prototype for **Interest Rate Risk in the Banking Book (IRRBB)**. It compares two parametric term-structure representations inside the same controlled banking-book measurement architecture:\n\n1. a **four-factor Nelson–Siegel–Svensson (NSS)** model, used as the more flexible curve-reconstruction challenger; and\n2. a **three-factor Nelson–Siegel (NS)** model, used as the more parsimonious structural level–slope–curvature benchmark.\n\nThe project does not stop at curve fitting. Both models feed an identical downstream engine covering:\n\n- temporal model development and out-of-sample validation;\n- parameter-identification and PCA factor-alignment diagnostics;\n- behavioural non-maturity deposit (NMD) segmentation;\n- a synthetic but reconciled banking book;\n- contractual and behavioural cash-flow generation;\n- economic value of equity (EVE) measurement;\n- prescribed IRRBB shock scenarios and Tier 1 normalisation;\n- twelve-month constant-balance-sheet net interest income (NII) simulation;\n- key-assumption sensitivity; and\n- machine-readable model artifacts for independent comparison.\n\nThe central model-risk question is:\n\n> **Does a better in-sample or out-of-sample curve fit justify a more complex model when its factors are poorly identified and the downstream EVE/NII decision impact is immaterial?**\n\nThe reference results show that the four-factor Svensson model reconstructs the observed curve slightly more accurately, while the three-factor Nelson–Siegel model provides materially stronger factor identification and economic interpretability. The difference between the two models' EVE and NII outputs is negligible relative to the sensitivity created by the behavioural NMD assumption. The recommended architecture therefore separates the purposes of **valuation**, **structural risk attribution**, **model challenge**, and **funds-transfer pricing** rather than forcing one parametric curve to perform all four roles.\n\n> **Scope statement:** this is an auditable research and portfolio prototype, not a production regulatory engine, an approved internal measurement system, or a complete implementation of the Basel standardised framework.\n\n---\n\n## 1. Problem statement\n\nBanks transform maturities and reprice assets and liabilities at different times. A change in the level or shape of interest-rate curves can therefore alter both:\n\n- the present value of future banking-book cash flows; and\n- the interest income and expense recognised over the planning horizon.\n\nThis creates three fundamental IRRBB sources:\n\n- **Gap risk:** assets and liabilities mature or reprice at different dates.\n- **Basis risk:** economically related positions reference rates that do not move together.\n- **Option risk:** customers or the bank can alter contractual cash-flow timing, for example through loan prepayment, deposit early redemption or administered-rate behaviour.\n\nA technically attractive yield-curve fit does not automatically produce a sound risk model. A complex parameterisation can achieve low reconstruction error while suffering from:\n\n- near-collinear loading functions;\n- unstable or offsetting coefficients;\n- poor correspondence with empirical yield-curve factors;\n- weak interpretability for ALCO, Treasury and independent validation; and\n- little measurable benefit in the final EVE or NII decision metrics.\n\nThe project is designed to test that distinction explicitly.\n\n### Decision questions\n\nThe two model notebooks answer the following questions under the same data, balance sheet and behavioural assumptions:\n\n1. Which model reconstructs the term structure more accurately on an untouched test sample?\n2. Which model has the more stable and identifiable loading matrix?\n3. Do the estimated beta changes align with empirical PCA factors?\n4. Does the more flexible curve representation materially alter EVE or NII?\n5. Are model-form differences more important than behavioural NMD assumptions?\n6. Which model should be used for valuation, attribution, challenge and governance?\n\n---\n\n## 2. Regulatory and business context\n\nIRRBB is treated within the Basel Pillar 2 framework. The project uses the current Basel architecture as the organising business framework rather than presenting the exercise as a generic fixed-income notebook.\n\nThe prototype reflects the following principles:\n\n- IRRBB must be evaluated from both an **economic-value** and an **earnings** perspective.\n- Prescribed scenarios should capture parallel and non-parallel changes in the term structure.\n- NMDs require behavioural segmentation into stable/non-stable and core/non-core components.\n- Fixed-rate loans subject to prepayment and term deposits subject to early redemption require explicit option assumptions.\n- The maximum adverse change in EVE is normalised by Tier 1 capital for the supervisory outlier diagnostic.\n- Model assumptions, data provenance, validation evidence, limitations and management use must be governed independently from model development.\n\n### Measures represented in the project\n\n#### Economic value of equity\n\nFor scenario \\(s\\), the engine calculates:\n\n$$\nEVE_s = PV_s(\\text{assets}) - PV_s(\\text{liabilities}) + PV_s(\\text{off-balance-sheet positions})\n$$\n\nand:\n\n$$\n\\Delta EVE_s = EVE_s - EVE_{base}.\n$$\n\nThe supervisory diagnostic is represented as:\n\n$$\n\\frac{\\max_s\\left(-\\Delta EVE_s,0\\right)}{\\text{Tier 1 capital}}.\n$$\n\n#### Net interest income\n\nThe project simulates a twelve-month constant-balance-sheet earnings path and compares shocked and base projections:\n\n$$\n\\Delta NII_s = NII_s^{12m} - NII_{base}^{12m}.\n$$\n\nThe NII engine applies contractual or behavioural repricing dates, asymmetric deposit betas, repricing lags and swap carry.\n\n### Prescribed scenario families\n\nThe EVE engine includes:\n\n- parallel up;\n- parallel down;\n- short-rate up;\n- short-rate down;\n- steepener; and\n- flattener.\n\nThe NII engine evaluates the two prescribed parallel scenarios over the forward twelve-month horizon.\n\n---\n\n## 3. Proposed solution\n\nThe solution is deliberately modular. It separates the components a bank would ordinarily govern through different owners and controls:\n\n1. **Market-data and curve service**  \n   Acquires an approved zero-curve panel or loads the deterministic offline reference panel. Data provenance is recorded as an output.\n\n2. **Term-structure model-development layer**  \n   Calibrates decay hyperparameters on training data, selects them using validation performance and freezes them before test evaluation.\n\n3. **Model-risk validation layer**  \n   Evaluates fit, numerical conditioning, loading collinearity, variance inflation, PCA factor alignment and basic curve-shape plausibility.\n\n4. **Behavioural-model layer**  \n   Segments NMDs and maps core balances into replicating maturity ladders subject to transparent caps and behavioural assumptions.\n\n5. **Position and cash-flow layer**  \n   Represents fixed, floating, amortising and derivative positions while distinguishing contractual maturity, next repricing and behavioural maturity.\n\n6. **IRRBB measurement layer**  \n   Produces repricing gaps, EVE, prescribed-scenario sensitivities, capital-normalised diagnostics and twelve-month NII.\n\n7. **Governance and reporting layer**  \n   Writes a common artifact contract for the two models, enabling a markdown-only case study to compare them without hidden recalculation or manual transcription.\n\n---\n\n## 4. Target banking architecture\n\n![IRRBB business architecture](artifacts/figures/business_architecture.png)\n\nA production bank implementation would normally sit within the following operating model:\n\n```text\nBoard / Risk Committee\n        │\n        ├── approves IRRBB framework and risk appetite\n        │\nALCO / Treasury Management\n        │\n        ├── funding, hedging, pricing and balance-sheet actions\n        │\nIndependent Risk Management\n        │\n        ├── limits, monitoring, challenge and escalation\n        │\nModel Risk Management / Internal Validation\n        │\n        ├── conceptual soundness, data, implementation and outcomes analysis\n        │\nApproved Data and Curve Services\n        │\n        ├── market data, product data, behavioural data and regulatory capital\n        │\nBehavioural Models\n        │\n        ├── NMDs, prepayment, early redemption and administered rates\n        │\nCash-Flow and Repricing Engine\n        │\n        ├── contractual + behavioural cash flows and optionality\n        │\nIRRBB Measurement Engine\n        │\n        ├── EVE, NII, gap, basis and option-risk views\n        │\nAggregation and Reporting\n        │\n        └── currency/entity aggregation, limits, disclosures and management MI\n```\n\n### Purpose-specific model architecture\n\nA key conclusion of the project is that one curve model should not automatically be used for every purpose.\n\n| Purpose | Recommended component |\n|---|---|\n| Valuation and scenario discounting | Approved market curve and instrument-convention service |\n| Structural level/slope/curvature attribution | Three-factor Nelson–Siegel benchmark |\n| Flexible reconstruction and challenger testing | Four-factor Nelson–Siegel–Svensson |\n| Funds-transfer pricing | Separate FTP stack including funding, liquidity, basis and optionality components |\n| Behavioural cash-flow generation | Independently developed and validated NMD/prepayment/early-redemption models |\n\nThe notebooks refit an observed zero-curve panel for controlled model comparison. They do **not** claim to bootstrap a production discount curve from raw instruments.\n\n---\n\n## 5. Notebook architecture\n\n### `01_four_factor_svensson_irrbb_engine.ipynb`\n\nImplements the four-beta Nelson–Siegel–Svensson challenger:\n\n$$\nz(\\tau)=\\beta_0+\\beta_1L_1(\\tau;\\lambda_1)+\\beta_2L_2(\\tau;\\lambda_1)+\\beta_3L_3(\\tau;\\lambda_2).\n$$\n\nMain responsibilities:\n\n- market-data loading and provenance;\n- train/validation/test segmentation;\n- two-decay calibration;\n- daily linear least-squares beta estimation;\n- out-of-sample fit diagnostics;\n- loading-matrix identification tests;\n- PCA factor alignment;\n- NMD calibration and replicating portfolio;\n- banking-book construction;\n- EVE and NII calculation;\n- artifact generation.\n\n### `02_three_factor_nelson_siegel_irrbb_engine.ipynb`\n\nImplements the parsimonious three-beta Nelson–Siegel model:\n\n$$\nz(\\tau)=\\beta_0+\\beta_1L_1(\\tau;\\lambda_1)+\\beta_2L_2(\\tau;\\lambda_1).\n$$\n\nIt uses the same data contract, behavioural assumptions, balance sheet, scenarios, cash-flow engine, controls and artifact schema as Notebook 1. The only intentional model-form difference is the term-structure representation.\n\n### `03_model_risk_business_architecture_case_study.ipynb`\n\nA **markdown-only** model-risk report. It consumes the CSV, JSON and PNG artifacts generated by the first two notebooks and covers:\n\n- the IRRBB business problem;\n- governance and target operating model;\n- implementation architecture;\n- selected code snippets;\n- model-development controls;\n- quantitative comparison;\n- EVE and NII implications;\n- NMD assumption sensitivity;\n- model-risk assessment;\n- final use-case recommendation; and\n- migration requirements for a production bank implementation.\n\nThe third notebook contains no executable code cells. This prevents analysis logic from being hidden inside the final narrative and makes the provenance of every reported result explicit.\n\n---\n\n## 6. End-to-end analytical workflow\n\n```text\nConfiguration and unit contract\n        ↓\nMarket-data acquisition / offline reference panel\n        ↓\nChronological 60% / 20% / 20% split\n        ↓\nDecay calibration on training data\n        ↓\nHyperparameter selection on validation data\n        ↓\nFrozen-model evaluation on untouched test data\n        ↓\nDaily beta estimation and curve reconstruction\n        ↓\nCondition number, loading correlation and VIF\n        ↓\nTraining PCA → test-period factor alignment\n        ↓\nNMD segmentation and behavioural maturity ladder\n        ↓\nBanking-book construction and reconciliation\n        ↓\nContractual + behavioural cash-flow generation\n        ↓\nBase EVE and six prescribed EVE scenarios\n        ↓\nMaximum adverse ΔEVE / Tier 1 diagnostic\n        ↓\nTwelve-month constant-balance-sheet ΔNII\n        ↓\nNMD key-assumption sensitivity\n        ↓\nVersioned JSON / CSV / PNG artifact contract\n        ↓\nMarkdown-only model-risk case study\n```\n\n### Temporal validation design\n\nThe observations are split chronologically:\n\n- **60% training:** decay calibration and PCA estimation;\n- **20% validation:** hyperparameter/model selection;\n- **20% test:** untouched final performance evaluation.\n\nNo test observation is used to calibrate decay parameters or fit PCA. This avoids the look-ahead leakage present when a single global parameter is calibrated using the entire history and then reported as historical performance.\n\n---\n\n## 7. Model-risk diagnostics\n\nThe project does not treat low RMSE as sufficient model evidence.\n\n### Reconstruction performance\n\nFor every date, curve error is measured in basis points. The artifact contract reports:\n\n- training mean RMSE;\n- validation mean RMSE;\n- test mean RMSE;\n- test 95th-percentile RMSE; and\n- maximum test point error.\n\n### Numerical identification\n\nThe loading design matrix is evaluated using:\n\n- condition number;\n- maximum pairwise loading correlation; and\n- variance inflation factors.\n\nThese diagnostics identify cases where different beta coefficients can offset one another while leaving the fitted curve nearly unchanged.\n\n### Empirical factor alignment\n\nPCA is fit only to training-period daily yield changes. Test-period changes are transformed into that frozen PCA basis. Model beta changes are then compared with empirical PCs.\n\nBecause PCA signs are arbitrary, absolute correlations are used for governance statistics. A Hungarian assignment maps model factors to the empirical PCs that maximise total absolute correspondence.\n\n### Curve-shape plausibility\n\nThe notebooks test whether fitted curves produce plausible:\n\n- discount factors;\n- discount-factor monotonicity; and\n- instantaneous forward-rate ranges.\n\nThese are diagnostics, not a proof that the parametric representation is fully arbitrage-free.\n\n### Hard controls\n\nAssertions cover:\n\n- time ordering and missing values;\n- finite model outputs;\n- loading-matrix governance thresholds;\n- balance-sheet reconciliation;\n- NMD caps and totals;\n- scenario sign conventions;\n- aggregation consistency; and\n- artifact existence.\n\n---\n\n## 8. Behavioural NMD architecture\n\nThe NMD prototype separates:\n\n- retail transactional deposits;\n- retail non-transactional deposits; and\n- wholesale deposits.\n\nFor each segment it represents:\n\n- current balance;\n- stable balance estimate;\n- core proportion;\n- governance cap;\n- average behavioural maturity cap;\n- beta under rising rates;\n- beta under falling rates;\n- repricing lag; and\n- core/non-core repricing ladder.\n\nThe reference configuration applies the Basel standardised-framework caps used in the notebook:\n\n| Segment | Maximum core proportion | Maximum average maturity |\n|---|---:|---:|\n| Retail transactional | 90% | 5.0 years |\n| Retail non-transactional | 70% | 4.5 years |\n| Wholesale | 50% | 4.0 years |\n\nNon-core balances are treated as overnight. Core balances are distributed across a transparent replicating maturity ladder. The engine then reruns EVE after scaling the core NMD allocation to demonstrate key-assumption risk.\n\nThis is an illustrative behavioural layer. It is not a substitute for account-level survival, attrition, administered-rate and pass-through models.\n\n---\n\n## 9. Banking-book and cash-flow representation\n\nThe synthetic banking book is constructed to demonstrate architecture rather than reproduce a specific institution.\n\nThe position schema distinguishes:\n\n- asset, liability and derivative side;\n- notional amount;\n- coupon rate and frequency;\n- fixed or floating rate type;\n- reference index;\n- legal maturity;\n- next repricing date;\n- amortisation type;\n- prepayment and early-redemption assumptions;\n- NII pass-through beta;\n- repricing lag; and\n- off-balance-sheet fixed-leg direction.\n\nThe book includes representative:\n\n- floating-rate corporate and SME lending;\n- retail and mortgage assets;\n- fixed-rate securities and loans;\n- wholesale and term funding;\n- core and non-core NMDs; and\n- an interest-rate swap.\n\nAccounting equity reconciles assets and liabilities. Tier 1 capital is maintained as a separate regulatory input rather than being equated automatically with accounting equity.\n\n### Valuation boundary\n\nThe project implements an **IMS-style full-cash-flow proxy**:\n\n- fixed-rate assets and liabilities generate contractual coupons and principal;\n- amortising assets follow transparent schedules;\n- floating positions are represented to the next reset;\n- prepayment and early-redemption assumptions change scenario cash-flow timing;\n- NMD core balances follow behavioural ladders; and\n- swaps contribute fixed-leg and floating-leg economics.\n\nIt remains simplified relative to production valuation because it does not implement all instrument conventions, multiple projection curves, basis curves, customer-level optionality or hedge accounting.\n\n---\n\n## 10. Reference results\n\nThe committed notebook outputs were generated using the deterministic offline reference panel so that the repository remains executable without network access. These values are demonstration results, not current market-risk numbers.\n\n### Term-structure model comparison\n\n| Metric | Four-factor Svensson | Three-factor Nelson–Siegel | Preferred result |\n|---|---:|---:|---|\n| Test mean RMSE | **0.496 bp** | 0.556 bp | Svensson |\n| Test 95th-percentile RMSE | **0.744 bp** | 0.828 bp | Svensson |\n| Test maximum point error | 2.080 bp | **2.075 bp** | Approximately equal |\n| Loading condition number | 231.5 | **17.5** | Nelson–Siegel |\n| Maximum loading correlation | 0.984 | **0.545** | Nelson–Siegel |\n| `|corr(PC1, Δβ0)|` | 0.246 | **0.924** | Nelson–Siegel |\n| Mean matched PC–beta correlation | 0.201 | **0.933** | Nelson–Siegel |\n\n### IRRBB outputs\n\n| Metric | Four-factor Svensson | Three-factor Nelson–Siegel |\n|---|---:|---:|\n| Base EVE | 277.370 mm | 277.365 mm |\n| Worst EVE scenario | Parallel up | Parallel up |\n| Maximum adverse ΔEVE | 25.051 mm | 25.050 mm |\n| Maximum adverse ΔEVE / Tier 1 | 14.736% | 14.735% |\n| Basel outlier threshold breached | No | No |\n| Base twelve-month NII | 49.184 mm | 49.184 mm |\n| ΔNII, parallel up | +1.885 mm | +1.885 mm |\n| ΔNII, parallel down | −2.482 mm | −2.482 mm |\n\n### NMD key-assumption sensitivity\n\nUsing the three-factor engine:\n\n| Core NMD scale | Worst adverse ΔEVE / Tier 1 | Diagnostic result |\n|---:|---:|---|\n| 80% | **17.177%** | Threshold breached |\n| 100% | 14.735% | Below threshold, high utilisation |\n| 120% | 12.894% | Below threshold |\n\n### Interpretation\n\nThe four-factor model buys approximately 0.06 bp of mean test RMSE improvement, but its loading matrix is substantially less identifiable and its beta changes have weak correspondence with empirical curve factors.\n\nThe two models generate almost identical EVE and NII outputs under the common downstream engine. By contrast, changing the core NMD assumption materially changes the Tier 1-normalised EVE result and can move the illustrative bank across the supervisory threshold.\n\nThe business conclusion is therefore:\n\n> **Behavioural-model risk dominates term-structure model-form risk in this reference banking book.**\n\n---\n\n## 11. Final model recommendation\n\n### Primary structural model\n\nUse the **three-factor Nelson–Siegel model** as the structural risk-attribution benchmark because it provides:\n\n- materially better numerical conditioning;\n- clearer level–slope–curvature interpretation;\n- stronger alignment with empirical PCA factors; and\n- essentially unchanged downstream EVE/NII decisions in the reference case.\n\n### Challenger model\n\nRetain the **four-factor Svensson model** as:\n\n- a flexible reconstruction challenger;\n- a residual-shape diagnostic;\n- a long-end fit benchmark; and\n- a model-risk sensitivity tool.\n\nIt should not be treated automatically as a four-factor economic attribution model when the additional loading is nearly collinear with existing factors.\n\n### Valuation curve\n\nNeither notebook should replace the bank's approved curve-construction service. Production valuation should use approved market instruments, conventions, interpolation, bootstrapping, projection/discount curves and governance controls.\n\n### Behavioural models\n\nNMD, prepayment, early-redemption and administered-rate models should be independently estimated, validated, monitored and subjected to sensitivity and stress testing because their impact can exceed the difference between curve parameterisations.\n\n---\n\n## 12. Data modes and provenance\n\nData mode is controlled by the `IRRBB_DATA_MODE` environment variable.\n\n| Mode | Behaviour |\n|---|---|\n| `AUTO` | Attempts to download the Federal Reserve Gürkaynak–Sack–Wright zero-curve panel; falls back to the labelled offline panel if unavailable. |\n| `LIVE` | Requires the Federal Reserve download to succeed and fails rather than silently substituting synthetic data. |\n| `OFFLINE` | Uses the deterministic reference panel committed under `data/`. |\n\nExample:\n\n```bash\nexport IRRBB_DATA_MODE=LIVE\njupyter lab\n```\n\nThe Federal Reserve `SVENYxx` series are continuously compounded zero-coupon yields produced by the Federal Reserve's nominal-yield-curve research model. They are already smoothed curve estimates; the notebooks refit those observed tenor points for model comparison and do not claim to construct the primary market curve from raw Treasury prices.\n\n---\n\n## 13. Project structure\n\n```text\nirrbb_portfolio_project/\n├── 01_four_factor_svensson_irrbb_engine.ipynb\n├── 02_three_factor_nelson_siegel_irrbb_engine.ipynb\n├── 03_model_risk_business_architecture_case_study.ipynb\n├── README.md\n├── requirements.txt\n├── build_project.py\n├── data/\n│   └── offline_reference_zero_curve.csv\n└── artifacts/\n    ├── figures/\n    │   ├── business_architecture.png\n    │   ├── comparison_01_model_tradeoff.png\n    │   ├── comparison_02_eve.png\n    │   ├── comparison_03_nmd_sensitivity.png\n    │   ├── ns3_*.png\n    │   └── nss4_*.png\n    ├── ns3/\n    │   ├── metrics.json\n    │   ├── curve_parameters.csv\n    │   ├── fit_diagnostics.csv\n    │   ├── factor_correlations.csv\n    │   ├── nmd_summary.csv\n    │   ├── nmd_ladder.csv\n    │   ├── repricing_gap.csv\n    │   ├── eve_scenarios.csv\n    │   ├── nii_attribution.csv\n    │   └── nmd_sensitivity.csv\n    └── nss4/\n        └── same artifact contract as ns3/\n```\n\n---\n\n## 14. Artifact contract\n\nEach model notebook writes the same output schema under its model key.\n\n### `metrics.json`\n\nContains high-level model and risk metrics, including:\n\n- data source and reference date;\n- sample sizes;\n- calibrated decay parameters;\n- condition number and loading correlation;\n- train/validation/test fit statistics;\n- factor-alignment statistics;\n- NMD summary metrics;\n- accounting equity and Tier 1 capital;\n- base EVE and maximum adverse ΔEVE;\n- outlier diagnostic; and\n- base and shocked NII.\n\n### CSV artifacts\n\n| File | Purpose |\n|---|---|\n| `curve_parameters.csv` | Daily beta and decay-parameter estimates |\n| `fit_diagnostics.csv` | Date-level reconstruction errors and sample labels |\n| `factor_correlations.csv` | PCA-to-beta change correlations |\n| `nmd_summary.csv` | Segment-level behavioural assumptions and core amounts |\n| `nmd_ladder.csv` | Core/non-core behavioural repricing allocation |\n| `repricing_gap.csv` | Nineteen-bucket repricing-gap view |\n| `eve_scenarios.csv` | Scenario EVE, ΔEVE, loss and Tier 1 utilisation |\n| `nii_attribution.csv` | Product-level twelve-month NII attribution |\n| `nmd_sensitivity.csv` | Core NMD assumption sensitivity |\n\nThis common contract allows the report notebook to remain markdown-only and prevents narrative numbers from drifting away from model outputs.\n\n---\n\n## 15. Installation\n\n### Tested environment\n\nThe committed outputs were generated and checked under:\n\n- **Python 3.13.5**\n- macOS/Linux-compatible Python environment\n- Jupyter Notebook 7 / JupyterLab 4\n\nCreate and activate a virtual environment:\n\n```bash\npython3.13 -m venv .venv\nsource .venv/bin/activate\npython -m pip install --upgrade pip\npip install -r requirements.txt\n```\n\nOn Windows PowerShell:\n\n```powershell\npy -3.13 -m venv .venv\n.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\npip install -r requirements.txt\n```\n\nLaunch Jupyter:\n\n```bash\njupyter lab\n```\n\nThe requirements file pins the direct runtime and notebook-execution dependencies used for the committed build. Standard-library modules such as `pathlib`, `dataclasses`, `json`, `io`, `os` and `warnings` are not listed.\n\n---\n\n## 16. Execution order\n\nRun from the project root:\n\n1. `01_four_factor_svensson_irrbb_engine.ipynb`\n2. `02_three_factor_nelson_siegel_irrbb_engine.ipynb`\n3. Open `03_model_risk_business_architecture_case_study.ipynb`\n\nThe first two notebooks regenerate their model-specific artifacts. The third notebook is a committed markdown snapshot built from those machine-readable outputs.\n\nFor the deterministic committed results:\n\n```bash\nexport IRRBB_DATA_MODE=OFFLINE\njupyter lab\n```\n\nFor current Federal Reserve data:\n\n```bash\nexport IRRBB_DATA_MODE=LIVE\njupyter lab\n```\n\nWhen live data are used, numerical results and the report snapshot should be regenerated before presentation.\n\n---\n\n## 17. Reproducibility and controls\n\nThe project supports reproducibility through:\n\n- an exact pinned Python dependency set;\n- deterministic offline input data;\n- explicit units for yields, shocks and monetary amounts;\n- chronological validation splits;\n- fixed random seeds where stochastic generation is used;\n- common configuration and artifact contracts;\n- hard assertions for integration and governance checks;\n- saved figures and tabular outputs; and\n- a transparent `build_project.py` script used to assemble and validate the package.\n\nThe build script is included for auditability. It should be reviewed before use in a different environment because rebuilding the package regenerates notebooks and reference artifacts.\n\n---\n\n## 18. Technology stack\n\n| Layer | Library | Version | Use |\n|---|---|---:|---|\n| Numerical computing | NumPy | 2.3.5 | Arrays, linear algebra and vectorised calculations |\n| Data engineering | pandas | 2.2.3 | Time-series panels, banking-book tables and artifact export |\n| Scientific optimisation | SciPy | 1.17.0 | Hyperparameter search support and Hungarian factor matching |\n| Machine learning | scikit-learn | 1.8.0 | PCA estimation and transformation |\n| Visualisation | Matplotlib | 3.10.8 | Curve, diagnostic, EVE, NII and sensitivity figures |\n| HTTP/data access | Requests | 2.32.5 | Federal Reserve data acquisition |\n| Notebook display | IPython | 9.14.0 | Rich dataframe and notebook display |\n| Notebook format | nbformat | 5.10.4 | Programmatic notebook construction and validation |\n| Notebook execution | nbclient | 0.10.4 | Clean-kernel notebook execution in the build workflow |\n| Kernel | ipykernel | 7.2.0 | Python Jupyter kernel |\n| Notebook UI | notebook | 7.5.3 | Classic notebook-compatible interface |\n| Lab UI | jupyterlab | 4.5.3 | Interactive notebook development environment |\n\n---\n\n## 19. Known limitations\n\nThe project intentionally remains a portfolio prototype. Important limitations include:\n\n- synthetic banking-book positions and customer behaviour;\n- no account-level NMD survival or rate-pass-through estimation;\n- no empirical prepayment or term-deposit early-redemption model;\n- no multi-curve discounting and projection framework;\n- limited basis-risk representation;\n- simplified product day-count, calendar and compounding conventions;\n- no stochastic dynamic-balance-sheet forecast;\n- no currency aggregation across legal entities;\n- no hedge-accounting or accounting-classification treatment;\n- no CSRBB engine;\n- no production data lineage, access control or model inventory integration;\n- no independent code validation against a separate implementation; and\n- no national-jurisdiction overlay beyond the Basel reference architecture.\n\nThe results must therefore be interpreted as a controlled model-risk case study, not as a bank's regulatory capital or disclosure calculation.\n\n---\n\n## 20. Production migration roadmap\n\nA production implementation would require at least the following extensions:\n\n1. **Approved market-data and curve construction**  \n   Instrument-level inputs, bootstrapping, discount/projection separation, interpolation standards, fallback procedures and independent price verification.\n\n2. **Full product cash-flow library**  \n   Day-count conventions, calendars, amortisation, caps/floors, callable structures, derivatives, behavioural options and instrument-level reconciliation.\n\n3. **Account-level behavioural modelling**  \n   NMD stability, attrition, migration, administered rates, deposit betas, prepayment and early-redemption models with backtesting and monitoring.\n\n4. **Basis and currency architecture**  \n   Multiple indices, currencies, legal entities, netting rules and materiality thresholds.\n\n5. **Dynamic earnings simulation**  \n   New business, replacement assumptions, volume forecasts, commercial margins, management actions and scenario-consistent balance-sheet evolution.\n\n6. **Model governance**  \n   Model inventory, ownership, independent validation, change control, limitations, performance thresholds, overrides and periodic review.\n\n7. **Technology and controls**  \n   Versioned data pipelines, automated testing, CI/CD, access control, run logging, lineage, reconciliation, exception management and reproducible reporting.\n\n8. **Regulatory mapping**  \n   Formal mapping to applicable local supervisory rules, disclosure templates, internal limits and board-approved risk appetite.\n\n---\n\n## 21. References\n\n1. Basel Committee on Banking Supervision, **SRP31 — Interest rate risk in the banking book**, version effective 1 January 2026.  \n   https://www.bis.org/basel_framework/chapter/SRP/31.htm?inforce=20260101&published=20240716\n\n2. Basel Committee on Banking Supervision, **SRP98 — Application guidance on interest rate risk in the banking book**, version effective 1 January 2026.  \n   https://www.bis.org/basel_framework/chapter/SRP/98.htm?inforce=20260101&published=20240716\n\n3. Basel Committee on Banking Supervision, **Recalibration of shocks for interest rate risk in the banking book**, 16 July 2024.  \n   https://www.bis.org/bcbs/publ/d578.htm\n\n4. Basel Committee on Banking Supervision, **DIS70 — Interest rate risk in the banking book disclosure requirements**.  \n   https://www.bis.org/basel_framework/chapter/DIS/70.htm\n\n5. Federal Reserve Board, **Nominal Yield Curve — Gürkaynak, Sack and Wright data**.  \n   https://www.federalreserve.gov/data/nominal-yield-curve.htm\n\n6. Gürkaynak, R. S., Sack, B. and Wright, J. H., **The U.S. Treasury Yield Curve: 1961 to the Present**, Finance and Economics Discussion Series 2006-28.\n\n---\n\n## 22. Portfolio positioning\n\nThis repository is intended to demonstrate capability across:\n\n- Treasury and ALM analytics;\n- IRRBB and balance-sheet risk;\n- fixed-income term-structure modelling;\n- numerical model-risk diagnostics;\n- behavioural assumption governance;\n- regulatory interpretation;\n- Python-based analytical engineering; and\n- executive model-risk communication.\n\nThe strongest result is not the selection of one curve formula over another. It is the construction of a defensible decision framework showing **where model complexity adds value, where it creates identification risk, and which assumptions actually drive the bank's capital and earnings sensitivity**.\n"
(ROOT / 'README.md').write_text(README_TEXT, encoding='utf-8')
REQUIREMENTS_TEXT = '# IRRBB portfolio project — reproducible direct dependencies\n# Reference runtime used to execute the committed notebooks and artifacts:\n# Python 3.13.5\n#\n# Install with:\n#   python -m pip install --upgrade pip\n#   pip install -r requirements.txt\n\n# Core numerical and data stack\nnumpy==2.3.5\npandas==2.2.3\nscipy==1.17.0\nscikit-learn==1.8.0\n\n# Visualisation and external data access\nmatplotlib==3.10.8\nrequests==2.32.5\n\n# Notebook runtime and rich display\nipython==9.14.0\nipykernel==7.2.0\njupyterlab==4.5.3\nnotebook==7.5.3\n\n# Programmatic notebook creation, execution and validation\nnbformat==5.10.4\nnbclient==0.10.4\n'
(ROOT / 'requirements.txt').write_text(REQUIREMENTS_TEXT, encoding='utf-8')

# Validate markdown-only report and artifact links.
loaded_report = nbf.read(report_path, as_version=4)
assert all(c.cell_type == 'markdown' for c in loaded_report.cells)
assert len(loaded_report.cells) >= 10
external_png_refs = []
embedded_png_names = set()
for cell_index, cell in enumerate(loaded_report.cells):
    if cell.cell_type != 'markdown':
        continue
    for _alt, image_ref in image_pattern.findall(cell.source):
        assert image_ref.startswith('attachment:'), (cell_index, image_ref)
        embedded_png_names.add(image_ref.split(':', 1)[1])
    embedded_png_names.update(cell.get('attachments', {}).keys())
assert len(embedded_png_names) == 12, sorted(embedded_png_names)
for p in [
    comparison_figure, eve_comparison, nmd_sensitivity_figure, architecture_figure,
    FIG/'nss4_01_curve_fit.png', FIG/'ns3_01_curve_fit.png',
    ART/'nss4'/'metrics.json', ART/'ns3'/'metrics.json',
]:
    assert p.exists(), p

# Copy build script into the project for transparent regeneration when the
# builder is executed from another location. Avoid copying a file onto itself.
build_script_target = ROOT / 'build_project.py'
if Path(__file__).resolve() != build_script_target.resolve():
    shutil.copy2(__file__, build_script_target)

zip_path = Path('/mnt/data/irrbb_portfolio_project.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in ROOT.rglob('*'):
        if path.is_file():
            z.write(path, path.relative_to(ROOT.parent))

print('PROJECT', ROOT)
print('ZIP', zip_path)
print('NOTEBOOKS')
for p in notebook_paths:
    print(p, p.stat().st_size)
print('METRICS 4', json.dumps(m4, indent=2)[:1200])
print('METRICS 3', json.dumps(m3, indent=2)[:1200])
