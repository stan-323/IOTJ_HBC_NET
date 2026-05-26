from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
HFSS_PACKAGE_ROOT = WORKSPACE_ROOT / "hfss"
HFSS_TORSO_ROOT = HFSS_PACKAGE_ROOT / "torso_main"
HFSS_LIBRARY_ROOT = HFSS_TORSO_ROOT / "outputs"
SCHED_LIBRARY_PATH = HFSS_LIBRARY_ROOT / "torso_scheduler_library.csv"
SCHED_LIBRARY_PATH_ENV = "IOTJ_HFSS_SCHED_LIBRARY_PATH"
SCHED_LIBRARY_PATH_ENV_NAMES = (
    SCHED_LIBRARY_PATH_ENV,
    "IOTJ_HFSS_LIBRARY_PATH",
    "HFSS_SCHED_LIBRARY_PATH",
)
ALLOWED_SCHED_LIBRARY_FILENAMES = {
    "calibrated_library_sched.csv",
    "torso_scheduler_library.csv",
    "torso_proxy_library.csv",
}

REQUIRED_LABELS = {
    "surface_rest",
    "surface_moderate_loose",
    "surface_contact_failure",
    "surface_sweat",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
}
REQUIRED_COLUMNS = {
    "label",
    "type",
    "g_norm",
    "r_norm",
    "chi_norm",
    "p_rx_cap_norm",
    "sched_include_main",
}


@dataclass(frozen=True)
class HFSSRow:
    label: str
    kind: str
    depth_mm: float
    condition: str
    sched_include_main: bool
    g_norm: float
    r_norm: float
    chi_norm: float
    p_rx_cap_norm: float


@dataclass(frozen=True)
class HFSSLibrary:
    source_path: Path
    rows: dict[str, HFSSRow]

    def coeff(self, label: str) -> HFSSRow:
        try:
            return self.rows[label]
        except KeyError as exc:
            raise KeyError(f"Unknown HFSS label: {label}") from exc

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.__dict__ for row in self.rows.values()])


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _selected_library_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    for env_name in SCHED_LIBRARY_PATH_ENV_NAMES:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return Path(env_value).expanduser()
    return SCHED_LIBRARY_PATH


def load_hfss_library(path: str | Path | None = None) -> HFSSLibrary:
    source = _selected_library_path(path)
    if source.name not in ALLOWED_SCHED_LIBRARY_FILENAMES:
        allowed = ", ".join(sorted(ALLOWED_SCHED_LIBRARY_FILENAMES))
        raise ValueError(f"Unsupported HFSS scheduling library filename: {source.name}; allowed filenames: {allowed}")
    if not source.exists():
        raise FileNotFoundError(f"HFSS scheduling library not found: {source}")
    data = pd.read_csv(source)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"HFSS scheduling library missing columns: {sorted(missing_columns)}")

    rows: dict[str, HFSSRow] = {}
    for _, item in data.iterrows():
        label = str(item["label"])
        rows[label] = HFSSRow(
            label=label,
            kind=str(item["type"]),
            depth_mm=float(item.get("depth_mm", 0.0)),
            condition=str(item.get("condition", "")),
            sched_include_main=_as_bool(item["sched_include_main"]),
            g_norm=float(item["g_norm"]),
            r_norm=float(item["r_norm"]),
            chi_norm=float(item["chi_norm"]),
            p_rx_cap_norm=float(item["p_rx_cap_norm"]),
        )
    missing_labels = REQUIRED_LABELS.difference(rows)
    if missing_labels:
        raise ValueError(f"HFSS scheduling library missing labels: {sorted(missing_labels)}")
    if rows["surface_contact_failure"].sched_include_main:
        raise ValueError("surface_contact_failure must not be included in main scheduling scenarios")
    return HFSSLibrary(source_path=source, rows=rows)


def _with_coefficients(row: HFSSRow, source: HFSSRow | None = None, **overrides: float) -> HFSSRow:
    base = source or row
    return replace(
        row,
        g_norm=float(overrides.get("g_norm", base.g_norm)),
        r_norm=float(overrides.get("r_norm", base.r_norm)),
        chi_norm=float(overrides.get("chi_norm", base.chi_norm)),
        p_rx_cap_norm=float(overrides.get("p_rx_cap_norm", base.p_rx_cap_norm)),
    )


def _average_rows(template: HFSSRow, *rows: HFSSRow) -> HFSSRow:
    if not rows:
        return template
    g = sum(row.g_norm for row in rows) / len(rows)
    r = sum(row.r_norm for row in rows) / len(rows)
    chi = sum(row.chi_norm for row in rows) / len(rows)
    p_cap = min(g / max(chi, 1e-12), 1.0)
    return _with_coefficients(template, g_norm=g, r_norm=r, chi_norm=chi, p_rx_cap_norm=p_cap)


def load_homogenized_library(path: str | Path | None = None) -> HFSSLibrary:
    """Return a decision-library ablation with all labels using reference coefficients."""
    full = load_hfss_library(path)
    reference = full.rows["surface_rest"]
    rows = {label: _with_coefficients(row, reference) for label, row in full.rows.items()}
    return HFSSLibrary(source_path=Path("<homogenized-reference>"), rows=rows)


def load_g_scaled_library(scale: float, path: str | Path | None = None) -> HFSSLibrary:
    """Return a decision library with only g_norm perturbed by a scalar factor."""
    full = load_hfss_library(path)
    rows = {
        label: _with_coefficients(row, g_norm=row.g_norm * float(scale))
        for label, row in full.rows.items()
    }
    return HFSSLibrary(source_path=Path(f"<g-scaled-{float(scale):.3f}>"), rows=rows)


def load_linkbudget_library(path: str | Path | None = None) -> HFSSLibrary:
    """Return a simple depth-based proxy library without HFSS calibration content."""
    full = load_hfss_library(path)
    depth_m = {
        "surface_rest": 0.0,
        "surface_moderate_loose": 0.0,
        "surface_contact_failure": 0.0,
        "surface_sweat": 0.0,
        "implant10_rest": 0.010,
        "implant10_stress": 0.010,
        "implant30_rest": 0.030,
        "implant30_stress": 0.030,
    }
    d0 = 0.005
    rows: dict[str, HFSSRow] = {}
    for label, row in full.rows.items():
        coef = 1.0 / (1.0 + depth_m[label] / d0) ** 2
        if row.kind == "surface" and row.condition in {"moderate_loose", "loose"}:
            coef *= 0.65
        if row.kind == "surface" and row.condition == "sweat":
            coef = min(1.0, coef * 1.25)
        rows[label] = _with_coefficients(
            row,
            g_norm=coef,
            r_norm=coef,
            chi_norm=1.0,
            p_rx_cap_norm=min(coef, 1.0),
        )
    return HFSSLibrary(source_path=Path("<link-budget-only>"), rows=rows)


def load_coarse_2state_library(path: str | Path | None = None) -> HFSSLibrary:
    """Return a fair coarse rest/stress proxy library for granular-library ablation."""
    full = load_hfss_library(path)
    rows = dict(full.rows)
    implant_rest = _average_rows(full.rows["implant10_rest"], full.rows["implant10_rest"], full.rows["implant30_rest"])
    implant_stress = _average_rows(full.rows["implant10_stress"], full.rows["implant10_stress"], full.rows["implant30_stress"])
    surface_stress = full.rows["surface_moderate_loose"]
    for label in ["implant10_rest", "implant30_rest"]:
        rows[label] = _with_coefficients(full.rows[label], implant_rest)
    for label in ["implant10_stress", "implant30_stress"]:
        rows[label] = _with_coefficients(full.rows[label], implant_stress)
    for label in ["surface_moderate_loose", "surface_sweat"]:
        rows[label] = _with_coefficients(full.rows[label], surface_stress)
    return HFSSLibrary(source_path=Path("<coarse-2state>"), rows=rows)
