from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SCHED_LIBRARY_ENV = "HFSS_SCHED_LIBRARY"
SCHED_LIBRARY_PATH = WORKSPACE_ROOT / "data" / "calibrated_library_sched.csv"

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


def default_sched_library_path() -> Path:
    configured = os.environ.get(SCHED_LIBRARY_ENV)
    return Path(configured) if configured else SCHED_LIBRARY_PATH


def load_hfss_library(path: str | Path | None = None) -> HFSSLibrary:
    source = Path(path) if path is not None else default_sched_library_path()
    if source.name != "calibrated_library_sched.csv":
        raise ValueError(f"Only calibrated_library_sched.csv is allowed: {source}")
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
