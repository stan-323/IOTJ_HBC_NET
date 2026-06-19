# IOTJ HBC Surface–Implant Scheduling: Code and HFSS Calibration Package

This repository supports the measurement-anchored, HFSS-derived field-proxy scheduling
experiments and figure generation in the paper. It contains the scheduling/analysis code
**and** the HFSS calibration package (proxy library, raw S-parameters, sampled
receiver-cube field statistics, and material/port tables) needed to inspect and reuse the
scheduler's physical interface.

## Contents

### Code
- `src/hfss_paper_results/`: scheduling simulation, metrics, gates, reporting, and plotting code.
- `scripts/render_main_figures_standard.py`: manuscript figure rendering utilities.
- `scripts/build_iter10_adaptive.py`: state-estimator robustness experiment driver.
- `hfss/scripts/`: HFSS/PyAEDT calibration and post-processing scripts.
- `tests/`: unit tests using synthetic fixtures only.

### HFSS calibration package
- `data/calibrated_library_sched.csv`: scheduler-facing proxy library queried by the online
  LP (corresponds to the proxy-library table in the paper).
- `hfss/torso_main/outputs/`:
  - `torso_scheduler_library.csv`: full proxy library with provenance columns.
  - `material_table.csv`, `setup_table.csv`: tissue/material settings and the torso geometry,
    port, and solver-setup definitions.
  - `case_manifest.csv`: per-case geometry, contact/stress mechanism, and solved frequencies
    for the eight main library states.
  - `sparams_raw.csv` and `t_*.s2p`: raw HFSS S-parameters per state and frequency (Touchstone).
  - `field_proxy_samples.csv` and `field_samples/*.fld`: sampled E-field statistics (Q_p95, etc.)
    over the 5 mm receiver cube used for the field-load coordinate.
  - `torso_frequency_summary.csv`: per-state frequency-response summary.

The sampled-field quantities are scheduler-side load proxies (σ|E|² statistics in a 5 mm
receiver cube). They are **not** SAR-compliance results and do not replace 1g/10g SAR,
Pennes bioheat, patient-specific, or hardware compliance testing. The HFSS model uses
representative (non-patient-specific) tissue parameters.

## Not included
- The full Ansys HFSS/AEDT project file (`hbc_torso_main.aedt`) and solver caches are not
  distributed here; they are available from the authors on reasonable request. The released
  package is sufficient to inspect and reuse the proxy library and to verify the published
  coefficients against the raw S-parameters and field samples. Regenerating or extending the
  full-wave model requires the project file.
- End-to-end figure regeneration additionally uses the per-experiment scheduling output CSVs;
  set `IOTJ_FIGURE_DATA_DIR` to a directory containing those files (kept local).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

HFSS automation (only needed to rebuild a project, not to use the released package)
additionally requires Ansys Electronics Desktop and PyAEDT:

```bash
python -m pip install -r requirements-hfss.txt
```

## Run

The scheduler defaults to the included library at `data/calibrated_library_sched.csv`;
override it with the `HFSS_SCHED_LIBRARY` environment variable if needed.

```bash
python -m hfss_paper_results.run --mode smoke
```

## Test

```bash
python -m pytest
```

The tests do not require private data.
