# IOTJ HFSS-Aware Scheduling Code

This repository contains code that supports the paper's HFSS-aware scheduling experiments and figure generation.

Data, simulation outputs, AEDT project files, paper drafts, generated figures, and review materials are intentionally not included. The code expects private/local inputs to be provided through file paths or environment variables.

## Contents

- `src/hfss_paper_results/`: scheduling simulation, metrics, gates, reporting, and plotting code.
- `scripts/render_main_figures_standard.py`: manuscript figure rendering utilities.
- `scripts/build_iter10_adaptive.py`: state-estimator robustness experiment driver.
- `hfss/scripts/`: HFSS/PyAEDT calibration and post-processing scripts.
- `tests/`: unit tests using synthetic fixtures only.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

HFSS automation additionally requires Ansys Electronics Desktop and PyAEDT:

```bash
python -m pip install -r requirements-hfss.txt
```

## Private Inputs

The repository is code-only. To run the full pipeline, provide local data paths:

- `HFSS_SCHED_LIBRARY`: path to `calibrated_library_sched.csv`.
- `IOTJ_FIGURE_DATA_DIR`: directory containing final figure CSV files.
- `IOTJ_ROBUSTNESS_SUMMARY`: path to state estimator robustness summary CSV.
- `IOTJ_ROBUSTNESS_LEADERBOARD`: path to adaptive robustness leaderboard CSV.
- `HBC_HFSS_ROOT`: local HFSS work directory for PyAEDT scripts.

Example:

```bash
set HFSS_SCHED_LIBRARY=D:\private_data\calibrated_library_sched.csv
python -m hfss_paper_results.run --mode smoke
```

## Test

```bash
python -m pytest
```

The tests do not require private data.

## Publication Note

This code release is intended for paper traceability and method inspection. Raw data and generated simulation products are withheld for safety, privacy, and compliance reasons.
