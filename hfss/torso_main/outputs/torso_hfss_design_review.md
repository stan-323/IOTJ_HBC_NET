# Torso HFSS Design Review

This package implements the reviewer-driven chest/torso HFSS experiment plan under this project's `hfss/torso_main` folder.

## Locked Assumptions

- Main scheduler states are clinically renamed around chest-wall surface patches and implanted devices.
- Implant stress is modeled as 1 mm fibrotic encapsulation with sigma=0.20 S/m and epsilon_r=60, without moving the receiver in x.
- The surface-loose state uses a receiver-side 0.5 mm local lift. A hub lift would be a global medium state and would not map cleanly to a per-node scheduling label.
- A 300 mm air margin is retained as a practical near-field margin. At 40 MHz it is about 0.04 lambda in air, not lambda/4.
- Reviewer-analysis rows intentionally contain 55 frequency requests. Solver rows deduplicate overlapping A/B requests to 49 actual case-frequency solves.
- SAR outputs must be reported as post-processed 1g/10g quantities. The manuscript should not claim compliance until the solved fields are post-processed.

## Generated Files

- `case_manifest.csv`: 20 unique solver geometries.
- `analysis_frequency_plan.csv`: 55 reviewer-facing frequency requests.
- `solver_frequency_plan.csv`: 49 deduplicated solver case-frequency rows.
- `material_table.csv` and `setup_table.csv`: Appendix A inputs.
- `physical_anchor_template.csv`: Appendix C fill-in table.
- `sar_chi_correlation_template.csv`: Appendix D fill-in table.
- `lit_cross_validation_template.csv`: Appendix E digitization plan.
