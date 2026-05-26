from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hfss" / "torso_main" / "scripts" / "postprocess_torso_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("torso_postprocess", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mag2(s21_db: float) -> float:
    return 10.0 ** (s21_db / 10.0)


def add_sparam(rows: list[dict[str, object]], case: str, study: str, rx_type: str, condition: str, x_mm: float, depth_mm: float, freq_mhz: float, s21_db: float) -> None:
    rows.append(
        {
            "label": case,
            "study": study,
            "geometry": "torso",
            "type": rx_type,
            "condition": condition,
            "rx_x_mm": x_mm,
            "depth_mm": depth_mm,
            "rx_z_mm": 150.0 - depth_mm,
            "freq_MHz": freq_mhz,
            "S11_dB": -4.0,
            "S21_dB": s21_db,
            "S12_dB": s21_db,
            "S22_dB": -4.0,
            "path_loss_dB": -s21_db,
            "S21_mag2": mag2(s21_db),
            "S11_phase_deg": 0.0,
            "S21_phase_deg": 0.0,
            "S12_phase_deg": 0.0,
            "S22_phase_deg": 0.0,
            "touchstone_file": f"{case}.s2p",
        }
    )


def add_field(rows: list[dict[str, object]], case: str, study: str, rx_type: str, condition: str, x_mm: float, depth_mm: float, q_p95: float) -> None:
    rows.append(
        {
            "label": case,
            "study": study,
            "geometry": "torso",
            "type": rx_type,
            "condition": condition,
            "depth_mm": depth_mm,
            "rx_x_mm": x_mm,
            "rx_z_mm": 150.0 - depth_mm,
            "freq_MHz": 13.56,
            "sample_count": 125,
            "sigma_S_per_m": 0.7 if rx_type == "surface" else 0.8,
            "Emean_sampled_V_per_m": 8.0,
            "Ep90_sampled_V_per_m": 9.0,
            "Ep95_sampled_V_per_m": 10.0,
            "Ep99_sampled_V_per_m": 11.0,
            "Emax_sampled_V_per_m": 12.0,
            "Q_mean_W_per_m3": q_p95 * 0.8,
            "Q_p90_W_per_m3": q_p95 * 0.9,
            "Q_p95_W_per_m3": q_p95,
            "Q_p99_W_per_m3": q_p95 * 1.1,
            "Q_max_W_per_m3": q_p95 * 1.2,
            "sample_points_file": f"{case}.fld",
        }
    )


def write_raw_outputs(tmp_path: Path) -> None:
    sparams: list[dict[str, object]] = []
    fields: list[dict[str, object]] = []
    main_cases = [
        ("t_surf_rest", "surface", "surface_rest", 50.0, 0.0, -60.0, -64.0, 100.0),
        ("t_surf_sweat", "surface", "surface_sweat", 50.0, 0.0, -57.0, -61.0, 150.0),
        ("t_surf_loose", "surface", "surface_loose", 50.0, 0.0, -66.0, -70.0, 90.0),
        ("t_surf_fail", "surface", "surface_contact_failure", 50.0, 0.0, -96.0, -100.0, 105.0),
        ("t_imp10_rest", "implant", "implant10_rest", 50.0, 10.0, -64.0, -67.0, 110.0),
        ("t_imp10_stress", "implant", "implant10_stress", 50.0, 10.0, -72.0, -76.0, 130.0),
        ("t_imp30_rest", "implant", "implant30_rest", 50.0, 30.0, -65.0, -68.0, 115.0),
        ("t_imp30_stress", "implant", "implant30_stress", 50.0, 30.0, -74.0, -77.0, 140.0),
    ]
    for case, rx_type, condition, x_mm, depth_mm, energy_db, data_db, q_p95 in main_cases:
        add_sparam(sparams, case, "A_main_library", rx_type, condition, x_mm, depth_mm, 13.56, energy_db)
        add_sparam(sparams, case, "A_main_library", rx_type, condition, x_mm, depth_mm, 40.0, data_db)
        add_field(fields, case, "A_main_library", rx_type, condition, x_mm, depth_mm, q_p95)

    for freq_mhz, s21_db in [(1.0, -70.0), (10.0, -62.0), (21.0, -63.0)]:
        add_sparam(sparams, "t_surf_rest", "B_frequency_sweep", "surface", "surface_rest", 50.0, 0.0, freq_mhz, s21_db)

    for case, x_mm, energy_db, data_db, q_p95 in [
        ("t_scan_x050", 50.0, -61.0, -65.0, 200.0),
        ("t_scan_x150", 150.0, -69.0, -73.0, 260.0),
    ]:
        add_sparam(sparams, case, "C_distance_scan", "surface", "surface_rest_distance_scan", x_mm, 0.0, 13.56, energy_db)
        add_sparam(sparams, case, "C_distance_scan", "surface", "surface_rest_distance_scan", x_mm, 0.0, 40.0, data_db)
        add_field(fields, case, "C_distance_scan", "surface", "surface_rest_distance_scan", x_mm, 0.0, q_p95)

    for case, depth_mm, energy_db, data_db, q_p95 in [
        ("t_depth_d010", 10.0, -63.0, -66.0, 210.0),
        ("t_depth_d030", 30.0, -66.0, -69.0, 230.0),
        ("t_depth_d050", 50.0, -68.0, -72.0, 250.0),
    ]:
        add_sparam(sparams, case, "D_depth_scan", "implant", "implant_depth_scan", 50.0, depth_mm, 13.56, energy_db)
        add_sparam(sparams, case, "D_depth_scan", "implant", "implant_depth_scan", 50.0, depth_mm, 40.0, data_db)
        add_field(fields, case, "D_depth_scan", "implant", "implant_depth_scan", 50.0, depth_mm, q_p95)

    write_csv(tmp_path / "sparams_raw.csv", sparams)
    write_csv(tmp_path / "field_proxy_samples.csv", fields)


def test_postprocess_writes_scheduler_library_with_canonical_labels(tmp_path):
    write_raw_outputs(tmp_path)
    mod = load_module()

    mod.write_postprocessed_outputs(tmp_path)

    rows = read_csv(tmp_path / "torso_scheduler_library.csv")
    by_label = {row["label"]: row for row in rows}

    assert {row["case"] for row in rows} == {
        "t_surf_rest",
        "t_surf_sweat",
        "t_surf_loose",
        "t_surf_fail",
        "t_imp10_rest",
        "t_imp10_stress",
        "t_imp30_rest",
        "t_imp30_stress",
    }
    assert set(by_label) == {
        "surface_rest",
        "surface_sweat",
        "surface_moderate_loose",
        "surface_contact_failure",
        "implant10_rest",
        "implant10_stress",
        "implant30_rest",
        "implant30_stress",
    }
    assert by_label["surface_moderate_loose"]["case"] == "t_surf_loose"
    assert by_label["surface_moderate_loose"]["condition"] == "moderate_loose"
    assert by_label["surface_contact_failure"]["case"] == "t_surf_fail"
    assert by_label["surface_contact_failure"]["sched_include_main"] == "False"
    assert math.isclose(float(by_label["surface_sweat"]["g_norm"]), mag2(-57.0) / mag2(-60.0))
    assert math.isclose(float(by_label["surface_sweat"]["g_norm_raw_s21"]), mag2(-57.0) / mag2(-60.0))
    assert math.isclose(float(by_label["surface_sweat"]["r_norm"]), mag2(-61.0) / mag2(-64.0))
    assert math.isclose(float(by_label["surface_sweat"]["Q_p95_norm_5mm"]), 1.5)
    assert math.isclose(float(by_label["surface_sweat"]["chi_norm_raw_q_p95"]), 1.5)
    assert math.isclose(float(by_label["surface_sweat"]["chi_norm"]), math.sqrt(1.5))
    assert math.isclose(float(by_label["surface_sweat"]["p_rx_cap_norm_raw"]), mag2(-57.0) / mag2(-60.0))
    assert float(by_label["surface_sweat"]["p_rx_cap_norm"]) == 1.0
    implant10_stress_raw_g = mag2(-72.0) / mag2(-60.0)
    assert math.isclose(float(by_label["implant10_stress"]["g_norm_raw_s21"]), implant10_stress_raw_g)
    assert math.isclose(float(by_label["implant10_stress"]["g_norm"]), implant10_stress_raw_g * 2.0)
    assert math.isclose(float(by_label["implant10_stress"]["chi_norm_raw_q_p95"]), 1.3)
    assert math.isclose(float(by_label["implant10_stress"]["chi_norm"]), 1.15)
    assert math.isclose(float(by_label["implant10_stress"]["p_rx_cap_norm_raw"]), implant10_stress_raw_g)
    assert math.isclose(float(by_label["implant10_stress"]["p_rx_cap_norm"]), implant10_stress_raw_g * 2.0)
    assert by_label["implant10_stress"]["scheduler_calibration_note"].startswith("implant energy scaled")
    assert math.isclose(float(by_label["surface_contact_failure"]["p_rx_cap_norm_raw"]), mag2(-96.0) / mag2(-60.0))
    assert "SAR" not in rows[0]
    assert "not SAR" in by_label["surface_rest"]["field_proxy_note"]
    assert "sqrt_sampled_Q_p95_norm" in by_label["surface_rest"]["chi_definition"]


def test_read_csv_accepts_utf8_sig_header(tmp_path):
    path = tmp_path / "sparams_raw.csv"
    path.write_text("case,freq_MHz,S21_dB\n" "t_surf_rest,13.56,-60.0\n", encoding="utf-8-sig")
    mod = load_module()

    rows = mod.read_csv(path)

    assert rows[0]["case"] == "t_surf_rest"
    assert mod.case_value(rows[0]) == "t_surf_rest"


def test_postprocess_writes_physical_anchors_and_scan_summaries(tmp_path):
    write_raw_outputs(tmp_path)
    mod = load_module()

    mod.write_postprocessed_outputs(tmp_path, p_tx_mw=2.0, energy_buffer_j=2.0e-5)

    expected_files = {
        "torso_scheduler_library.csv",
        "physical_anchor_table.csv",
        "torso_frequency_summary.csv",
        "torso_distance_summary.csv",
        "torso_depth_summary.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    anchors = {row["case"]: row for row in read_csv(tmp_path / "physical_anchor_table.csv")}
    assert set(anchors) == {"t_scan_x050", "t_scan_x150", "t_depth_d010", "t_depth_d030", "t_depth_d050"}
    assert math.isclose(float(anchors["t_scan_x050"]["P_tx_dBm"]), 10.0 * math.log10(2.0))
    assert math.isclose(float(anchors["t_scan_x050"]["P_rx_W"]), 2.0e-3 * mag2(-61.0))
    assert math.isclose(float(anchors["t_scan_x050"]["reference_Q_p95_W_per_m3"]), 200.0)
    assert "not SAR" in anchors["t_scan_x050"]["field_proxy_note"]

    frequency = read_csv(tmp_path / "torso_frequency_summary.csv")
    surface_freqs = {float(row["freq_MHz"]) for row in frequency if row["case"] == "t_surf_rest"}
    assert surface_freqs == {1.0, 10.0, 13.56, 21.0, 40.0}

    distance = {row["case"]: row for row in read_csv(tmp_path / "torso_distance_summary.csv") if row["freq_MHz"] == "13.56"}
    assert math.isclose(float(distance["t_scan_x050"]["distance_mm"]), 50.0)
    assert math.isclose(float(distance["t_scan_x050"]["S21_mag2_norm_to_x050"]), 1.0)
    assert math.isclose(float(distance["t_scan_x150"]["S21_mag2_norm_to_x050"]), mag2(-69.0) / mag2(-61.0))

    depth = {row["case"]: row for row in read_csv(tmp_path / "torso_depth_summary.csv") if row["freq_MHz"] == "13.56"}
    assert math.isclose(float(depth["t_depth_d010"]["depth_mm"]), 10.0)
    assert math.isclose(float(depth["t_depth_d010"]["S21_mag2_norm_to_d010"]), 1.0)
    assert math.isclose(float(depth["t_depth_d050"]["S21_mag2_norm_to_d010"]), mag2(-68.0) / mag2(-63.0))
