from __future__ import annotations

import csv
import math
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ansys.aedt.core import Hfss

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hbc_hfss_calibration import (  # noqa: E402
    AEDT_VERSION,
    CASES,
    DATA_FREQ_MHZ,
    ENERGY_FREQ_MHZ,
    MATERIALS,
    ROOT,
    Case,
    add_materials,
    create_case_geometry,
    field_max,
    local_material_for_case,
    parse_touchstone,
)


PROJECT_DIR = ROOT / "projects"
REVIEW_DIR = ROOT / "outputs" / "hfss_review"
FIELD_SAMPLE_DIR = REVIEW_DIR / "field_samples"
PROJECT_PATH = PROJECT_DIR / "hbc_hfss_review_experiments.aedt"
SETUP_NAME = "Setup1"
FREQS_MHZ = [ENERGY_FREQ_MHZ, 20.0, DATA_FREQ_MHZ]
LOOSE_SCAN_CASES = [
    ("surface_loose_001um", 0.001),
    ("surface_loose_002um", 0.002),
    ("surface_loose_005um", 0.005),
    ("surface_loose_010um", 0.010),
    ("surface_loose_020um", 0.020),
    ("surface_loose_005", 0.050),
    ("surface_loose_010", 0.100),
    ("surface_loose_020", 0.200),
]


def ensure_dirs() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_review_project() -> None:
    project_root = PROJECT_DIR.resolve()
    for target in [
        PROJECT_PATH,
        PROJECT_PATH.with_suffix(".aedt.lock"),
        PROJECT_PATH.with_suffix(".aedtresults"),
        PROJECT_PATH.with_suffix(".pyaedt"),
    ]:
        try:
            resolved = target.resolve()
        except FileNotFoundError:
            resolved = target.absolute()
        if not str(resolved).startswith(str(project_root)):
            raise RuntimeError(f"Refusing to remove path outside project directory: {resolved}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def clean_review_outputs() -> None:
    review_root = REVIEW_DIR.resolve()
    if REVIEW_DIR.exists():
        for item in REVIEW_DIR.iterdir():
            resolved = item.resolve()
            if not str(resolved).startswith(str(review_root)):
                raise RuntimeError(f"Refusing to remove path outside review directory: {resolved}")
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    FIELD_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def main_case(label: str) -> Case:
    for case in CASES:
        if case.label == label:
            return case
    raise RuntimeError(f"Unknown main case label: {label}")


def loose_case(label: str, gap_mm: float) -> Case:
    base = main_case("surface_rest")
    return replace(
        base,
        name=f"review_{label}",
        label=label,
        condition="loose",
        loose_gap_mm=gap_mm,
    )


def review_cases() -> list[Case]:
    return [
        main_case("surface_rest"),
        *[loose_case(label, gap_mm) for label, gap_mm in LOOSE_SCAN_CASES],
        main_case("implant10_rest"),
        main_case("implant30_rest"),
        main_case("implant30_stress"),
    ]


def create_review_setup(hfss: Hfss, max_delta_s: float = 0.02, maximum_passes: int = 6, save_fields: bool = True) -> None:
    setup = hfss.create_setup(SETUP_NAME)
    setup.props["Frequency"] = f"{DATA_FREQ_MHZ}MHz"
    setup.props["MaximumPasses"] = maximum_passes
    setup.props["MinimumPasses"] = 1
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["MaxDeltaS"] = max_delta_s
    setup.update()
    for freq in FREQS_MHZ:
        hfss.create_single_point_sweep(
            setup=SETUP_NAME,
            unit="MHz",
            freq=freq,
            name=sweep_name(freq),
            save_single_field=save_fields and abs(freq - ENERGY_FREQ_MHZ) < 1e-6,
            save_fields=save_fields and abs(freq - ENERGY_FREQ_MHZ) < 1e-6,
            save_rad_fields=False,
        )
    try:
        hfss.mesh.assign_length_mesh(
            ["hub_sig", "hub_ref", "rx_sig", "rx_ref"],
            maximum_length=0.8,
            inside_selection=True,
            name="mesh_electrodes_0p8mm",
        )
    except Exception:
        pass


def sweep_name(freq_mhz: float) -> str:
    return "SP" + str(freq_mhz).replace(".", "p")


def design_name(case: Case) -> str:
    return "review_" + case.label


def create_observation_box(hfss: Hfss, case: Case, rx_z_mm: float, side_mm: float) -> tuple[str, str]:
    half = side_mm / 2.0
    if case.rx_type == "surface":
        origin = [case.rx_x_mm - half, -half, -side_mm]
    else:
        origin = [case.rx_x_mm - half, -half, rx_z_mm - half]
    obs = hfss.modeler.create_box(origin, [side_mm, side_mm, side_mm], name=f"obs_rx_{int(side_mm)}mm_cube", material="air")
    try:
        obs.model = False
    except Exception:
        pass
    return obs.name, f"{side_mm:g}mm cube origin=({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}) mm"


def build_design(hfss: Hfss, case: Case, local_mesh: bool = False) -> dict[str, Any]:
    add_materials(hfss)
    geom = create_case_geometry(hfss, case)
    obs10_name, obs10_desc = create_observation_box(hfss, case, float(geom["rx_z_mm"]), 10.0)
    geom["obs_10mm_name"] = obs10_name
    geom["obs_10mm_desc"] = obs10_desc
    if local_mesh:
        try:
            hfss.mesh.assign_length_mesh(
                ["rx_sig", "rx_ref"],
                maximum_length=0.35,
                inside_selection=True,
                name="mesh_implant_electrodes_0p35mm",
            )
            hfss.mesh.assign_length_mesh(
                ["muscle_layer"],
                maximum_length=0.8,
                inside_selection=True,
                name="mesh_muscle_0p8mm",
            )
        except Exception as exc:
            geom["local_mesh_warning"] = str(exc)
    create_review_setup(
        hfss,
        max_delta_s=0.01 if local_mesh else 0.02,
        maximum_passes=8 if local_mesh else 6,
        save_fields=True,
    )
    return geom


def build_project() -> tuple[Hfss, dict[str, dict[str, Any]]]:
    ensure_dirs()
    clean_review_project()
    clean_review_outputs()
    cases = review_cases()
    hfss = Hfss(
        project=str(PROJECT_PATH),
        design=design_name(cases[0]),
        solution_type="Modal",
        version=AEDT_VERSION,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
        remove_lock=True,
    )
    geometry: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if index == 0:
            hfss.set_active_design(design_name(case))
        else:
            hfss.insert_design(design_name(case), solution_type="Modal")
            hfss.set_active_design(design_name(case))
        geometry[case.label] = build_design(hfss, case, local_mesh=False)

    local_case = replace(main_case("implant30_stress"), name="review_implant30_stress_localmesh", label="implant30_stress_localmesh")
    hfss.insert_design(design_name(local_case), solution_type="Modal")
    hfss.set_active_design(design_name(local_case))
    geometry[local_case.label] = build_design(hfss, local_case, local_mesh=True)
    hfss.save_project(str(PROJECT_PATH), overwrite=True)
    return hfss, geometry


def export_design_touchstones(hfss: Hfss, case: Case, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hfss.set_active_design(prefix)
    ok = hfss.analyze_setup(SETUP_NAME, cores=4, tasks=1, blocking=True)
    if not ok:
        raise RuntimeError(f"HFSS solve failed for {prefix}")
    for freq in FREQS_MHZ:
        sname = sweep_name(freq)
        out = REVIEW_DIR / f"{prefix}_{sname}.s2p"
        hfss.export_touchstone(setup=SETUP_NAME, sweep=sname, output_file=str(out), renormalization=True, impedance=50)
        parsed = parse_touchstone(out)
        if not parsed:
            raise RuntimeError(f"Touchstone parse failed: {out}")
        row = min(parsed, key=lambda item: abs(float(item["freq_MHz"]) - freq))
        rows.append(
            {
                "case": case.name,
                "label": case.label,
                "type": case.rx_type,
                "condition": case.condition,
                "depth_mm": case.rx_depth_mm,
                "loose_gap_mm": case.loose_gap_mm,
                "freq_MHz": freq,
                "touchstone_file": str(out),
                **row,
            }
        )
    return rows


def sample_points_mm(case: Case, rx_z_mm: float, side_mm: float, n: int = 5) -> list[list[float]]:
    half = side_mm / 2.0
    if case.rx_type == "surface":
        start = [case.rx_x_mm - half, -half, -side_mm]
        stop = [case.rx_x_mm + half, half, 0.0]
    else:
        start = [case.rx_x_mm - half, -half, rx_z_mm - half]
        stop = [case.rx_x_mm + half, half, rx_z_mm + half]
    pts: list[list[float]] = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                frac = [ix / (n - 1), iy / (n - 1), iz / (n - 1)]
                pts.append([start[axis] + frac[axis] * (stop[axis] - start[axis]) for axis in range(3)])
    return pts


def export_sampled_e_stats(hfss: Hfss, case: Case, rx_z_mm: float, side_mm: float, suffix: str) -> dict[str, Any]:
    pts_m = [[coord / 1000.0 for coord in point] for point in sample_points_mm(case, rx_z_mm, side_mm)]
    out = FIELD_SAMPLE_DIR / f"{suffix}_{int(side_mm)}mm.fld"
    hfss.post.export_field_file(
        quantity="E",
        solution=f"{SETUP_NAME} : {sweep_name(ENERGY_FREQ_MHZ)}",
        output_file=str(out),
        sample_points=pts_m,
        intrinsics={"Freq": f"{ENERGY_FREQ_MHZ}MHz", "Phase": "0deg"},
        export_in_si_system=True,
    )
    magnitudes = parse_field_magnitudes(out)
    if not magnitudes:
        raise RuntimeError(f"No sampled field rows parsed from {out}")
    magnitudes_sorted = sorted(magnitudes)
    p95_index = min(len(magnitudes_sorted) - 1, math.ceil(0.95 * len(magnitudes_sorted)) - 1)
    e_max = max(magnitudes)
    e_p95 = magnitudes_sorted[p95_index]
    e_avg = sum(magnitudes) / len(magnitudes)
    sigma = local_material_for_case(case).conductivity
    return {
        "sample_points_file": str(out),
        "sample_count": len(magnitudes),
        "Emax_sampled_V_per_m": e_max,
        "Ep95_sampled_V_per_m": e_p95,
        "Eavg_sampled_V_per_m": e_avg,
        "Q_max": sigma * e_max * e_max,
        "Q_p95": sigma * e_p95 * e_p95,
        "Q_avg": sigma * e_avg * e_avg,
    }


def parse_field_magnitudes(path: Path) -> list[float]:
    mags: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("Complex") or line.startswith("#"):
                continue
            vals = [float(token) for token in line.split()]
            if len(vals) < 9:
                continue
            ex = complex(vals[3], vals[4])
            ey = complex(vals[5], vals[6])
            ez = complex(vals[7], vals[8])
            mags.append(math.sqrt(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2))
    return mags


def scalar_field_qmax(hfss: Hfss, case: Case, obs_name: str, side_mm: float) -> dict[str, Any]:
    emax, source = field_max(hfss, obs_name, f"{SETUP_NAME} : {sweep_name(ENERGY_FREQ_MHZ)}", ENERGY_FREQ_MHZ)
    sigma = local_material_for_case(case).conductivity
    return {
        f"Q{int(side_mm)}_calculator_max": sigma * emax * emax if emax is not None else "",
        f"E{int(side_mm)}_calculator_max": emax if emax is not None else "",
        f"calculator_source_{int(side_mm)}mm": source,
    }


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baseline_mag2(label: str, freq_mhz: float) -> float:
    rows = read_existing_rows(ROOT / "outputs" / "sparams_raw.csv")
    candidates = [row for row in rows if row["label"] == label]
    row = min(candidates, key=lambda item: abs(float(item["freq_MHz"]) - freq_mhz))
    return float(row["S21_mag2"])


def existing_library_row(label: str) -> dict[str, str]:
    rows = read_existing_rows(ROOT / "outputs" / "calibrated_library_full.csv")
    for row in rows:
        if row["label"] == label:
            return row
    raise RuntimeError(f"Missing existing library row: {label}")


def normalize_rows(rows: list[dict[str, Any]], baseline_energy: float, baseline_data: float) -> None:
    for row in rows:
        freq = float(row["freq_MHz"])
        if abs(freq - ENERGY_FREQ_MHZ) < 1e-6:
            row["g_norm"] = float(row["S21_mag2"]) / baseline_energy
        elif abs(freq - DATA_FREQ_MHZ) < 1e-6:
            row["r_norm"] = float(row["S21_mag2"]) / baseline_data


def pair_freq_rows(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matches = [row for row in rows if row["label"] == label]
    energy = min(matches, key=lambda item: abs(float(item["freq_MHz"]) - ENERGY_FREQ_MHZ))
    aux = min(matches, key=lambda item: abs(float(item["freq_MHz"]) - 20.0))
    data = min(matches, key=lambda item: abs(float(item["freq_MHz"]) - DATA_FREQ_MHZ))
    return energy, aux, data


def select_moderate(loose_summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in loose_summary_rows if row["label"].startswith("surface_loose_")]
    passing = [
        row
        for row in candidates
        if 0.1 < float(row["g_norm"]) < 0.8 and 0.05 < float(row["r_norm"]) < 0.8
    ]
    if passing:
        chosen = sorted(passing, key=lambda item: float(item["loose_gap_mm"]))[0]
        chosen["selection_status"] = "selected_in_target_band"
        return chosen

    def penalty(row: dict[str, Any]) -> float:
        g = float(row["g_norm"])
        r = float(row["r_norm"])
        gp = 0.0 if 0.1 <= g <= 0.8 else min(abs(g - 0.1), abs(g - 0.8))
        rp = 0.0 if 0.05 <= r <= 0.8 else min(abs(r - 0.05), abs(r - 0.8))
        return gp * gp + rp * rp

    chosen = sorted(candidates, key=penalty)[0]
    chosen["selection_status"] = "no_candidate_in_target_band_best_available"
    return chosen


def loose_candidate_classification(row: dict[str, Any]) -> str:
    g = float(row["g_norm"])
    r = float(row["r_norm"])
    if g < 0.01 or r < 0.01:
        return "contact_failure_candidate"
    if g < 0.1 or r < 0.05:
        return "severe_loose_candidate"
    if g > 0.8 or r > 0.8:
        return "mild_contact_change"
    return "loose_scan_candidate"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_review(hfss: Hfss, geometry: dict[str, dict[str, Any]]) -> None:
    base_energy = baseline_mag2("surface_rest", ENERGY_FREQ_MHZ)
    base_data = baseline_mag2("surface_rest", DATA_FREQ_MHZ)
    solved_rows: list[dict[str, Any]] = []
    field_stats_by_label: dict[str, dict[float, dict[str, Any]]] = {}

    for case in review_cases():
        prefix = design_name(case)
        print(f"Solving review design {prefix}", flush=True)
        rows = export_design_touchstones(hfss, case, prefix)
        normalize_rows(rows, base_energy, base_data)
        solved_rows.extend(rows)
        hfss.set_active_design(prefix)
        field_stats_by_label[case.label] = {}
        for side_mm, obs_name in [(5.0, geometry[case.label]["obs_name"]), (10.0, geometry[case.label]["obs_10mm_name"])]:
            field_stats_by_label[case.label][side_mm] = {
                **scalar_field_qmax(hfss, case, str(obs_name), side_mm),
                **export_sampled_e_stats(hfss, case, float(geometry[case.label]["rx_z_mm"]), side_mm, case.label),
            }

    loose_rows = summarize_loose_scan(solved_rows)
    contact_failure = existing_library_row("surface_loose")
    loose_rows.append(
        {
            "label": "surface_contact_failure",
            "source_label": "surface_loose",
            "loose_gap_mm": 0.50,
            "s21_energy_db": contact_failure["s21_energy_db"],
            "g_norm": contact_failure["g_norm"],
            "s21_data_db": contact_failure["s21_data_db"],
            "r_norm": contact_failure["r_norm"],
            "classification": "contact_failure",
            "selection_status": "",
        }
    )
    moderate = select_moderate(loose_rows)
    for row in loose_rows:
        if row["label"] == moderate["label"]:
            row["selection_status"] = moderate["selection_status"]
            row["classification"] = (
                "surface_moderate_loose"
                if moderate["selection_status"] == "selected_in_target_band"
                else "best_out_of_band_candidate"
            )
        elif row["label"] != "surface_contact_failure":
            row.setdefault("selection_status", "")
            row["classification"] = loose_candidate_classification(row)
    write_csv(
        REVIEW_DIR / "surface_loose_scan.csv",
        ["label", "source_label", "loose_gap_mm", "s21_energy_db", "g_norm", "s21_data_db", "r_norm", "classification", "selection_status"],
        loose_rows,
    )

    write_field_stability(field_stats_by_label, moderate["label"])
    run_local_mesh_case(hfss)
    write_aux_ordering(solved_rows)
    write_review_library(loose_rows, moderate["label"])
    write_summary(loose_rows, moderate["label"])


def summarize_loose_scan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, _gap_mm in LOOSE_SCAN_CASES:
        energy, _, data = pair_freq_rows(rows, label)
        out.append(
            {
                "label": label,
                "source_label": label,
                "loose_gap_mm": energy["loose_gap_mm"],
                "s21_energy_db": energy["S21_dB"],
                "g_norm": energy["g_norm"],
                "s21_data_db": data["S21_dB"],
                "r_norm": data["r_norm"],
                "classification": "loose_scan_candidate",
                "selection_status": "",
            }
        )
    return out


def normalize_field_stability(rows: list[dict[str, Any]]) -> None:
    for metric in ["Q_max", "Q_p95", "Q_avg", "Q5_calculator_max", "Q10_calculator_max"]:
        baseline = next((row for row in rows if row["label"] == "surface_rest" and row["cube_mm"] == 5.0), None)
        if not baseline:
            continue
        try:
            base = float(baseline[metric])
        except (KeyError, TypeError, ValueError):
            continue
        for row in rows:
            try:
                row[f"{metric}_norm_to_surface_rest_5mm"] = float(row[metric]) / base if base > 0 else ""
            except (KeyError, TypeError, ValueError):
                row[f"{metric}_norm_to_surface_rest_5mm"] = ""


def write_field_stability(field_stats: dict[str, dict[float, dict[str, Any]]], moderate_label: str) -> None:
    labels = ["surface_rest", moderate_label, "implant30_stress"]
    selection = read_existing_rows(REVIEW_DIR / "surface_loose_scan.csv")
    selected = next((row for row in selection if row["label"] == moderate_label), {})
    selected_display_label = (
        "surface_moderate_loose"
        if selected.get("selection_status") == "selected_in_target_band"
        else "surface_best_loose_candidate"
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        for side in [5.0, 10.0]:
            stats = field_stats[label][side]
            rows.append(
                {
                    "label": selected_display_label if label == moderate_label else label,
                    "source_label": label,
                    "cube_mm": side,
                    "sample_count": stats["sample_count"],
                    "Emax_sampled_V_per_m": stats["Emax_sampled_V_per_m"],
                    "Ep95_sampled_V_per_m": stats["Ep95_sampled_V_per_m"],
                    "Eavg_sampled_V_per_m": stats["Eavg_sampled_V_per_m"],
                    "Q_max": stats["Q_max"],
                    "Q_p95": stats["Q_p95"],
                    "Q_avg": stats["Q_avg"],
                    "Q5_calculator_max": stats.get("Q5_calculator_max", ""),
                    "Q10_calculator_max": stats.get("Q10_calculator_max", ""),
                    "sample_points_file": stats["sample_points_file"],
                }
            )
    normalize_field_stability(rows)
    write_csv(
        REVIEW_DIR / "field_proxy_stability.csv",
        [
            "label",
            "source_label",
            "cube_mm",
            "sample_count",
            "Emax_sampled_V_per_m",
            "Ep95_sampled_V_per_m",
            "Eavg_sampled_V_per_m",
            "Q_max",
            "Q_p95",
            "Q_avg",
            "Q_max_norm_to_surface_rest_5mm",
            "Q_p95_norm_to_surface_rest_5mm",
            "Q_avg_norm_to_surface_rest_5mm",
            "Q5_calculator_max",
            "Q10_calculator_max",
            "sample_points_file",
        ],
        rows,
    )


def run_local_mesh_case(hfss: Hfss) -> None:
    case = replace(main_case("implant30_stress"), name="review_implant30_stress_localmesh", label="implant30_stress_localmesh")
    prefix = design_name(case)
    print(f"Solving local mesh design {prefix}", flush=True)
    rows = export_design_touchstones(hfss, case, prefix)
    baseline_rows = read_existing_rows(ROOT / "outputs" / "sparams_raw.csv")
    output_rows: list[dict[str, Any]] = []
    for freq in [ENERGY_FREQ_MHZ, DATA_FREQ_MHZ]:
        old = min([row for row in baseline_rows if row["label"] == "implant30_stress"], key=lambda item: abs(float(item["freq_MHz"]) - freq))
        new = min(rows, key=lambda item: abs(float(item["freq_MHz"]) - freq))
        diff = float(new["S21_dB"]) - float(old["S21_dB"])
        output_rows.append(
            {
                "label": "implant30_stress",
                "freq_MHz": freq,
                "baseline_S21_dB": old["S21_dB"],
                "local_mesh_S21_dB": new["S21_dB"],
                "diff_dB": diff,
                "mesh_status": "ideal" if abs(diff) < 0.5 else "acceptable_caution" if abs(diff) < 1.0 else "needs_refinement",
                "local_mesh_setting": "MaxDeltaS=0.01, MaximumPasses=8, rx electrodes 0.35 mm, muscle 0.8 mm",
                "touchstone_file": new["touchstone_file"],
            }
        )
    write_csv(
        REVIEW_DIR / "implant30_stress_local_mesh.csv",
        ["label", "freq_MHz", "baseline_S21_dB", "local_mesh_S21_dB", "diff_dB", "mesh_status", "local_mesh_setting", "touchstone_file"],
        output_rows,
    )


def write_aux_ordering(rows: list[dict[str, Any]]) -> None:
    labels = ["surface_rest", "implant10_rest", "implant30_rest", "implant30_stress"]
    aux_rows: list[dict[str, Any]] = []
    base = None
    for label in labels:
        _, aux, _ = pair_freq_rows(rows, label)
        if label == "surface_rest":
            base = float(aux["S21_mag2"])
        aux_rows.append(
            {
                "label": label,
                "freq_MHz": 20.0,
                "S21_dB": aux["S21_dB"],
                "path_loss_dB": aux["path_loss_dB"],
                "relative_mag2_norm": float(aux["S21_mag2"]) / base if base else "",
                "ordering_rank_by_S21": "",
                "touchstone_file": aux["touchstone_file"],
            }
        )
    ranked = sorted(aux_rows, key=lambda item: float(item["S21_dB"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["ordering_rank_by_S21"] = rank
    expected = labels
    observed = [row["label"] for row in ranked]
    for row in aux_rows:
        row["ordering_expected"] = " > ".join(expected)
        row["ordering_observed"] = " > ".join(observed)
        row["ordering_stable"] = observed == expected
    write_csv(
        REVIEW_DIR / "aux_frequency_ordering.csv",
        [
            "label",
            "freq_MHz",
            "S21_dB",
            "path_loss_dB",
            "relative_mag2_norm",
            "ordering_rank_by_S21",
            "ordering_expected",
            "ordering_observed",
            "ordering_stable",
            "touchstone_file",
        ],
        aux_rows,
    )


def selected_display_label(selection_status: str) -> str:
    return "surface_moderate_loose" if selection_status == "selected_in_target_band" else "surface_best_loose_candidate"


def field_stability_row(label: str, cube_mm: float = 5.0) -> dict[str, str] | None:
    path = REVIEW_DIR / "field_proxy_stability.csv"
    if not path.exists():
        return None
    for row in read_existing_rows(path):
        if row["label"] == label and abs(float(row["cube_mm"]) - cube_mm) < 1e-9:
            return row
    return None


def blank_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def write_review_library(loose_rows: list[dict[str, Any]], moderate_label: str) -> None:
    main_rows = read_existing_rows(ROOT / "outputs" / "calibrated_library_full.csv")
    rows: list[dict[str, Any]] = []
    for row in main_rows:
        if row["label"] == "surface_loose":
            new = dict(row)
            new["label"] = "surface_contact_failure"
            new["review_note"] = "renamed from previous surface_loose because 0.5 mm air gap is an extreme contact failure"
        else:
            new = dict(row)
            new["review_note"] = "carried from 7-case calibration library"
        new["Q_rx_max_norm"] = new.get("chi_norm", "")
        new["Q_rx_sampled_max_norm"] = ""
        new["Q_rx_sampled_p95_norm"] = ""
        new["Q_rx_sampled_avg_norm"] = ""
        rows.append(new)
    moderate = next(row for row in loose_rows if row["label"] == moderate_label)
    rest = existing_library_row("surface_rest")
    selection_status = str(moderate.get("selection_status", ""))
    display_label = selected_display_label(selection_status)
    rest_field = field_stability_row("surface_rest")
    selected_field = field_stability_row(display_label)
    q_calc = blank_float(selected_field.get("Q5_calculator_max") if selected_field else None)
    q_base_calc = blank_float(rest_field.get("Q5_calculator_max") if rest_field else rest.get("Q_rx_max"))
    baseline_energy = baseline_mag2("surface_rest", ENERGY_FREQ_MHZ)
    g_raw = baseline_energy * float(moderate["g_norm"])
    base_eta = (q_base_calc / baseline_energy) if q_base_calc is not None and baseline_energy > 0 else None
    eta = (q_calc / g_raw) if q_calc is not None and g_raw > 0 else None
    chi = (q_calc / q_base_calc) if q_calc is not None and q_base_calc is not None and q_base_calc > 0 else ""
    p_cap = (base_eta / eta) if eta is not None and eta > 0 and base_eta is not None else ""
    mod = dict(rest)
    mod.update(
        {
            "case": "review_" + moderate_label,
            "label": display_label,
            "condition": "moderate_loose" if selection_status == "selected_in_target_band" else "loose_candidate",
            "s21_energy_db": moderate["s21_energy_db"],
            "path_loss_energy_db": -float(moderate["s21_energy_db"]),
            "g_norm": moderate["g_norm"],
            "s21_data_db": moderate["s21_data_db"],
            "path_loss_data_db": -float(moderate["s21_data_db"]),
            "r_norm": moderate["r_norm"],
            "Q_rx_max": q_calc if q_calc is not None else "",
            "chi_norm": chi,
            "eta_field_over_g": eta if eta is not None else "",
            "p_rx_cap_norm": p_cap,
            "Q_rx_max_norm": chi,
            "Q_rx_sampled_max_norm": selected_field.get("Q_max_norm_to_surface_rest_5mm", "") if selected_field else "",
            "Q_rx_sampled_p95_norm": selected_field.get("Q_p95_norm_to_surface_rest_5mm", "") if selected_field else "",
            "Q_rx_sampled_avg_norm": selected_field.get("Q_avg_norm_to_surface_rest_5mm", "") if selected_field else "",
            "review_note": f"selected from {moderate_label}, gap={moderate['loose_gap_mm']} mm, status={selection_status}",
        }
    )
    rows.append(mod)
    fieldnames = list(rows[0].keys())
    for extra in ["review_note", "Q_rx_max_norm", "Q_rx_sampled_max_norm", "Q_rx_sampled_p95_norm", "Q_rx_sampled_avg_norm"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv(REVIEW_DIR / "calibrated_library_review.csv", fieldnames, rows)


def write_summary(loose_rows: list[dict[str, Any]], moderate_label: str) -> None:
    moderate = next(row for row in loose_rows if row["label"] == moderate_label)
    aux = read_existing_rows(REVIEW_DIR / "aux_frequency_ordering.csv")
    local_mesh = read_existing_rows(REVIEW_DIR / "implant30_stress_local_mesh.csv")
    field_rows = read_existing_rows(REVIEW_DIR / "field_proxy_stability.csv")
    selection_status = str(moderate.get("selection_status", ""))
    is_moderate = selection_status == "selected_in_target_band"
    selection_title = (
        f"Selected `{moderate_label}` as `surface_moderate_loose`."
        if is_moderate
        else f"No target-band moderate loose was found; `{moderate_label}` is the best out-of-band candidate."
    )
    lines = [
        "# HFSS Review Experiments Summary",
        "",
        "Scope: minimum HFSS follow-up checks only; this does not expand to scheduling simulation.",
        "",
        f"Project: `{PROJECT_PATH}`",
        "",
        "## Moderate Loose Selection",
        "",
        selection_title,
        f"Gap: `{moderate['loose_gap_mm']}` mm; `g_norm={float(moderate['g_norm']):.4g}`, `r_norm={float(moderate['r_norm']):.4g}`.",
        f"Selection status: `{selection_status}`.",
        "The prior 0.5 mm loose case is relabeled as `surface_contact_failure`.",
        "The originally requested 0.05/0.10/0.20 mm air gaps are classified as contact-failure candidates when their coupling falls below the loose range.",
        "",
        "## Field Proxy Stability",
        "",
        "| label | cube mm | Q_max norm | Q_p95 norm | Q_avg norm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in field_rows:
        lines.append(
            f"| {row['label']} | {float(row['cube_mm']):.0f} | "
            f"{float(row['Q_max_norm_to_surface_rest_5mm']):.3g} | "
            f"{float(row['Q_p95_norm_to_surface_rest_5mm']):.3g} | "
            f"{float(row['Q_avg_norm_to_surface_rest_5mm']):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Local Mesh",
            "",
            "| freq MHz | diff dB | status |",
            "| ---: | ---: | --- |",
        ]
    )
    for row in local_mesh:
        lines.append(f"| {float(row['freq_MHz']):.2f} | {float(row['diff_dB']):.3f} | {row['mesh_status']} |")
    lines.extend(
        [
            "",
            "## 20 MHz Ordering",
            "",
            f"Observed: `{aux[0]['ordering_observed']}`",
            f"Stable against target: `{aux[0]['ordering_stable']}`",
            "",
            "## Files",
            "",
            "- `surface_loose_scan.csv`",
            "- `field_proxy_stability.csv`",
            "- `implant30_stress_local_mesh.csv`",
            "- `aux_frequency_ordering.csv`",
            "- `calibrated_library_review.csv`",
        ]
    )
    (REVIEW_DIR / "hfss_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REVIEW_DIR / "review_run_summary.txt").write_text(
        "\n".join(
            [
                "status=completed",
                f"project={PROJECT_PATH}",
                f"selected_moderate_loose={moderate_label}",
                f"selected_gap_mm={moderate['loose_gap_mm']}",
                f"selected_status={selection_status}",
                f"aux_ordering_stable={aux[0]['ordering_stable']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    hfss = None
    try:
        hfss, geometry = build_project()
        run_review(hfss, geometry)
        hfss.save_project(str(PROJECT_PATH), overwrite=True)
        return 0
    except Exception:
        ensure_dirs()
        (REVIEW_DIR / "review_run_summary.txt").write_text("status=failed\n" + traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if hfss:
            hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
