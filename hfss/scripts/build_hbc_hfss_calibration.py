from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ansys.aedt.core import Hfss


ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
PROJECT_DIR = ROOT / "projects"
OUT_DIR = ROOT / "outputs"
FIELD_DIR = OUT_DIR / "field_samples"
PROJECT_PATH = PROJECT_DIR / "hbc_hfss_calibration_minimal.aedt"
AEDT_VERSION = "2024.1"
SETUP_NAME = "Setup1"
SWEEP_NAME = "CalFreqs"
ENERGY_FREQ_MHZ = 13.56
DATA_FREQ_MHZ = 40.0
FIELD_FREQ_MHZ = ENERGY_FREQ_MHZ


@dataclass(frozen=True)
class TissueMaterial:
    permittivity: float
    conductivity: float
    density: float
    note: str


MATERIALS = {
    "skin_rest": TissueMaterial(120.0, 0.70, 1109.0, "Representative constant skin proxy near 13.56-40 MHz"),
    "skin_sweat": TissueMaterial(120.0, 2.10, 1109.0, "Sweat proxy: skin conductivity scaled 3x"),
    "fat": TissueMaterial(18.0, 0.05, 911.0, "Representative constant fat proxy near 13.56-40 MHz"),
    "muscle": TissueMaterial(95.0, 0.80, 1090.0, "Representative constant muscle proxy near 13.56-40 MHz"),
}


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    rx_type: str
    condition: str
    rx_depth_mm: float
    rx_x_mm: float
    rx_pair_gap_mm: float
    rx_pad_x_mm: float
    rx_pad_y_mm: float
    rx_pad_z_mm: float
    loose_gap_mm: float = 0.0


CASES = [
    Case(
        name="case01_surface_rest",
        label="surface_rest",
        rx_type="surface",
        condition="rest",
        rx_depth_mm=0.0,
        rx_x_mm=50.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
    ),
    Case(
        name="case02_surface_loose",
        label="surface_loose",
        rx_type="surface",
        condition="loose",
        rx_depth_mm=0.0,
        rx_x_mm=50.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        loose_gap_mm=0.5,
    ),
    Case(
        name="case03_surface_sweat",
        label="surface_sweat",
        rx_type="surface",
        condition="sweat",
        rx_depth_mm=0.0,
        rx_x_mm=50.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
    ),
    Case(
        name="case04_implant10_rest",
        label="implant10_rest",
        rx_type="implant",
        condition="rest",
        rx_depth_mm=10.0,
        rx_x_mm=50.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
    ),
    Case(
        name="case05_implant10_stress",
        label="implant10_stress",
        rx_type="implant",
        condition="stress",
        rx_depth_mm=10.0,
        rx_x_mm=60.0,
        rx_pair_gap_mm=6.0,
        rx_pad_x_mm=3.0,
        rx_pad_y_mm=1.5,
        rx_pad_z_mm=1.0,
    ),
    Case(
        name="case06_implant30_rest",
        label="implant30_rest",
        rx_type="implant",
        condition="rest",
        rx_depth_mm=30.0,
        rx_x_mm=50.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
    ),
    Case(
        name="case07_implant30_stress",
        label="implant30_stress",
        rx_type="implant",
        condition="stress",
        rx_depth_mm=30.0,
        rx_x_mm=60.0,
        rx_pair_gap_mm=6.0,
        rx_pad_x_mm=3.0,
        rx_pad_y_mm=1.5,
        rx_pad_z_mm=1.0,
    ),
]


def ensure_dirs() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)


def clean_project() -> None:
    resolved_project_dir = PROJECT_DIR.resolve()
    for target in [
        PROJECT_PATH,
        PROJECT_PATH.with_suffix(".aedt.lock"),
        PROJECT_PATH.with_name(PROJECT_PATH.name + "results"),
        PROJECT_PATH.with_suffix(".aedtresults"),
        PROJECT_PATH.with_suffix(".pyaedt"),
    ]:
        try:
            resolved_target = target.resolve()
        except FileNotFoundError:
            resolved_target = target.absolute()
        if not str(resolved_target).startswith(str(resolved_project_dir)):
            raise RuntimeError(f"Refusing to remove path outside project directory: {resolved_target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def safe_unlink_generated_outputs() -> None:
    for pattern in [
        "case*.s2p",
        "mesh_*.s2p",
        "sparams_raw.csv",
        "field_proxy.csv",
        "calibrated_library.csv",
        "calibrated_library_full.csv",
        "updated_simulation_results.csv",
        "mesh_convergence.csv",
        "mesh_convergence_note.md",
        "hfss_calibration_summary.md",
        "run_summary.txt",
        "case_manifest.csv",
        "hfss_setup_table.csv",
        "tissue_materials.csv",
    ]:
        for path in OUT_DIR.glob(pattern):
            resolved = path.resolve()
            if not str(resolved).startswith(str(OUT_DIR.resolve())):
                raise RuntimeError(f"Refusing to remove path outside output directory: {resolved}")
            if path.is_file():
                path.unlink()


def safe_add_material(hfss: Hfss, name: str, material: TissueMaterial) -> None:
    material_names = {str(key).lower() for key in hfss.materials.material_keys}
    if name.lower() in material_names:
        mat = hfss.materials[name]
    else:
        mat = hfss.materials.add_material(name)
    mat.permittivity = material.permittivity
    mat.conductivity = material.conductivity
    try:
        mat.mass_density = material.density
    except Exception:
        pass


def add_materials(hfss: Hfss) -> None:
    safe_add_material(hfss, "hbc_skin_proxy", MATERIALS["skin_rest"])
    safe_add_material(hfss, "hbc_skin_sweat_proxy", MATERIALS["skin_sweat"])
    safe_add_material(hfss, "hbc_fat_proxy", MATERIALS["fat"])
    safe_add_material(hfss, "hbc_muscle_proxy", MATERIALS["muscle"])


def set_visual(obj: Any, color: tuple[int, int, int], transparency: float) -> None:
    try:
        obj.color = color
    except Exception:
        pass
    try:
        obj.transparency = transparency
    except Exception:
        pass


def box_centered_xy(
    hfss: Hfss,
    name: str,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    material: str,
):
    return hfss.modeler.create_box(
        [cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0],
        [sx, sy, sz],
        name=name,
        material=material,
    )


def skin_material_for_case(case: Case) -> str:
    if case.condition == "sweat":
        return "hbc_skin_sweat_proxy"
    return "hbc_skin_proxy"


def local_material_for_case(case: Case) -> TissueMaterial:
    if case.rx_type == "surface":
        return MATERIALS["skin_sweat"] if case.condition == "sweat" else MATERIALS["skin_rest"]
    return MATERIALS["muscle"]


def create_tissue_stack(hfss: Hfss, case: Case) -> None:
    hfss.modeler.model_units = "mm"

    air = hfss.modeler.create_box([-150, -125, -150], [300, 250, 240], name="air_region", material="air")
    set_visual(air, (180, 220, 255), 0.9)
    hfss.assign_radiation_boundary_to_objects(air.name, name="Radiation_Air")

    skin = hfss.modeler.create_box([-60, -50, -2], [120, 100, 2], name="skin_layer", material=skin_material_for_case(case))
    fat = hfss.modeler.create_box([-60, -50, -10], [120, 100, 8], name="fat_layer", material="hbc_fat_proxy")
    muscle = hfss.modeler.create_box([-60, -50, -60], [120, 100, 50], name="muscle_layer", material="hbc_muscle_proxy")
    set_visual(skin, (230, 160, 150) if case.condition != "sweat" else (125, 185, 255), 0.55)
    set_visual(fat, (240, 215, 120), 0.65)
    set_visual(muscle, (190, 80, 85), 0.65)


def create_electrode_pair(
    hfss: Hfss,
    prefix: str,
    cx: float,
    cz: float,
    pad_x: float,
    pad_y: float,
    pad_z: float,
    gap_y: float,
    port_name: str,
) -> tuple[str, str]:
    center_offset = gap_y / 2.0 + pad_y / 2.0
    sig = box_centered_xy(hfss, f"{prefix}_sig", cx, center_offset, cz, pad_x, pad_y, pad_z, "copper")
    ref = box_centered_xy(hfss, f"{prefix}_ref", cx, -center_offset, cz, pad_x, pad_y, pad_z, "copper")
    set_visual(sig, (255, 190, 60), 0.0)
    set_visual(ref, (255, 190, 60), 0.0)
    sheet = hfss.modeler.create_rectangle(
        "YZ",
        [cx, -gap_y / 2.0, cz - pad_z / 2.0],
        [gap_y, pad_z],
        name=f"{prefix}_port_sheet",
    )
    set_visual(sheet, (90, 180, 255), 0.35)
    hfss.lumped_port(
        assignment=sheet.name,
        create_port_sheet=False,
        integration_line=[[cx, -gap_y / 2.0, cz], [cx, gap_y / 2.0, cz]],
        impedance=50,
        name=port_name,
        renormalize=True,
        deembed=False,
    )
    return sig.name, ref.name


def receiver_z_for_case(case: Case) -> float:
    if case.rx_type == "surface":
        return 0.1 + case.loose_gap_mm
    rx_z = -case.rx_depth_mm
    muscle_top_z = -10.0
    if rx_z + case.rx_pad_z_mm / 2.0 >= muscle_top_z:
        rx_z = muscle_top_z - case.rx_pad_z_mm / 2.0 - 0.05
    return rx_z


def create_observation_box(hfss: Hfss, case: Case, rx_z: float) -> tuple[str, str]:
    if case.rx_type == "surface":
        origin = [case.rx_x_mm - 2.5, -2.5, -5.0]
        size = [5.0, 5.0, 5.0]
    else:
        origin = [case.rx_x_mm - 2.5, -2.5, rx_z - 2.5]
        size = [5.0, 5.0, 5.0]
    obs = hfss.modeler.create_box(origin, size, name="obs_rx_5mm_cube", material="air")
    try:
        obs.model = False
    except Exception:
        pass
    set_visual(obs, (70, 140, 250), 0.82)
    desc = f"5mm x 5mm x 5mm cube, origin=({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f}) mm"
    return obs.name, desc


def create_case_geometry(hfss: Hfss, case: Case) -> dict[str, Any]:
    create_tissue_stack(hfss, case)
    create_electrode_pair(
        hfss,
        prefix="hub",
        cx=0.0,
        cz=0.1,
        pad_x=10.0,
        pad_y=6.0,
        pad_z=0.2,
        gap_y=8.0,
        port_name="HubPort",
    )
    rx_z = receiver_z_for_case(case)
    rx_sig, rx_ref = create_electrode_pair(
        hfss,
        prefix="rx",
        cx=case.rx_x_mm,
        cz=rx_z,
        pad_x=case.rx_pad_x_mm,
        pad_y=case.rx_pad_y_mm,
        pad_z=case.rx_pad_z_mm,
        gap_y=case.rx_pair_gap_mm,
        port_name="NodePort",
    )
    if case.rx_type == "implant":
        hfss.modeler.subtract(["muscle_layer"], [rx_sig, rx_ref], keep_originals=True)
    obs_name, obs_desc = create_observation_box(hfss, case, rx_z)
    return {"rx_z_mm": rx_z, "obs_name": obs_name, "observation_volume": obs_desc}


def create_setup(
    hfss: Hfss,
    max_delta_s: float = 0.02,
    maximum_passes: int = 6,
    save_fields: bool = True,
) -> None:
    setup = hfss.create_setup(SETUP_NAME)
    setup.props["Frequency"] = f"{DATA_FREQ_MHZ}MHz"
    setup.props["MaximumPasses"] = maximum_passes
    setup.props["MinimumPasses"] = 1
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["MaxDeltaS"] = max_delta_s
    setup.update()
    hfss.create_linear_count_sweep(
        setup=SETUP_NAME,
        unit="MHz",
        start_frequency=ENERGY_FREQ_MHZ,
        stop_frequency=DATA_FREQ_MHZ,
        num_of_freq_points=2,
        name=SWEEP_NAME,
        save_fields=save_fields,
        save_rad_fields=False,
        sweep_type="Discrete",
    )
    try:
        hfss.mesh.assign_length_mesh(
            ["hub_sig", "hub_ref", "rx_sig", "rx_ref"],
            maximum_length=1.0,
            inside_selection=True,
            name="mesh_electrodes_1mm",
        )
    except Exception:
        pass


def build_project(non_graphical: bool, overwrite: bool) -> tuple[Hfss, dict[str, dict[str, Any]]]:
    ensure_dirs()
    if overwrite:
        clean_project()
        safe_unlink_generated_outputs()

    hfss = Hfss(
        project=str(PROJECT_PATH),
        design=CASES[0].name,
        solution_type="Modal",
        version=AEDT_VERSION,
        non_graphical=non_graphical,
        new_desktop=True,
        close_on_exit=False,
        remove_lock=True,
    )
    add_materials(hfss)

    geometry: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(CASES):
        if index == 0:
            hfss.set_active_design(case.name)
        else:
            hfss.insert_design(case.name, solution_type="Modal")
            hfss.set_active_design(case.name)
        add_materials(hfss)
        geometry[case.label] = create_case_geometry(hfss, case)
        create_setup(hfss)

    hfss.save_project(str(PROJECT_PATH), overwrite=True)
    return hfss, geometry


def pair_to_db(a: float, b: float, data_format: str) -> tuple[float, float]:
    if data_format == "ri":
        magnitude = math.hypot(a, b)
        phase_deg = math.degrees(math.atan2(b, a))
    elif data_format == "ma":
        magnitude = abs(a)
        phase_deg = b
    else:
        return a, b
    return 20.0 * math.log10(max(magnitude, 1e-300)), phase_deg


def parse_touchstone(path: Path) -> list[dict[str, float]]:
    unit_scale = {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3}
    freq_scale = 1.0
    data_format = "ri"
    rows: list[dict[str, float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line[1:].strip().lower().split()
                if parts:
                    freq_scale = unit_scale.get(parts[0], 1.0)
                if "ri" in parts:
                    data_format = "ri"
                elif "ma" in parts:
                    data_format = "ma"
                elif "db" in parts:
                    data_format = "db"
                continue
            values = [float(token) for token in line.split()]
            if len(values) < 9:
                continue
            freq_mhz = values[0] * freq_scale
            s11_db, s11_phase = pair_to_db(values[1], values[2], data_format)
            s21_db, s21_phase = pair_to_db(values[3], values[4], data_format)
            s12_db, s12_phase = pair_to_db(values[5], values[6], data_format)
            s22_db, s22_phase = pair_to_db(values[7], values[8], data_format)
            rows.append(
                {
                    "freq_MHz": freq_mhz,
                    "S11_dB": s11_db,
                    "S21_dB": s21_db,
                    "S12_dB": s12_db,
                    "S22_dB": s22_db,
                    "S11_phase_deg": s11_phase,
                    "S21_phase_deg": s21_phase,
                    "S12_phase_deg": s12_phase,
                    "S22_phase_deg": s22_phase,
                    "path_loss_dB": -s21_db,
                    "S21_mag2": 10.0 ** (s21_db / 10.0),
                }
            )
    return rows


def finite_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def field_max(hfss: Hfss, object_name: str, solution: str, freq_mhz: float) -> tuple[float | None, str]:
    intrinsics = {"Freq": f"{freq_mhz}MHz", "Phase": "0deg"}
    errors = []
    for quantity, is_vector in [("E", True), ("Mag_E", False)]:
        try:
            value = hfss.post.get_scalar_field_value(
                quantity=quantity,
                scalar_function="Maximum",
                solution=solution,
                intrinsics=intrinsics,
                object_name=object_name,
                object_type="volume",
                is_vector=is_vector,
            )
            value = float(value)
            if math.isfinite(value):
                return value, f"hfss_field_calculator:{quantity}"
        except Exception as exc:
            errors.append(f"{quantity}: {exc}")
    return None, "field_unavailable: " + " | ".join(errors)


def extract_field_proxy(hfss: Hfss, case: Case, geom: dict[str, Any]) -> dict[str, Any]:
    solution = f"{SETUP_NAME} : {SWEEP_NAME}"
    local_e, local_source = field_max(hfss, str(geom["obs_name"]), solution, FIELD_FREQ_MHZ)
    local_material = local_material_for_case(case)
    q_rx = local_material.conductivity * local_e * local_e if local_e is not None else None

    global_candidates: list[tuple[str, float]] = []
    global_sources: list[str] = []
    for obj_name, material in [
        ("skin_layer", MATERIALS["skin_sweat"] if case.condition == "sweat" else MATERIALS["skin_rest"]),
        ("fat_layer", MATERIALS["fat"]),
        ("muscle_layer", MATERIALS["muscle"]),
    ]:
        emax, source = field_max(hfss, obj_name, solution, FIELD_FREQ_MHZ)
        global_sources.append(f"{obj_name}:{source}")
        if emax is not None:
            global_candidates.append((obj_name, material.conductivity * emax * emax))

    if global_candidates:
        global_obj, q_global = max(global_candidates, key=lambda item: item[1])
        global_note = f"max over tissue layers; controlling_object={global_obj}"
    else:
        q_global = None
        global_note = "global field unavailable"

    e2 = local_e * local_e if local_e is not None else None
    return {
        "case": case.name,
        "label": case.label,
        "type": case.rx_type,
        "depth_mm": case.rx_depth_mm,
        "condition": case.condition,
        "freq_MHz": FIELD_FREQ_MHZ,
        "obs_region": str(geom["obs_name"]),
        "observation_volume": geom["observation_volume"],
        "Emax_V_per_m": finite_or_blank(local_e),
        "E2max_V2_per_m2": finite_or_blank(e2),
        "sigma_S_per_m": local_material.conductivity,
        "Q_rx_max": finite_or_blank(q_rx),
        "Q_rx_norm": "",
        "Q_global_max": finite_or_blank(q_global),
        "Q_global_norm": "",
        "field_source": local_source,
        "field_note": f"Q_rx_max=sigma*max(|E|)^2 at receiver observation volume; {global_note}; {'; '.join(global_sources)}",
    }


def run_and_export(hfss: Hfss, geometry: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for case in CASES:
        print(f"Solving {case.name} ({case.label})", flush=True)
        hfss.set_active_design(case.name)
        ok = hfss.analyze_setup(SETUP_NAME, cores=4, tasks=1, blocking=True)
        touchstone_path = OUT_DIR / f"{case.name}.s2p"
        if ok:
            hfss.export_touchstone(setup=SETUP_NAME, sweep=SWEEP_NAME, output_file=str(touchstone_path), renormalization=True, impedance=50)
        if not touchstone_path.exists():
            raise RuntimeError(f"Touchstone export failed for {case.name}")
        parsed = parse_touchstone(touchstone_path)
        if not parsed:
            raise RuntimeError(f"Touchstone parse failed for {case.name}")
        for result in parsed:
            all_rows.append(
                {
                    "case": case.name,
                    "label": case.label,
                    "type": case.rx_type,
                    "condition": case.condition,
                    "depth_mm": case.rx_depth_mm,
                    "rx_z_mm": geometry[case.label]["rx_z_mm"],
                    "touchstone_file": str(touchstone_path),
                    **result,
                }
            )
        field_rows.append(extract_field_proxy(hfss, case, geometry[case.label]))
    normalize_field_rows(field_rows)
    return all_rows, field_rows


def normalize_field_rows(field_rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in field_rows if row["label"] == "surface_rest"), None)
    if not baseline:
        return
    try:
        q_base = float(baseline["Q_rx_max"])
    except (TypeError, ValueError):
        q_base = float("nan")
    try:
        q_global_base = float(baseline["Q_global_max"])
    except (TypeError, ValueError):
        q_global_base = float("nan")
    for row in field_rows:
        try:
            q = float(row["Q_rx_max"])
            row["Q_rx_norm"] = q / q_base if math.isfinite(q) and math.isfinite(q_base) and q_base > 0 else ""
        except (TypeError, ValueError):
            row["Q_rx_norm"] = ""
        try:
            qg = float(row["Q_global_max"])
            row["Q_global_norm"] = qg / q_global_base if math.isfinite(qg) and math.isfinite(q_global_base) and q_global_base > 0 else ""
        except (TypeError, ValueError):
            row["Q_global_norm"] = ""


def write_raw_results(rows: list[dict[str, Any]]) -> None:
    path = OUT_DIR / "sparams_raw.csv"
    fieldnames = [
        "case",
        "label",
        "type",
        "condition",
        "depth_mm",
        "rx_z_mm",
        "freq_MHz",
        "S11_dB",
        "S21_dB",
        "S22_dB",
        "path_loss_dB",
        "S21_mag2",
        "S11_phase_deg",
        "S21_phase_deg",
        "S12_dB",
        "S12_phase_deg",
        "S22_phase_deg",
        "touchstone_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_field_proxy(field_rows: list[dict[str, Any]]) -> None:
    path = OUT_DIR / "field_proxy.csv"
    fieldnames = [
        "case",
        "label",
        "type",
        "depth_mm",
        "condition",
        "freq_MHz",
        "obs_region",
        "observation_volume",
        "Emax_V_per_m",
        "E2max_V2_per_m2",
        "sigma_S_per_m",
        "Q_rx_max",
        "Q_rx_norm",
        "Q_global_max",
        "Q_global_norm",
        "field_source",
        "field_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in field_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def nearest_row(rows: list[dict[str, Any]], label: str, target_freq_mhz: float) -> dict[str, Any]:
    candidates = [row for row in rows if row["label"] == label]
    if not candidates:
        raise RuntimeError(f"No rows for label {label}")
    return min(candidates, key=lambda row: abs(float(row["freq_MHz"]) - target_freq_mhz))


def field_row_by_label(field_rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    return next((row for row in field_rows if row["label"] == label), None)


def blank_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def calibrated_library(rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_energy = float(nearest_row(rows, "surface_rest", ENERGY_FREQ_MHZ)["S21_mag2"])
    baseline_data = float(nearest_row(rows, "surface_rest", DATA_FREQ_MHZ)["S21_mag2"])
    base_field = field_row_by_label(field_rows, "surface_rest")
    base_q = blank_float(base_field.get("Q_rx_max") if base_field else None)
    base_eta = base_q / baseline_energy if base_q is not None and baseline_energy > 0 else None

    library: list[dict[str, Any]] = []
    for case in CASES:
        energy = nearest_row(rows, case.label, ENERGY_FREQ_MHZ)
        data = nearest_row(rows, case.label, DATA_FREQ_MHZ)
        g_raw = float(energy["S21_mag2"])
        r_raw = float(data["S21_mag2"])
        field = field_row_by_label(field_rows, case.label)
        q = blank_float(field.get("Q_rx_max") if field else None)
        eta = q / g_raw if q is not None and g_raw > 0 else None
        library.append(
            {
                "case": case.name,
                "label": case.label,
                "type": case.rx_type,
                "depth_mm": case.rx_depth_mm,
                "condition": case.condition,
                "s21_energy_db": energy["S21_dB"],
                "path_loss_energy_db": energy["path_loss_dB"],
                "g_norm": g_raw / baseline_energy if baseline_energy else "",
                "s21_data_db": data["S21_dB"],
                "path_loss_data_db": data["path_loss_dB"],
                "r_norm": r_raw / baseline_data if baseline_data else "",
                "Q_rx_max": finite_or_blank(q),
                "chi_norm": q / base_q if q is not None and base_q is not None and base_q > 0 else "",
                "eta_field_over_g": finite_or_blank(eta),
                "p_rx_cap_norm": base_eta / eta if eta is not None and eta > 0 and base_eta is not None else "",
                "field_source": field.get("field_source", "") if field else "",
                "source_note": "HFSS reduced-order proxy; not SAR certification or patient-specific validation",
            }
        )
    return library


def write_calibrated_library(rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    library = calibrated_library(rows, field_rows)
    fieldnames = [
        "case",
        "label",
        "type",
        "depth_mm",
        "condition",
        "s21_energy_db",
        "path_loss_energy_db",
        "g_norm",
        "s21_data_db",
        "path_loss_data_db",
        "r_norm",
        "Q_rx_max",
        "chi_norm",
        "eta_field_over_g",
        "p_rx_cap_norm",
        "field_source",
        "source_note",
    ]
    for filename in ["calibrated_library.csv", "calibrated_library_full.csv", "updated_simulation_results.csv"]:
        with (OUT_DIR / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in library:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    return library


def write_case_manifest() -> None:
    path = OUT_DIR / "case_manifest.csv"
    fieldnames = [
        "case",
        "label",
        "type",
        "condition",
        "depth_mm",
        "rx_x_mm",
        "rx_pair_gap_mm",
        "rx_pad_x_mm",
        "rx_pad_y_mm",
        "rx_pad_z_mm",
        "loose_gap_mm",
        "condition_model",
    ]
    descriptions = {
        "rest": "Baseline direct contact and nominal tissue material.",
        "loose": "Surface receiver lifted by 0.5 mm air gap; hub remains in direct contact.",
        "sweat": "Skin conductivity scaled 3x for wet-skin proxy.",
        "stress": "Implant receiver shifted laterally and electrode pair gap increased with smaller pads.",
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in CASES:
            writer.writerow(
                {
                    "case": case.name,
                    "label": case.label,
                    "type": case.rx_type,
                    "condition": case.condition,
                    "depth_mm": case.rx_depth_mm,
                    "rx_x_mm": case.rx_x_mm,
                    "rx_pair_gap_mm": case.rx_pair_gap_mm,
                    "rx_pad_x_mm": case.rx_pad_x_mm,
                    "rx_pad_y_mm": case.rx_pad_y_mm,
                    "rx_pad_z_mm": case.rx_pad_z_mm,
                    "loose_gap_mm": case.loose_gap_mm,
                    "condition_model": descriptions[case.condition],
                }
            )


def write_setup_tables() -> None:
    with (OUT_DIR / "hfss_setup_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "setting"])
        writer.writerow(["model", "120 mm x 100 mm x 60 mm three-layer tissue block"])
        writer.writerow(["coordinates", "skin surface z=0 mm; tissue depth is negative z"])
        writer.writerow(["layers", "skin: 0 to -2 mm; fat: -2 to -10 mm; muscle: -10 to -60 mm"])
        writer.writerow(["frequencies", "f_E=13.56 MHz, f_D=40 MHz"])
        writer.writerow(["ports", "two 50 ohm lumped ports with manually created YZ port sheets"])
        writer.writerow(["hub electrodes", "rectangular copper pads, 10 x 6 x 0.2 mm, 8 mm gap, fixed at x=0 mm"])
        writer.writerow(["surface receiver", "rectangular copper pads, 6 x 3 x 0.2 mm, 4 mm gap, x=50 mm"])
        writer.writerow(["implant receiver", "rectangular copper pads, 4 x 2 x 1.0 mm, 4 mm gap, x=50 mm, depths 10/30 mm"])
        writer.writerow(["implant stress", "same stress rule at 10 and 30 mm: x=60 mm, 6 mm gap, smaller pads"])
        writer.writerow(["surface loose", "receiver lifted by 0.5 mm air gap; hub unchanged"])
        writer.writerow(["surface sweat", "skin conductivity scaled from 0.70 to 2.10 S/m"])
        writer.writerow(["boundary", "air box with radiation boundary"])
        writer.writerow(["mesh", "adaptive, max Delta S 0.02, up to 6 passes; electrode local length mesh target 1 mm"])
        writer.writerow(["field proxy", "Q_rx_max=sigma*max(|E|)^2 at 13.56 MHz in 5 mm receiver observation cube"])

    with (OUT_DIR / "tissue_materials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "relative_permittivity", "conductivity_S_per_m", "density_kg_per_m3", "note"])
        for name, material in MATERIALS.items():
            writer.writerow([name, material.permittivity, material.conductivity, material.density, material.note])


def run_mesh_convergence(hfss: Hfss, baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mesh_cases = [case for case in CASES if case.label in {"surface_rest", "implant30_stress"}]
    mesh_rows: list[dict[str, Any]] = []
    for case in mesh_cases:
        strict_name = f"mesh_{case.name}_strict"
        print(f"Solving mesh convergence design {strict_name}", flush=True)
        hfss.insert_design(strict_name, solution_type="Modal")
        hfss.set_active_design(strict_name)
        add_materials(hfss)
        create_case_geometry(hfss, case)
        create_setup(hfss, max_delta_s=0.01, maximum_passes=8, save_fields=False)
        ok = hfss.analyze_setup(SETUP_NAME, cores=4, tasks=1, blocking=True)
        touchstone_path = OUT_DIR / f"{strict_name}.s2p"
        if ok:
            hfss.export_touchstone(setup=SETUP_NAME, sweep=SWEEP_NAME, output_file=str(touchstone_path), renormalization=True, impedance=50)
        if not touchstone_path.exists():
            raise RuntimeError(f"Mesh convergence touchstone export failed for {strict_name}")
        strict_rows = parse_touchstone(touchstone_path)
        old_e = nearest_row(baseline_rows, case.label, ENERGY_FREQ_MHZ)
        old_d = nearest_row(baseline_rows, case.label, DATA_FREQ_MHZ)
        new_e = min(strict_rows, key=lambda row: abs(float(row["freq_MHz"]) - ENERGY_FREQ_MHZ))
        new_d = min(strict_rows, key=lambda row: abs(float(row["freq_MHz"]) - DATA_FREQ_MHZ))
        mesh_rows.append(
            {
                "case": case.name,
                "label": case.label,
                "old_S21_13p56_dB": old_e["S21_dB"],
                "new_S21_13p56_dB": new_e["S21_dB"],
                "diff_13p56_dB": float(new_e["S21_dB"]) - float(old_e["S21_dB"]),
                "old_S21_40_dB": old_d["S21_dB"],
                "new_S21_40_dB": new_d["S21_dB"],
                "diff_40_dB": float(new_d["S21_dB"]) - float(old_d["S21_dB"]),
                "criterion_abs_diff_dB": 0.5,
                "acceptable_abs_diff_dB": 1.0,
                "strict_mesh_setting": "MaxDeltaS=0.01, MaximumPasses=8",
                "touchstone_file": str(touchstone_path),
            }
        )
    write_mesh_convergence(mesh_rows)
    return mesh_rows


def write_mesh_convergence(mesh_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "label",
        "old_S21_13p56_dB",
        "new_S21_13p56_dB",
        "diff_13p56_dB",
        "old_S21_40_dB",
        "new_S21_40_dB",
        "diff_40_dB",
        "criterion_abs_diff_dB",
        "acceptable_abs_diff_dB",
        "mesh_status",
        "strict_mesh_setting",
        "touchstone_file",
    ]
    with (OUT_DIR / "mesh_convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mesh_rows:
            d13 = abs(float(row["diff_13p56_dB"]))
            d40 = abs(float(row["diff_40_dB"]))
            if d13 < 0.5 and d40 < 0.5:
                row["mesh_status"] = "ideal"
            elif d13 < 1.0 and d40 < 1.0:
                row["mesh_status"] = "acceptable_caution"
            else:
                row["mesh_status"] = "needs_refinement"
            writer.writerow(row)

    lines = [
        "# Mesh Convergence Note",
        "",
        "Representative cases were rerun with MaxDeltaS=0.01 and MaximumPasses=8.",
        "Ideal target: abs(Delta S21) < 0.5 dB at both calibration frequencies.",
        "Cautious acceptance band: 0.5 to 1.0 dB. Values above 1.0 dB need further mesh refinement.",
        "",
        "| case | diff 13.56 MHz dB | diff 40 MHz dB | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in mesh_rows:
        d13 = float(row["diff_13p56_dB"])
        d40 = float(row["diff_40_dB"])
        if abs(d13) < 0.5 and abs(d40) < 0.5:
            status = "ideal"
        elif abs(d13) < 1.0 and abs(d40) < 1.0:
            status = "acceptable_caution"
        else:
            status = "needs_refinement"
        lines.append(f"| {row['label']} | {d13:.3f} | {d40:.3f} | {status} |")
    (OUT_DIR / "mesh_convergence_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(library: list[dict[str, Any]], mesh_rows: list[dict[str, Any]] | None) -> None:
    lines = [
        "# HFSS Calibration Summary",
        "",
        "Scope: reduced-order HFSS calibration proxy for HBC scheduling parameters. This is not SAR certification or patient-specific validation.",
        "",
        f"Project: `{PROJECT_PATH}`",
        f"Frequencies: f_E={ENERGY_FREQ_MHZ} MHz, f_D={DATA_FREQ_MHZ} MHz",
        "",
        "## Calibrated Library",
        "",
        "| case | type | depth mm | condition | g_norm | r_norm | chi_norm | p_rx_cap_norm |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in library:
        def fmt(key: str) -> str:
            value = row.get(key, "")
            if value == "":
                return ""
            return f"{float(value):.3f}"

        lines.append(
            f"| {row['label']} | {row['type']} | {float(row['depth_mm']):.1f} | {row['condition']} | "
            f"{fmt('g_norm')} | {fmt('r_norm')} | {fmt('chi_norm')} | {fmt('p_rx_cap_norm')} |"
        )
    if mesh_rows:
        lines.extend(
            [
                "",
                "## Mesh Convergence",
                "",
                "| case | diff 13.56 MHz dB | diff 40 MHz dB |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in mesh_rows:
            lines.append(f"| {row['label']} | {float(row['diff_13p56_dB']):.3f} | {float(row['diff_40_dB']):.3f} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `sparams_raw.csv`: S11/S21/S22 and path loss.",
            "- `field_proxy.csv`: Q_rx_max, chi normalization source, and observation volume.",
            "- `calibrated_library_full.csv`: g_norm, r_norm, chi_norm, and p_rx_cap_norm.",
            "- `mesh_convergence.csv`: representative strict-mesh reruns when mesh check is enabled.",
        ]
    )
    (OUT_DIR / "hfss_calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(status: str, detail: str) -> None:
    (OUT_DIR / "run_summary.txt").write_text(
        "\n".join(
            [
                f"status={status}",
                f"project={PROJECT_PATH}",
                f"cases={len(CASES)}",
                f"energy_frequency_mhz={ENERGY_FREQ_MHZ}",
                f"data_frequency_mhz={DATA_FREQ_MHZ}",
                "field_proxy=Q_rx_max=sigma*max(|E|)^2 at f_E in receiver observation volume",
                "limitation=Reduced-order HFSS calibration proxy only; not SAR certification or patient-specific validation.",
                detail,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally run the HBC HFSS calibration project.")
    parser.add_argument("--run", action="store_true", help="Analyze all cases and export S-parameter and field-proxy outputs.")
    parser.add_argument("--mesh-check", action="store_true", help="Rerun surface_rest and implant30_stress with stricter mesh settings.")
    parser.add_argument("--graphical", action="store_true", help="Launch AEDT with the GUI instead of non-graphical mode.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove the previously generated project before rebuilding.")
    args = parser.parse_args()

    hfss = None
    try:
        hfss, geometry = build_project(non_graphical=not args.graphical, overwrite=not args.keep_existing)
        write_case_manifest()
        write_setup_tables()
        if args.run:
            rows, field_rows = run_and_export(hfss, geometry)
            write_raw_results(rows)
            write_field_proxy(field_rows)
            library = write_calibrated_library(rows, field_rows)
            mesh_rows = run_mesh_convergence(hfss, rows) if args.mesh_check else []
            write_markdown_summary(library, mesh_rows)
            write_summary("completed", f"analysis=completed; mesh_check={'completed' if args.mesh_check else 'not_requested'}")
        else:
            write_summary("build_only", "analysis=not_requested")
        hfss.save_project(str(PROJECT_PATH), overwrite=True)
        return 0
    except Exception:
        write_summary("failed", traceback.format_exc())
        raise
    finally:
        if hfss:
            hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
