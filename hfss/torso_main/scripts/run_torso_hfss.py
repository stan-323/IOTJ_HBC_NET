from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("TORSO_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT_DIR = ROOT / "outputs"
PROJECT_DIR = ROOT / "projects"
FIELD_DIR = OUT_DIR / "field_samples"
PROJECT_PATH = PROJECT_DIR / "hbc_torso_main.aedt"

AEDT_VERSION = "2024.1"
SETUP_NAME = "Setup1"
ENERGY_FREQ_MHZ = 13.56
DATA_FREQ_MHZ = 40.0
FREQUENCY_SWEEP_MHZ = [1.0, 10.0, ENERGY_FREQ_MHZ, 21.0, DATA_FREQ_MHZ]
BASE_REPORT_FREQS_MHZ = [ENERGY_FREQ_MHZ, DATA_FREQ_MHZ]

TORSO_RADIUS_MM = 150.0
TORSO_LENGTH_MM = 300.0
SKIN_THICKNESS_MM = 2.0
FAT_THICKNESS_MM = 5.0
MUSCLE_RADIUS_MM = TORSO_RADIUS_MM - SKIN_THICKNESS_MM - FAT_THICKNESS_MM
FAT_OUTER_RADIUS_MM = TORSO_RADIUS_MM - SKIN_THICKNESS_MM
AIR_MARGIN_MM = 300.0
FIBROSIS_THICKNESS_MM = 1.0

FREQUENCY_SWEEP_LABELS = {"t_surf_rest", "t_imp10_rest", "t_imp30_stress"}

SOLVER_SETTINGS: dict[str, float] = {
    "max_delta_s": 0.01,
    "max_passes": 8,
    "electrode_mesh_mm": 0.5,
    "loose_gap_mesh_mm": 0.1,
    "fibrosis_mesh_mm": 0.2,
    "incident_power_w": 1.0,
    "port_impedance_ohm": 50.0,
}


@dataclass(frozen=True)
class TissueMaterial:
    permittivity: float
    conductivity: float
    density: float
    note: str


@dataclass(frozen=True)
class TorsoCase:
    label: str
    study: str
    rx_type: str
    clinical_scene: str
    condition: str
    rx_x_mm: float
    rx_depth_mm: float
    rx_pair_gap_mm: float
    rx_pad_x_mm: float
    rx_pad_y_mm: float
    rx_pad_z_mm: float
    skin_material_key: str = "skin_rest"
    rx_lift_mm: float = 0.0
    fibrosis_shell_mm: float = 0.0
    contact_layer_mm: float = 0.0  # thickness of degraded-contact interface layer under RX pads
    stress_mechanism: str = ""
    note: str = ""


@dataclass(frozen=True)
class TorsoLayer:
    name: str
    inner_radius_mm: float
    outer_radius_mm: float
    material_key: str


MATERIALS = {
    "skin_rest": TissueMaterial(120.0, 0.70, 1109.0, "Dry chest-wall skin proxy near 1-40 MHz"),
    "skin_sweat": TissueMaterial(120.0, 1.10, 1109.0, "Wet-skin proxy for exercise sweat"),
    "fat": TissueMaterial(18.0, 0.05, 911.0, "Chest/abdominal subcutaneous fat proxy"),
    "muscle": TissueMaterial(95.0, 0.80, 1090.0, "Pectoral/intercostal muscle proxy"),
    "fibrosis": TissueMaterial(60.0, 0.20, 1050.0, "Fibrotic encapsulation sensitivity proxy"),
    "degraded_contact": TissueMaterial(60.0, 0.30, 1000.0, "Dried/aged hydrogel adhesive interface; degraded ECG patch proxy"),
}


HFSS_MATERIAL_NAMES = {
    "skin_rest": "hbc_skin_rest_proxy",
    "skin_sweat": "hbc_skin_sweat_proxy",
    "fat": "hbc_fat_proxy",
    "muscle": "hbc_muscle_proxy",
    "fibrosis": "hbc_fibrosis_proxy",
    "degraded_contact": "hbc_degraded_contact_proxy",
    "copper": "copper",
    "air": "air",
}


MAIN_LIBRARY_CASES = [
    TorsoCase(
        label="t_surf_rest",
        study="A_main_library",
        rx_type="surface",
        clinical_scene="Dry chest-wall skin with ECG-style patch electrode",
        condition="surface_rest",
        rx_x_mm=50.0,
        rx_depth_mm=0.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        note="Normalization baseline for scheduler coefficients",
    ),
    TorsoCase(
        label="t_surf_sweat",
        study="A_main_library",
        rx_type="surface",
        clinical_scene="Exercise sweat on chest-wall skin",
        condition="surface_sweat",
        rx_x_mm=50.0,
        rx_depth_mm=0.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        skin_material_key="skin_sweat",
        note="Skin conductivity raised to 1.10 S/m",
    ),
    TorsoCase(
        label="t_surf_loose",
        study="A_main_library",
        rx_type="surface",
        clinical_scene="Aged ECG patch adhesive; dried hydrogel creates resistive contact interface",
        condition="surface_loose",
        rx_x_mm=50.0,
        rx_depth_mm=0.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        rx_lift_mm=0.0,
        contact_layer_mm=0.3,
        stress_mechanism="degraded_contact_interface",
        note="0.3 mm degraded-contact layer (sigma=0.30 S/m, eps_r=60) replaces air-gap model; air gaps break galvanic coupling catastrophically at 13.56 MHz",
    ),
    TorsoCase(
        label="t_surf_fail",
        study="A_main_library",
        rx_type="surface",
        clinical_scene="ECG patch contact failure with full local air gap",
        condition="surface_contact_failure",
        rx_x_mm=50.0,
        rx_depth_mm=0.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        rx_lift_mm=0.5,
        stress_mechanism="full_air_gap",
        note="Compatibility state for scheduler surface_contact_failure; excluded from main-condition library rows",
    ),
    TorsoCase(
        label="t_imp10_rest",
        study="A_main_library",
        rx_type="implant",
        clinical_scene="Subcutaneous neurostimulator or IPG",
        condition="implant10_rest",
        rx_x_mm=50.0,
        rx_depth_mm=10.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
    ),
    TorsoCase(
        label="t_imp10_stress",
        study="A_main_library",
        rx_type="implant",
        clinical_scene="Subcutaneous implant with chronic fibrotic encapsulation",
        condition="implant10_stress",
        rx_x_mm=50.0,
        rx_depth_mm=10.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
        fibrosis_shell_mm=FIBROSIS_THICKNESS_MM,
        stress_mechanism="fibrotic_encapsulation",
    ),
    TorsoCase(
        label="t_imp30_rest",
        study="A_main_library",
        rx_type="implant",
        clinical_scene="Deeper chest/pericardial-near implanted device",
        condition="implant30_rest",
        rx_x_mm=50.0,
        rx_depth_mm=30.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
    ),
    TorsoCase(
        label="t_imp30_stress",
        study="A_main_library",
        rx_type="implant",
        clinical_scene="Deeper implanted device with chronic fibrotic encapsulation",
        condition="implant30_stress",
        rx_x_mm=50.0,
        rx_depth_mm=30.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
        fibrosis_shell_mm=FIBROSIS_THICKNESS_MM,
        stress_mechanism="fibrotic_encapsulation",
    ),
]


DISTANCE_SCAN_CASES = [
    TorsoCase(
        label=f"t_scan_x{x:03d}",
        study="C_distance_scan",
        rx_type="surface",
        clinical_scene=f"Chest-wall surface path-loss scan at x={x} mm",
        condition="surface_rest_distance_scan",
        rx_x_mm=float(x),
        rx_depth_mm=0.0,
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=6.0,
        rx_pad_y_mm=3.0,
        rx_pad_z_mm=0.2,
        note="Physical anchor for surface distance scaling",
    )
    for x in [30, 50, 70, 110, 150, 200]
]


DEPTH_SCAN_CASES = [
    TorsoCase(
        label=f"t_depth_d{depth:03d}",
        study="D_depth_scan",
        rx_type="implant",
        clinical_scene=f"Implant-depth efficiency scan at d={depth} mm",
        condition="implant_depth_scan",
        rx_x_mm=50.0,
        rx_depth_mm=float(depth),
        rx_pair_gap_mm=4.0,
        rx_pad_x_mm=4.0,
        rx_pad_y_mm=2.0,
        rx_pad_z_mm=1.0,
        note="Depth anchor for zeta_i and implant feasibility range",
    )
    for depth in [10, 20, 30, 40, 50, 70, 100]
]


def unique_cases() -> list[TorsoCase]:
    return [*MAIN_LIBRARY_CASES, *DISTANCE_SCAN_CASES, *DEPTH_SCAN_CASES]


def select_cases(studies: list[str] | None = None, case_labels: list[str] | None = None) -> list[TorsoCase]:
    cases = unique_cases()
    if studies:
        requested_studies = set(studies)
        cases = [case for case in cases if case.study in requested_studies]
        missing_studies = requested_studies.difference({case.study for case in unique_cases()})
        if missing_studies:
            raise ValueError(f"Unknown torso HFSS study: {sorted(missing_studies)}")
    if case_labels:
        requested_labels = list(case_labels)
        by_label = {case.label: case for case in cases}
        missing_labels = [label for label in requested_labels if label not in by_label]
        if missing_labels:
            raise ValueError(f"Unknown torso HFSS case label: {missing_labels}")
        cases = [by_label[label] for label in requested_labels]
    return cases


def torso_layers(skin_material_key: str = "skin_rest") -> list[TorsoLayer]:
    return [
        TorsoLayer("muscle_core", 0.0, MUSCLE_RADIUS_MM, "muscle"),
        TorsoLayer("fat_shell", MUSCLE_RADIUS_MM, FAT_OUTER_RADIUS_MM, "fat"),
        TorsoLayer("skin_shell", FAT_OUTER_RADIUS_MM, TORSO_RADIUS_MM, skin_material_key),
    ]


def receiver_z_for_case(case: TorsoCase) -> float:
    return TORSO_RADIUS_MM - case.rx_depth_mm + case.rx_lift_mm


def analysis_frequency_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in MAIN_LIBRARY_CASES:
        for freq in BASE_REPORT_FREQS_MHZ:
            rows.append({"analysis_group": "A_main_library", "label": case.label, "freq_MHz": freq})
    for label in ["t_surf_rest", "t_imp10_rest", "t_imp30_stress"]:
        for freq in FREQUENCY_SWEEP_MHZ:
            rows.append({"analysis_group": "B_frequency_sweep", "label": label, "freq_MHz": freq})
    for case in DISTANCE_SCAN_CASES:
        for freq in BASE_REPORT_FREQS_MHZ:
            rows.append({"analysis_group": "C_distance_scan", "label": case.label, "freq_MHz": freq})
    for case in DEPTH_SCAN_CASES:
        for freq in BASE_REPORT_FREQS_MHZ:
            rows.append({"analysis_group": "D_depth_scan", "label": case.label, "freq_MHz": freq})
    return rows


def solver_frequency_rows() -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    rows: list[dict[str, Any]] = []
    for row in analysis_frequency_rows():
        key = (str(row["label"]), float(row["freq_MHz"]))
        if key in seen:
            continue
        seen.add(key)
        out = dict(row)
        out["solver_note"] = "deduped from reviewer-analysis frequency requests"
        rows.append(out)
    return rows


def air_wavelength_fraction_at_40mhz() -> float:
    wavelength_m = 299_792_458.0 / (DATA_FREQ_MHZ * 1.0e6)
    return (AIR_MARGIN_MM / 1000.0) / wavelength_m


def ensure_dirs() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)


def clean_generated() -> None:
    for path in [PROJECT_PATH, PROJECT_PATH.with_suffix(".aedtresults"), FIELD_DIR]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    FIELD_DIR.mkdir(parents=True, exist_ok=True)


def require_hfss():
    try:
        from ansys.aedt.core import Hfss
    except Exception as exc:  # pragma: no cover - depends on local AEDT installation
        raise RuntimeError("PyAEDT/Ansys AEDT is required for --build-only or --run.") from exc
    return Hfss


def safe_add_material(hfss: Any, name: str, material: TissueMaterial) -> None:
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


def add_materials(hfss: Any) -> None:
    for key, mat in MATERIALS.items():
        safe_add_material(hfss, HFSS_MATERIAL_NAMES[key], mat)


def set_visual(obj: Any, color: tuple[int, int, int], transparency: float) -> None:
    try:
        obj.color = color
    except Exception:
        pass
    try:
        obj.transparency = transparency
    except Exception:
        pass


def case_short_name(case: TorsoCase) -> str:
    return case.label[:48]


def design_name(case: TorsoCase) -> str:
    return case_short_name(case)


def freq_token(freq_mhz: float) -> str:
    return str(freq_mhz).replace(".", "p")


def sweep_name(freq_mhz: float) -> str:
    return "SP" + freq_token(freq_mhz)


def create_cylinder_shell(hfss: Any, layer: TorsoLayer) -> Any:
    material = HFSS_MATERIAL_NAMES[layer.material_key]
    outer = hfss.modeler.create_cylinder(
        orientation=0,
        origin=[-TORSO_LENGTH_MM / 2.0, 0.0, 0.0],
        radius=layer.outer_radius_mm,
        height=TORSO_LENGTH_MM,
        num_sides=128,
        name=layer.name,
        material=material,
    )
    if layer.inner_radius_mm > 0.0:
        cut = hfss.modeler.create_cylinder(
            orientation=0,
            origin=[-TORSO_LENGTH_MM / 2.0 - 0.1, 0.0, 0.0],
            radius=layer.inner_radius_mm,
            height=TORSO_LENGTH_MM + 0.2,
            num_sides=128,
            name=f"{layer.name}_inner_cut",
            material=HFSS_MATERIAL_NAMES["air"],
        )
        hfss.modeler.subtract([outer.name], [cut.name], keep_originals=False)
    return outer


def create_contact_layer(hfss: Any, case: TorsoCase, rx_z: float) -> list[str]:
    """Create thin degraded-contact boxes directly below each RX electrode pad.

    The contact layer (sigma=0.30 S/m) sits between the copper electrode and the skin
    (sigma=0.70 S/m), modelling a dried or aged hydrogel adhesive interface.  It is
    carved out of skin_shell (keep_originals=True) so both the layer solid and the carved
    skin remain part of the model.  The electrode is subsequently subtracted from skin in
    the normal path inside create_case_geometry.
    """
    thickness = case.contact_layer_mm
    # Electrode centre Y offset from axis (same formula as create_electrode_pair)
    offset = case.rx_pair_gap_mm / 2.0 + case.rx_pad_y_mm / 2.0
    bottom_z = rx_z - case.rx_pad_z_mm / 2.0 - thickness
    layer_names: list[str] = []
    for sign, suffix in [(1, "sig"), (-1, "ref")]:
        # Box-origin Y = minimum Y of the electrode footprint.
        # Signal (sign=+1): centre at +offset, origin at +offset - pad_y/2
        # Ref    (sign=-1): centre at -offset, origin at -offset - pad_y/2
        cy_near = sign * offset - case.rx_pad_y_mm / 2.0
        layer = hfss.modeler.create_box(
            [
                case.rx_x_mm - case.rx_pad_x_mm / 2.0,
                cy_near,
                bottom_z,
            ],
            [case.rx_pad_x_mm, case.rx_pad_y_mm, thickness],
            name=f"contact_layer_{suffix}",
            material=HFSS_MATERIAL_NAMES["degraded_contact"],
        )
        set_visual(layer, (180, 130, 90), 0.50)
        layer_names.append(layer.name)
    hfss.modeler.subtract(["skin_shell"], layer_names, keep_originals=True)
    return layer_names


def host_tissue_layers_for_implant(case: TorsoCase, extra_margin_mm: float = 0.0) -> list[str]:
    rx_z = receiver_z_for_case(case)
    half_z = case.rx_pad_z_mm / 2.0 + extra_margin_mm
    pair_y_extent = case.rx_pair_gap_mm + 2.0 * case.rx_pad_y_mm
    max_y = pair_y_extent / 2.0 + extra_margin_mm
    radial_min = max(0.0, abs(rx_z) - half_z)
    radial_max = math.sqrt(max_y * max_y + (abs(rx_z) + half_z) ** 2)
    layers: list[str] = []
    for layer in torso_layers(case.skin_material_key):
        if radial_max >= layer.inner_radius_mm and radial_min <= layer.outer_radius_mm:
            layers.append(layer.name)
    if not layers:
        raise ValueError(f"Implant case {case.label} does not overlap any torso tissue layer")
    return layers


def local_material_for_case(case: TorsoCase) -> TissueMaterial:
    if case.fibrosis_shell_mm > 0:
        return MATERIALS["fibrosis"]
    if case.rx_type == "surface":
        return MATERIALS[case.skin_material_key]
    host = host_tissue_layers_for_implant(case)[0]
    if host == "fat_shell":
        return MATERIALS["fat"]
    if host == "skin_shell":
        return MATERIALS[case.skin_material_key]
    return MATERIALS["muscle"]


def subtract_from_layers(hfss: Any, layer_names: list[str], tool_names: list[str]) -> None:
    for layer_name in layer_names:
        hfss.modeler.subtract([layer_name], tool_names, keep_originals=True)


def create_torso_tissue_stack(hfss: Any, case: TorsoCase) -> None:
    hfss.modeler.model_units = "mm"
    for layer in torso_layers(case.skin_material_key):
        obj = create_cylinder_shell(hfss, layer)
        if layer.name == "muscle_core":
            set_visual(obj, (190, 80, 85), 0.68)
        elif layer.name == "fat_shell":
            set_visual(obj, (240, 215, 120), 0.66)
        else:
            set_visual(obj, (125, 185, 255) if case.skin_material_key == "skin_sweat" else (230, 160, 150), 0.55)
    region = hfss.modeler.create_region(
        [f"{AIR_MARGIN_MM:g}mm"] * 6,
        pad_type="Absolute Offset",
    )
    set_visual(region, (180, 220, 255), 0.92)
    hfss.assign_radiation_boundary_to_objects(region, name="Radiation_Air")


def create_box_electrode(hfss: Any, name: str, cx: float, cy: float, cz: float, pad_x: float, pad_y: float, pad_z: float) -> Any:
    return hfss.modeler.create_box(
        [cx - pad_x / 2.0, cy - pad_y / 2.0, cz - pad_z / 2.0],
        [pad_x, pad_y, pad_z],
        name=name,
        material=HFSS_MATERIAL_NAMES["copper"],
    )


def create_electrode_pair(
    hfss: Any,
    prefix: str,
    cx: float,
    cz: float,
    pad_x: float,
    pad_y: float,
    pad_z: float,
    gap_y: float,
    port_name: str,
) -> tuple[str, str]:
    offset = gap_y / 2.0 + pad_y / 2.0
    sig = create_box_electrode(hfss, f"{prefix}_sig", cx, offset, cz, pad_x, pad_y, pad_z)
    ref = create_box_electrode(hfss, f"{prefix}_ref", cx, -offset, cz, pad_x, pad_y, pad_z)
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
        impedance=SOLVER_SETTINGS["port_impedance_ohm"],
        name=port_name,
        renormalize=True,
        deembed=False,
    )
    return sig.name, ref.name


def create_fibrosis_capsule(hfss: Any, case: TorsoCase, rx_z: float) -> str:
    margin = case.fibrosis_shell_mm
    pair_y_extent = case.rx_pair_gap_mm + 2.0 * case.rx_pad_y_mm
    capsule = hfss.modeler.create_box(
        [
            case.rx_x_mm - case.rx_pad_x_mm / 2.0 - margin,
            -pair_y_extent / 2.0 - margin,
            rx_z - case.rx_pad_z_mm / 2.0 - margin,
        ],
        [
            case.rx_pad_x_mm + 2.0 * margin,
            pair_y_extent + 2.0 * margin,
            case.rx_pad_z_mm + 2.0 * margin,
        ],
        name="fibrosis_capsule",
        material=HFSS_MATERIAL_NAMES["fibrosis"],
    )
    set_visual(capsule, (120, 90, 160), 0.45)
    subtract_from_layers(hfss, host_tissue_layers_for_implant(case, extra_margin_mm=case.fibrosis_shell_mm), [capsule.name])
    return capsule.name


def create_case_geometry(hfss: Any, case: TorsoCase) -> dict[str, Any]:
    create_torso_tissue_stack(hfss, case)
    hub_z = TORSO_RADIUS_MM - 0.5
    rx_z = receiver_z_for_case(case)
    create_electrode_pair(
        hfss,
        prefix="hub",
        cx=0.0,
        cz=hub_z,
        pad_x=10.0,
        pad_y=6.0,
        pad_z=0.2,
        gap_y=8.0,
        port_name="HubPort",
    )
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
    if case.rx_type == "surface":
        if case.contact_layer_mm > 0.0:
            # Carve degraded-contact boxes from skin before the electrode subtract so
            # the contact layer solids sit continuously between the electrode copper and skin.
            create_contact_layer(hfss, case, rx_z)
        hfss.modeler.subtract(["skin_shell"], ["hub_sig", "hub_ref", rx_sig, rx_ref], keep_originals=True)
    else:
        if case.fibrosis_shell_mm > 0.0:
            capsule_name = create_fibrosis_capsule(hfss, case, rx_z)
            hfss.modeler.subtract([capsule_name], [rx_sig, rx_ref], keep_originals=True)
        hfss.modeler.subtract(["skin_shell"], ["hub_sig", "hub_ref"], keep_originals=True)
        subtract_from_layers(hfss, host_tissue_layers_for_implant(case), [rx_sig, rx_ref])
    return {
        "rx_z_mm": rx_z,
        "hub_z_mm": hub_z,
        "surface_radius_mm": TORSO_RADIUS_MM,
        "observation_cube_mm": 5.0,
    }


def create_observation_box(hfss: Any, case: TorsoCase, rx_z: float, side_mm: float = 5.0) -> tuple[str, str]:
    half = side_mm / 2.0
    if case.rx_type == "surface":
        origin = [case.rx_x_mm - half, -half, TORSO_RADIUS_MM - side_mm]
    else:
        origin = [case.rx_x_mm - half, -half, rx_z - half]
    obs = hfss.modeler.create_box(origin, [side_mm, side_mm, side_mm], name=f"obs_rx_{int(side_mm)}mm_cube", material=HFSS_MATERIAL_NAMES["air"])
    try:
        obs.model = False
    except Exception:
        pass
    set_visual(obs, (70, 140, 250), 0.82)
    desc = f"{side_mm:g}mm cube origin=({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}) mm"
    return obs.name, desc


def create_setup(hfss: Any, case: TorsoCase, save_fields: bool = True) -> None:
    setup = hfss.create_setup(SETUP_NAME)
    setup.props["Frequency"] = f"{DATA_FREQ_MHZ}MHz"
    setup.props["MaximumPasses"] = int(SOLVER_SETTINGS["max_passes"])
    setup.props["MinimumPasses"] = 1
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["MaxDeltaS"] = SOLVER_SETTINGS["max_delta_s"]
    setup.update()
    for freq in case_frequencies(case.label):
        should_save_fields = save_fields and abs(freq - ENERGY_FREQ_MHZ) < 1.0e-9
        hfss.create_single_point_sweep(
            setup=SETUP_NAME,
            unit="MHz",
            freq=freq,
            name=sweep_name(freq),
            save_single_field=should_save_fields,
            save_fields=should_save_fields,
            save_rad_fields=False,
        )
    try:
        hfss.mesh.assign_length_mesh(
            ["hub_sig", "hub_ref", "rx_sig", "rx_ref"],
            maximum_length=SOLVER_SETTINGS["electrode_mesh_mm"],
            inside_selection=True,
            name="mesh_electrodes_0p5mm",
        )
    except Exception:
        pass
    if case.rx_lift_mm > 0:
        try:
            hfss.mesh.assign_length_mesh(
                ["rx_sig", "rx_ref"],
                maximum_length=SOLVER_SETTINGS["loose_gap_mesh_mm"],
                inside_selection=True,
                name="mesh_loose_gap_0p1mm",
            )
        except Exception:
            pass
    if case.fibrosis_shell_mm > 0:
        try:
            hfss.mesh.assign_length_mesh(
                ["fibrosis_capsule"],
                maximum_length=SOLVER_SETTINGS["fibrosis_mesh_mm"],
                inside_selection=True,
                name="mesh_fibrosis_0p2mm",
            )
        except Exception:
            pass


def build_project(cases: list[TorsoCase], non_graphical: bool = True, overwrite: bool = False) -> tuple[Any, dict[str, dict[str, Any]]]:
    ensure_dirs()
    if overwrite:
        clean_generated()
    if not cases:
        raise ValueError("At least one torso HFSS case is required")
    Hfss = require_hfss()
    if not overwrite and PROJECT_PATH.exists():
        hfss = Hfss(
            project=str(PROJECT_PATH),
            design=design_name(cases[0]),
            solution_type="Modal",
            version=AEDT_VERSION,
            non_graphical=non_graphical,
            new_desktop=True,
            close_on_exit=False,
            remove_lock=True,
        )
        geometry: dict[str, dict[str, Any]] = {}
        for case in cases:
            ok = hfss.set_active_design(design_name(case))
            if ok is False:
                raise RuntimeError(f"Existing AEDT project does not contain design {design_name(case)}")
            geometry[case.label] = {
                "rx_z_mm": receiver_z_for_case(case),
                "hub_z_mm": TORSO_RADIUS_MM - 0.5,
                "surface_radius_mm": TORSO_RADIUS_MM,
                "observation_cube_mm": 5.0,
            }
        return hfss, geometry
    hfss = Hfss(
        project=str(PROJECT_PATH),
        design=design_name(cases[0]),
        solution_type="Modal",
        version=AEDT_VERSION,
        non_graphical=non_graphical,
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
        add_materials(hfss)
        geometry[case.label] = create_case_geometry(hfss, case)
        obs_name, obs_desc = create_observation_box(hfss, case, float(geometry[case.label]["rx_z_mm"]))
        geometry[case.label]["obs_name"] = obs_name
        geometry[case.label]["observation_volume"] = obs_desc
        create_setup(hfss, case)
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
            try:
                values = [float(token) for token in line.split()]
            except ValueError:
                continue
            if len(values) < 9:
                continue
            s11_db, s11_phase = pair_to_db(values[1], values[2], data_format)
            s21_db, s21_phase = pair_to_db(values[3], values[4], data_format)
            s12_db, s12_phase = pair_to_db(values[5], values[6], data_format)
            s22_db, s22_phase = pair_to_db(values[7], values[8], data_format)
            rows.append(
                {
                    "freq_MHz": values[0] * freq_scale,
                    "S11_dB": s11_db,
                    "S21_dB": s21_db,
                    "S12_dB": s12_db,
                    "S22_dB": s22_db,
                    "path_loss_dB": -s21_db,
                    "S21_mag2": 10.0 ** (s21_db / 10.0),
                    "S11_phase_deg": s11_phase,
                    "S21_phase_deg": s21_phase,
                    "S12_phase_deg": s12_phase,
                    "S22_phase_deg": s22_phase,
                }
            )
    return rows


def sample_points_mm(case: TorsoCase, rx_z_mm: float, side_mm: float = 5.0, n: int = 5) -> list[list[float]]:
    if n < 2:
        raise ValueError("n must be at least 2")
    half = side_mm / 2.0
    if case.rx_type == "surface":
        start = [case.rx_x_mm - half, -half, TORSO_RADIUS_MM - side_mm]
        stop = [case.rx_x_mm + half, half, TORSO_RADIUS_MM]
    else:
        start = [case.rx_x_mm - half, -half, rx_z_mm - half]
        stop = [case.rx_x_mm + half, half, rx_z_mm + half]
    points: list[list[float]] = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                frac = [ix / (n - 1), iy / (n - 1), iz / (n - 1)]
                points.append([start[axis] + frac[axis] * (stop[axis] - start[axis]) for axis in range(3)])
    return points


def sample_points_si(case: TorsoCase, rx_z_mm: float, side_mm: float = 5.0, n: int = 5) -> list[list[float]]:
    return [[coord / 1000.0 for coord in point] for point in sample_points_mm(case, rx_z_mm, side_mm, n)]


def parse_field_magnitudes(path: Path) -> list[float]:
    magnitudes: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.lower().startswith("complex"):
                continue
            try:
                values = [float(token.strip(",")) for token in line.replace(",", " ").split()]
            except ValueError:
                continue
            if len(values) >= 9:
                ex = complex(values[3], values[4])
                ey = complex(values[5], values[6])
                ez = complex(values[7], values[8])
                magnitudes.append(math.sqrt(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2))
            elif len(values) >= 6:
                magnitudes.append(math.sqrt(values[-3] ** 2 + values[-2] ** 2 + values[-1] ** 2))
    return magnitudes


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    index = min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1)
    return sorted_values[index]


def field_stats_from_magnitudes(case: TorsoCase, magnitudes: list[float]) -> dict[str, float]:
    if not magnitudes:
        raise ValueError("field statistics require at least one magnitude")
    sorted_magnitudes = sorted(magnitudes)
    sigma = local_material_for_case(case).conductivity
    e_mean = sum(magnitudes) / len(magnitudes)
    e_p90 = percentile(sorted_magnitudes, 0.90)
    e_p95 = percentile(sorted_magnitudes, 0.95)
    e_p99 = percentile(sorted_magnitudes, 0.99)
    e_max = max(magnitudes)
    return {
        "Emean_sampled_V_per_m": e_mean,
        "Ep90_sampled_V_per_m": e_p90,
        "Ep95_sampled_V_per_m": e_p95,
        "Ep99_sampled_V_per_m": e_p99,
        "Emax_sampled_V_per_m": e_max,
        "Q_mean_W_per_m3": sigma * e_mean * e_mean,
        "Q_p90_W_per_m3": sigma * e_p90 * e_p90,
        "Q_p95_W_per_m3": sigma * e_p95 * e_p95,
        "Q_p99_W_per_m3": sigma * e_p99 * e_p99,
        "Q_max_W_per_m3": sigma * e_max * e_max,
    }


def export_sampled_field(hfss: Any, case: TorsoCase, rx_z_mm: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    hfss.post.export_field_file(
        quantity="E",
        solution=f"{SETUP_NAME} : {sweep_name(ENERGY_FREQ_MHZ)}",
        output_file=str(out),
        sample_points=sample_points_si(case, rx_z_mm, 5.0, 5),
        intrinsics={"Freq": f"{ENERGY_FREQ_MHZ}MHz", "Phase": "0deg"},
        export_in_si_system=True,
    )


def run_and_export(
    hfss: Any,
    cases: list[TorsoCase],
    geometry: dict[str, dict[str, Any]],
    out_dir: Path = OUT_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    field_dir = out_dir / "field_samples"
    field_dir.mkdir(parents=True, exist_ok=True)
    sparam_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for case in cases:
        hfss.set_active_design(design_name(case))
        ok = hfss.analyze_setup(SETUP_NAME, cores=4, tasks=1, blocking=True)
        if not ok:
            raise RuntimeError(f"HFSS solve failed for {design_name(case)}")
        for freq in case_frequencies(case.label):
            touchstone_path = out_dir / f"{design_name(case)}_{sweep_name(freq)}.s2p"
            hfss.export_touchstone(
                setup=SETUP_NAME,
                sweep=sweep_name(freq),
                output_file=str(touchstone_path),
                renormalization=True,
                impedance=SOLVER_SETTINGS["port_impedance_ohm"],
            )
            parsed = parse_touchstone(touchstone_path)
            if not parsed:
                raise RuntimeError(f"Touchstone parse failed for {design_name(case)} at {freq} MHz")
            nearest = min(parsed, key=lambda row: abs(float(row["freq_MHz"]) - freq))
            sparam_rows.append(
                {
                    "case": case.label,
                    "study": case.study,
                    "type": case.rx_type,
                    "condition": case.condition,
                    "rx_x_mm": case.rx_x_mm,
                    "depth_mm": case.rx_depth_mm,
                    "rx_z_mm": geometry[case.label]["rx_z_mm"],
                    "touchstone_file": str(touchstone_path),
                    **nearest,
                }
            )
        field_path = field_dir / f"{case.label}_5mm.fld"
        export_sampled_field(hfss, case, float(geometry[case.label]["rx_z_mm"]), field_path)
        magnitudes = parse_field_magnitudes(field_path)
        if not magnitudes:
            raise RuntimeError(f"Field export parse failed for {case.label}")
        field_rows.append(
            {
                "case": case.label,
                "study": case.study,
                "type": case.rx_type,
                "condition": case.condition,
                "depth_mm": case.rx_depth_mm,
                "rx_x_mm": case.rx_x_mm,
                "rx_z_mm": geometry[case.label]["rx_z_mm"],
                "freq_MHz": ENERGY_FREQ_MHZ,
                "sample_count": len(magnitudes),
                "sigma_S_per_m": local_material_for_case(case).conductivity,
                "sample_points_file": str(field_path),
                **field_stats_from_magnitudes(case, magnitudes),
            }
        )
    return sparam_rows, field_rows


def write_raw_outputs(sparam_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], out_dir: Path = OUT_DIR) -> None:
    write_csv(
        out_dir / "sparams_raw.csv",
        sparam_rows,
        [
            "case",
            "study",
            "type",
            "condition",
            "rx_x_mm",
            "depth_mm",
            "rx_z_mm",
            "freq_MHz",
            "S11_dB",
            "S21_dB",
            "S12_dB",
            "S22_dB",
            "path_loss_dB",
            "S21_mag2",
            "S11_phase_deg",
            "S21_phase_deg",
            "S12_phase_deg",
            "S22_phase_deg",
            "touchstone_file",
        ],
    )
    write_csv(
        out_dir / "field_proxy_samples.csv",
        field_rows,
        [
            "case",
            "study",
            "type",
            "condition",
            "depth_mm",
            "rx_x_mm",
            "rx_z_mm",
            "freq_MHz",
            "sample_count",
            "sigma_S_per_m",
            "Emean_sampled_V_per_m",
            "Ep90_sampled_V_per_m",
            "Ep95_sampled_V_per_m",
            "Ep99_sampled_V_per_m",
            "Emax_sampled_V_per_m",
            "Q_mean_W_per_m3",
            "Q_p90_W_per_m3",
            "Q_p95_W_per_m3",
            "Q_p99_W_per_m3",
            "Q_max_W_per_m3",
            "sample_points_file",
        ],
    )


def write_run_summary(status: str, cases: list[TorsoCase], out_dir: Path = OUT_DIR, detail: str = "") -> None:
    selected_frequency_rows = sum(len(case_frequencies(case.label)) for case in cases)
    lines = [
        "# Torso HFSS Run Summary",
        "",
        f"status: `{status}`",
        f"project: `{PROJECT_PATH}`",
        f"case_count: `{len(cases)}`",
        f"selected_solver_frequency_rows: `{selected_frequency_rows}`",
        f"manifest_analysis_frequency_requests: `{len(analysis_frequency_rows())}`",
        f"manifest_deduplicated_solver_frequency_rows: `{len(solver_frequency_rows())}`",
        "",
        "The outputs are electromagnetic simulation artifacts for scheduler-library construction. They do not replace SAR, Pennes bioheat, patient-specific simulation, or hardware compliance testing.",
    ]
    if detail:
        lines.extend(["", "## Detail", "", detail])
    (out_dir / "torso_hfss_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "run_summary.txt").write_text(
        "\n".join([f"status={status}", f"project={PROJECT_PATH}", f"case_count={len(cases)}", f"detail={detail}"]) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def case_rows() -> list[dict[str, Any]]:
    rows = []
    for case in unique_cases():
        row = asdict(case)
        row["frequencies_MHz"] = ";".join(str(freq) for freq in case_frequencies(case.label))
        rows.append(row)
    return rows


def case_frequencies(label: str) -> list[float]:
    if label in FREQUENCY_SWEEP_LABELS:
        return FREQUENCY_SWEEP_MHZ
    return BASE_REPORT_FREQS_MHZ


def material_rows() -> list[dict[str, Any]]:
    return [
        {
            "material": key,
            "epsilon_r": mat.permittivity,
            "sigma_S_per_m": mat.conductivity,
            "density_kg_per_m3": mat.density,
            "note": mat.note,
        }
        for key, mat in MATERIALS.items()
    ]


def setup_rows() -> list[dict[str, Any]]:
    return [
        {"item": "torso_model", "setting": f"{TORSO_LENGTH_MM:.0f} mm long, radius {TORSO_RADIUS_MM:.0f} mm circular chest/torso surrogate"},
        {"item": "layers", "setting": f"skin {SKIN_THICKNESS_MM:.0f} mm; fat {FAT_THICKNESS_MM:.0f} mm; muscle core radius {MUSCLE_RADIUS_MM:.0f} mm"},
        {"item": "frequencies", "setting": "1, 10, 13.56, 21, and 40 MHz where requested; all main/scan cases include 13.56 and 40 MHz"},
        {"item": "ports", "setting": "50 ohm lumped ports; 1.0 W incident-power reference for absolute field/SAR tables"},
        {"item": "electrodes", "setting": "hub 10 x 6 x 0.2 mm gap 8 mm; surface RX 6 x 3 x 0.2 mm gap 4 mm; implant RX 4 x 2 x 1 mm gap 4 mm"},
        {"item": "fat_thickness", "setting": f"{FAT_THICKNESS_MM:.0f} mm subcutaneous fat; muscle boundary at {MUSCLE_RADIUS_MM:.0f} mm radius; 10 mm implant lands in muscle"},
        {"item": "stress_model", "setting": "implant stress: 1 mm fibrotic encapsulation shell (sigma=0.20 S/m, eps_r=60); surface loose: 0.3 mm degraded-contact layer (sigma=0.30 S/m); no x-position drift"},
        {"item": "mesh", "setting": "adaptive max Delta S 0.01, max 8 passes, electrode mesh 0.5 mm, gap mesh 0.1 mm, fibrosis mesh 0.2 mm"},
        {"item": "air_region", "setting": f"{AIR_MARGIN_MM:.0f} mm margin is a practical near-field margin; it is {air_wavelength_fraction_at_40mhz():.3f} lambda at 40 MHz, not lambda/4"},
        {"item": "field_proxy", "setting": "receiver observation cube exports sigma*|E|^2 statistics; SAR uses dedicated 1g/10g post-processing"},
    ]


def physical_anchor_template_rows() -> list[dict[str, Any]]:
    return [
        {"scenario": "surface_x050", "label": "t_scan_x050", "freq_MHz": ENERGY_FREQ_MHZ, "path_loss_dB": "", "P_rx_W_at_1mW": "", "Q_p95_W_per_m3": "", "literature_anchor": "Bae 2012 21 MHz surface 50 mm"},
        {"scenario": "surface_x150", "label": "t_scan_x150", "freq_MHz": ENERGY_FREQ_MHZ, "path_loss_dB": "", "P_rx_W_at_1mW": "", "Q_p95_W_per_m3": "", "literature_anchor": "Distance-scale extrapolation"},
        {"scenario": "implant_d10", "label": "t_depth_d010", "freq_MHz": ENERGY_FREQ_MHZ, "path_loss_dB": "", "P_rx_W_at_1mW": "", "Q_p95_W_per_m3": "", "literature_anchor": "Subcutaneous IPG depth anchor"},
        {"scenario": "implant_d30", "label": "t_depth_d030", "freq_MHz": ENERGY_FREQ_MHZ, "path_loss_dB": "", "P_rx_W_at_1mW": "", "Q_p95_W_per_m3": "", "literature_anchor": "Callejon/Li depth trend cross-check"},
        {"scenario": "implant_d50", "label": "t_depth_d050", "freq_MHz": ENERGY_FREQ_MHZ, "path_loss_dB": "", "P_rx_W_at_1mW": "", "Q_p95_W_per_m3": "", "literature_anchor": "Pacemaker-depth feasibility anchor"},
    ]


def sar_chi_template_rows() -> list[dict[str, Any]]:
    return [
        {
            "label": case.label,
            "chi_norm": "",
            "SAR_1g_W_per_kg_hub_hotspot": "",
            "SAR_10g_W_per_kg_hub_hotspot": "",
            "SAR_1g_W_per_kg_rx_hotspot": "",
            "SAR_10g_W_per_kg_rx_hotspot": "",
            "chi_sar_rank_consistent": "",
        }
        for case in MAIN_LIBRARY_CASES
    ]


def lit_cross_validation_rows() -> list[dict[str, Any]]:
    return [
        {"source_key": "bae_tmt_2012", "published_quantity": "surface 50 mm around 21 MHz path loss", "hfss_case": "t_scan_x050 @ 21 MHz", "acceptance_band": "trend/scale, target within about 5 dB after digitization", "status": "pending_digitization"},
        {"source_key": "maity_tbme_2019", "published_quantity": "surface about 100 mm around 1 MHz path loss", "hfss_case": "t_scan_x110 @ 1 MHz", "acceptance_band": "trend/scale, target within about 6 dB after digitization", "status": "pending_digitization"},
        {"source_key": "datta_tbme_2021", "published_quantity": "EQS body-channel variability envelope", "hfss_case": "t_scan_x110 @ 1/10 MHz", "acceptance_band": "inside published variability envelope", "status": "pending_digitization"},
        {"source_key": "li_ne_2021", "published_quantity": "body-area power transfer depth/efficiency trend", "hfss_case": "t_depth_d020 normalized g", "acceptance_band": "order-of-magnitude consistency only", "status": "pending_digitization"},
        {"source_key": "callejon_tbme_2012", "published_quantity": "galvanic/body-coupled depth response", "hfss_case": "t_depth_d030", "acceptance_band": "trend/scale, target within about 8 dB after geometry caveats", "status": "pending_digitization"},
    ]


def design_review_text() -> str:
    return "\n".join(
        [
            "# Torso HFSS Design Review",
            "",
            "This package implements the reviewer-driven chest/torso HFSS experiment plan under this project's `hfss/torso_main` folder.",
            "",
            "## Locked Assumptions",
            "",
            "- Main scheduler states are clinically renamed around chest-wall surface patches and implanted devices.",
            "- Implant stress is modeled as 1 mm fibrotic encapsulation with sigma=0.20 S/m and epsilon_r=60, without moving the receiver in x.",
            "- The surface-loose state uses a receiver-side 0.5 mm local lift. A hub lift would be a global medium state and would not map cleanly to a per-node scheduling label.",
            "- A 300 mm air margin is retained as a practical near-field margin. At 40 MHz it is about 0.04 lambda in air, not lambda/4.",
            "- Reviewer-analysis rows intentionally contain 55 frequency requests. Solver rows deduplicate overlapping A/B requests to 49 actual case-frequency solves.",
            "- SAR outputs must be reported as post-processed 1g/10g quantities. The manuscript should not claim compliance until the solved fields are post-processed.",
            "",
            "## Generated Files",
            "",
            "- `case_manifest.csv`: 20 unique solver geometries.",
            "- `analysis_frequency_plan.csv`: 55 reviewer-facing frequency requests.",
            "- `solver_frequency_plan.csv`: 49 deduplicated solver case-frequency rows.",
            "- `material_table.csv` and `setup_table.csv`: Appendix A inputs.",
            "- `physical_anchor_template.csv`: Appendix C fill-in table.",
            "- `sar_chi_correlation_template.csv`: Appendix D fill-in table.",
            "- `lit_cross_validation_template.csv`: Appendix E digitization plan.",
        ]
    ) + "\n"


def write_all_manifests(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "case_manifest.csv",
        case_rows(),
        [
            "label",
            "study",
            "rx_type",
            "clinical_scene",
            "condition",
            "rx_x_mm",
            "rx_depth_mm",
            "rx_pair_gap_mm",
            "rx_pad_x_mm",
            "rx_pad_y_mm",
            "rx_pad_z_mm",
            "skin_material_key",
            "rx_lift_mm",
            "fibrosis_shell_mm",
            "contact_layer_mm",
            "stress_mechanism",
            "frequencies_MHz",
            "note",
        ],
    )
    write_csv(out_dir / "analysis_frequency_plan.csv", analysis_frequency_rows(), ["analysis_group", "label", "freq_MHz"])
    write_csv(out_dir / "solver_frequency_plan.csv", solver_frequency_rows(), ["analysis_group", "label", "freq_MHz", "solver_note"])
    write_csv(out_dir / "material_table.csv", material_rows(), ["material", "epsilon_r", "sigma_S_per_m", "density_kg_per_m3", "note"])
    write_csv(out_dir / "setup_table.csv", setup_rows(), ["item", "setting"])
    write_csv(
        out_dir / "physical_anchor_template.csv",
        physical_anchor_template_rows(),
        ["scenario", "label", "freq_MHz", "path_loss_dB", "P_rx_W_at_1mW", "Q_p95_W_per_m3", "literature_anchor"],
    )
    write_csv(
        out_dir / "sar_chi_correlation_template.csv",
        sar_chi_template_rows(),
        [
            "label",
            "chi_norm",
            "SAR_1g_W_per_kg_hub_hotspot",
            "SAR_10g_W_per_kg_hub_hotspot",
            "SAR_1g_W_per_kg_rx_hotspot",
            "SAR_10g_W_per_kg_rx_hotspot",
            "chi_sar_rank_consistent",
        ],
    )
    write_csv(
        out_dir / "lit_cross_validation_template.csv",
        lit_cross_validation_rows(),
        ["source_key", "published_quantity", "hfss_case", "acceptance_band", "status"],
    )
    (out_dir / "torso_hfss_design_review.md").write_text(design_review_text(), encoding="utf-8")


def comma_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, run, or prepare the reviewer-driven chest/torso HFSS experiment package.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manifest-only", action="store_true", help="Write manifests and template tables without opening AEDT.")
    mode.add_argument("--build-only", action="store_true", help="Build selected AEDT designs and save the project without solving.")
    mode.add_argument("--run", action="store_true", help="Build selected AEDT designs, solve them, and export raw S-parameter/field outputs.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for manifests and later solved tables.")
    parser.add_argument("--graphical", action="store_true", help="Open AEDT with a graphical desktop instead of non-graphical mode.")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing generated AEDT/output artifacts when building.")
    parser.add_argument("--studies", help="Comma-separated studies to build/run, e.g. A_main_library,C_distance_scan.")
    parser.add_argument("--case-labels", help="Comma-separated case labels to build/run, e.g. t_surf_rest,t_imp30_stress.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    studies = comma_list(args.studies)
    case_labels = comma_list(args.case_labels)
    cases = select_cases(studies=studies, case_labels=case_labels)
    write_all_manifests(args.out_dir)
    print(f"Wrote torso HFSS manifests to {args.out_dir}")
    print(f"Unique solver geometries: {len(unique_cases())}")
    print(f"Reviewer-analysis frequency requests: {len(analysis_frequency_rows())}")
    print(f"Deduplicated solver frequency rows: {len(solver_frequency_rows())}")
    if not args.build_only and not args.run:
        print("HFSS AEDT build/run is intentionally not started by default. Use --build-only or --run for gated execution.")
        return
    try:
        hfss, geometry = build_project(cases, non_graphical=not args.graphical, overwrite=not args.keep_existing)
        if args.build_only:
            write_run_summary("built", cases, out_dir=args.out_dir, detail="AEDT project built but not solved.")
            print(f"Built torso HFSS AEDT project: {PROJECT_PATH}")
            return
        sparam_rows, field_rows = run_and_export(hfss, cases, geometry, out_dir=args.out_dir)
        write_raw_outputs(sparam_rows, field_rows, out_dir=args.out_dir)
        write_run_summary("completed", cases, out_dir=args.out_dir, detail=f"Exported {len(sparam_rows)} S-parameter rows and {len(field_rows)} field rows.")
        print(f"Exported {len(sparam_rows)} S-parameter rows and {len(field_rows)} field rows to {args.out_dir}")
    except Exception:
        detail = traceback.format_exc()
        write_run_summary("failed", cases, out_dir=args.out_dir, detail=detail)
        raise


if __name__ == "__main__":
    main()
