from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ansys.aedt.core import Hfss
from ansys.aedt.core.generic.constants import GRAVITY, PLANE


ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs"
SNP = OUT / "touchstone"
IMG = OUT / "images"
LOG = OUT / "hfss_run.log"
PROJECT = ROOT / "hbc_tissue_block_calibration.aedt"


@dataclass(frozen=True)
class Tissue:
    eps_r: float
    sigma: float
    rho: float


@dataclass(frozen=True)
class Case:
    case_id: int
    node_class: str
    depth_mm: float
    regime: str


FREQS = {
    "FD": {"freq_mhz": 40.0, "label": "f_D", "source_note": "40.68 MHz tissue constants used as 40 MHz proxy"},
    "FE": {"freq_mhz": 13.56, "label": "f_E", "source_note": "13.56 MHz tissue constants"},
}

# Three-layer tissue constants from the literature table at 13.56 and 40.68 MHz.
# Densities are standard scheduling-proxy values used only for sigma|E|^2/rho normalization.
TISSUES = {
    "FE": {
        "skin": Tissue(285.25, 0.23802, 1109.0),
        "wet_skin": Tissue(177.13, 0.38421, 1109.0),
        "fat": Tissue(11.827, 0.030354, 911.0),
        "muscle": Tissue(138.44, 0.62818, 1090.0),
    },
    "FD": {
        "skin": Tissue(122.91, 0.37982, 1109.0),
        "wet_skin": Tissue(92.985, 0.45519, 1109.0),
        "fat": Tissue(7.286, 0.034136, 911.0),
        "muscle": Tissue(82.115, 0.66986, 1090.0),
    },
}

CASES = [
    Case(1, "surface", 0.0, "Rest"),
    Case(2, "implant", 10.0, "Rest"),
    Case(3, "implant", 30.0, "Rest"),
    Case(4, "surface", 0.0, "Sweat"),
    Case(5, "implant", 10.0, "Sweat"),
    Case(6, "implant", 30.0, "Sweat"),
    Case(7, "surface", 0.0, "Loose"),
    Case(8, "implant", 10.0, "Stress"),
    Case(9, "implant", 30.0, "Stress"),
]


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def add_material(hfss: Hfss, name: str, tissue: Tissue, color: tuple[int, int, int]) -> None:
    mat = hfss.materials.add_material(name)
    mat.permittivity = tissue.eps_r
    mat.conductivity = tissue.sigma
    try:
        mat.mass_density = tissue.rho
    except Exception:
        pass
    try:
        mat.appearance = color
    except Exception:
        pass


def setup_materials(hfss: Hfss, freq_key: str) -> dict[str, str]:
    names = {}
    palette = {
        "skin": (230, 160, 150),
        "wet_skin": (120, 180, 255),
        "fat": (245, 220, 130),
        "muscle": (200, 80, 80),
    }
    for key, tissue in TISSUES[freq_key].items():
        name = f"hbc_{freq_key}_{key}"
        add_material(hfss, name, tissue, palette[key])
        names[key] = name
    capsule = hfss.materials.add_material(f"hbc_{freq_key}_bio_capsule")
    capsule.permittivity = 3.2
    capsule.conductivity = 0.0
    names["capsule"] = capsule.name
    return names


def cylinder_pair(
    hfss: Hfss,
    prefix: str,
    orientation: int,
    centers: list[list[float]],
    radius: float,
    thickness: float,
    material: str = "copper",
) -> tuple[str, str]:
    objs = []
    for idx, center in enumerate(centers):
        origin = list(center)
        if orientation == 0:  # X axis
            origin[0] -= thickness / 2
        elif orientation == 1:  # Y axis
            origin[1] -= thickness / 2
        else:  # Z axis
            origin[2] -= thickness / 2
        obj = hfss.modeler.create_cylinder(
            orientation=orientation,
            origin=origin,
            radius=radius,
            height=thickness,
            num_sides=32,
            name=f"{prefix}_{idx + 1}",
            material=material,
        )
        objs.append(obj.name)
    return objs[0], objs[1]


def assign_lumped_sheet_port(
    hfss: Hfss,
    sheet_name: str,
    plane: int,
    origin: list[float],
    sizes: list[float],
    start: list[float],
    end: list[float],
    port_name: str,
) -> None:
    sheet = hfss.modeler.create_rectangle(
        orientation=plane,
        origin=origin,
        sizes=sizes,
        name=sheet_name,
        material="vacuum",
        is_covered=True,
    )
    hfss.lumped_port(
        assignment=sheet.name,
        create_port_sheet=False,
        integration_line=[start, end],
        impedance=50,
        name=port_name,
        renormalize=True,
    )


def create_tissue_block(hfss: Hfss, mats: dict[str, str], case: Case) -> None:
    hfss.modeler.create_box([-60, -50, 0], [120, 100, 2], name="skin_2mm", material=mats["skin"])
    hfss.modeler.create_box([-60, -50, 2], [120, 100, 8], name="fat_8mm", material=mats["fat"])
    hfss.modeler.create_box([-60, -50, 10], [120, 100, 50], name="muscle_50mm", material=mats["muscle"])
    if case.regime == "Sweat":
        hfss.modeler.create_box([-60, -50, -0.2], [120, 100, 0.2], name="wet_surface_layer_0p2mm", material=mats["wet_skin"])
    region = hfss.modeler.create_region(["100mm", "100mm", "80mm", "100mm", "100mm", "80mm"], pad_type="Absolute Offset")
    hfss.assign_radiation_boundary_to_objects(region, name="Radiation_Region")


def create_electrodes_and_ports(hfss: Hfss, case: Case) -> str:
    # Hub electrodes: circular copper plates on or above skin surface.
    if case.regime == "Sweat":
        hub_z = -0.3
    elif case.node_class == "surface" and case.regime == "Loose":
        hub_z = -1.0
    else:
        hub_z = -0.1
    hub_a, hub_b = cylinder_pair(
        hfss,
        "hub_electrode",
        orientation=2,
        centers=[[0, -10, hub_z], [0, 10, hub_z]],
        radius=8.0,
        thickness=0.2,
    )
    assign_lumped_sheet_port(
        hfss,
        sheet_name="PortSheet_Hub",
        plane=PLANE.XY,
        origin=[-2.0, -2.0, hub_z],
        sizes=[4.0, 4.0],
        start=[0.0, -2.0, hub_z],
        end=[0.0, 2.0, hub_z],
        port_name="Port_Hub",
    )

    # Receiver electrodes.
    if case.node_class == "surface":
        if case.regime == "Sweat":
            rx_z = -0.3
        elif case.regime == "Loose":
            rx_z = -1.0
        else:
            rx_z = -0.1
        rx_a, rx_b = cylinder_pair(
            hfss,
            "rx_surface_electrode",
            orientation=2,
            centers=[[50, -6, rx_z], [50, 6, rx_z]],
            radius=5.0,
            thickness=0.2,
        )
        obs = hfss.modeler.create_box([45, -5, 0], [10, 10, 5], name="Omega_surface_10mm", material="vacuum")
        obs.model = False
        assign_lumped_sheet_port(
            hfss,
            sheet_name="PortSheet_Rx",
            plane=PLANE.XY,
            origin=[48.0, -1.0, rx_z],
            sizes=[4.0, 2.0],
            start=[50.0, -1.0, rx_z],
            end=[50.0, 1.0, rx_z],
            port_name="Port_Rx",
        )
    else:
        d = case.depth_mm
        if case.regime == "Stress":
            # Implant stress proxy: electrode pair rotated toward the hub path and effective radius reduced.
            rx_a, rx_b = cylinder_pair(
                hfss,
                f"rx_implant_{int(d)}mm_stress",
                orientation=0,
                centers=[[50 - 2.5, 0, d], [50 + 2.5, 0, d]],
                radius=1.0,
                thickness=0.2,
            )
            assign_lumped_sheet_port(
                hfss,
                sheet_name="PortSheet_Rx",
                plane=PLANE.ZX,
                origin=[50.0 - 2.4, 0.0, d - 1.2],
                sizes=[2.4, 4.8],
                start=[50.0 - 2.4, 0.0, d],
                end=[50.0 + 2.4, 0.0, d],
                port_name="Port_Rx",
            )
        else:
            rx_a, rx_b = cylinder_pair(
                hfss,
                f"rx_implant_{int(d)}mm",
                orientation=1,
                centers=[[50, -2.5, d], [50, 2.5, d]],
                radius=1.5,
                thickness=0.2,
            )
            assign_lumped_sheet_port(
                hfss,
                sheet_name="PortSheet_Rx",
                plane=PLANE.YZ,
                origin=[50.0, -2.4, d - 1.5],
                sizes=[4.8, 3.0],
                start=[50.0, -2.4, d],
                end=[50.0, 2.4, d],
                port_name="Port_Rx",
            )
        obs = hfss.modeler.create_box([45, -5, d - 5], [10, 10, 10], name=f"Omega_implant_{int(d)}mm", material="vacuum")
        obs.model = False

    return obs.name


def setup_solution(hfss: Hfss, freq_mhz: float) -> None:
    setup = hfss.create_setup("Setup1")
    setup.props["Frequency"] = f"{freq_mhz}MHz"
    setup.props["MaxDeltaS"] = 0.05
    setup.props["MaximumPasses"] = 3
    setup.props["MinimumPasses"] = 1
    setup.props["MinimumConvergedPasses"] = 1
    setup.props["PercentRefinement"] = 20
    setup.update()
    try:
        hfss.mesh.assign_initial_mesh_from_slider(level=4)
        hfss.mesh.assign_length_mesh(
            ["hub_electrode_1", "hub_electrode_2"],
            inside_selection=True,
            maximum_length=1,
            name="mesh_hub_1mm",
        )
    except Exception as exc:
        log(f"Mesh setup warning in {hfss.design_name}: {exc}")


def build_design(hfss: Hfss, case: Case, freq_key: str) -> str:
    mats = setup_materials(hfss, freq_key)
    create_tissue_block(hfss, mats, case)
    obs_name = create_electrodes_and_ports(hfss, case)
    setup_solution(hfss, FREQS[freq_key]["freq_mhz"])
    return obs_name


def parse_touchstone(path: Path) -> dict[str, float]:
    # Supports standard MA/DB/RI single-frequency .s2p export. Returns S11/S21/S22 in dB.
    fmt = "MA"
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line.upper().split()
                if "DB" in parts:
                    fmt = "DB"
                elif "RI" in parts:
                    fmt = "RI"
                else:
                    fmt = "MA"
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) >= 9:
                rows.append(vals)
    if not rows:
        raise RuntimeError(f"No numeric rows found in {path}")
    vals = rows[0]
    # Touchstone order: f, S11, S21, S12, S22, with two columns per S parameter.
    def pair_to_db(a: float, b: float) -> float:
        if fmt == "DB":
            return a
        if fmt == "RI":
            mag = math.hypot(a, b)
        else:
            mag = abs(a)
        return 20.0 * math.log10(max(mag, 1e-300))

    return {
        "s11_db": pair_to_db(vals[1], vals[2]),
        "s21_db": pair_to_db(vals[3], vals[4]),
        "s12_db": pair_to_db(vals[5], vals[6]),
        "s22_db": pair_to_db(vals[7], vals[8]),
    }


def field_proxy(hfss: Hfss, obs_name: str, case: Case, freq_key: str) -> tuple[float, float]:
    if freq_key != "FE":
        return float("nan"), float("nan")
    try:
        intrinsics = {"Freq": f"{FREQS[freq_key]['freq_mhz']}MHz", "Phase": "0deg"}
        emax = hfss.post.get_scalar_field_value(
            quantity="Mag_E",
            scalar_function="Maximum",
            solution="Setup1 : LastAdaptive",
            intrinsics=intrinsics,
            object_name=obs_name,
            object_type="Volume",
            is_vector=False,
        )
        if case.node_class == "surface":
            tissue_key = "wet_skin" if case.regime == "Sweat" else "skin"
        else:
            tissue_key = "muscle"
        t = TISSUES[freq_key][tissue_key]
        q = t.sigma * float(emax) ** 2 / t.rho
        return float(emax), float(q)
    except Exception as exc:
        log(f"Field proxy extraction warning in {hfss.design_name}: {exc}")
        return float("nan"), float("nan")


def write_setup_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "hfss_setup_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "setting"])
        w.writerow(["model", "120 mm x 100 mm x 60 mm three-layer tissue block"])
        w.writerow(["layers", "skin 2 mm / fat 8 mm / muscle 50 mm"])
        w.writerow(["frequencies", "f_D=40 MHz, f_E=13.56 MHz"])
        w.writerow(["ports", "two differential lumped ports, 50 ohm reference"])
        w.writerow(["hub electrodes", "two circular copper plates, radius 8 mm, spacing 20 mm, thickness 0.2 mm"])
        w.writerow(["surface receiver", "two circular copper plates, radius 5 mm, spacing 12 mm, x=50 mm"])
        w.writerow(["implant receiver", "two circular copper plates, radius 1.5 mm, spacing 5 mm, depths 10/30 mm"])
        w.writerow(["sweat", "0.2 mm wet/high-conductivity surface layer"])
        w.writerow(["loose surface contact", "0.8 mm air gap under surface electrodes"])
        w.writerow(["implant stress", "receiver electrode pair rotated 90 degrees and reduced to radius 1.0 mm"])
        w.writerow(["boundary", "air region with radiation boundary, 80-100 mm absolute padding"])
        w.writerow(["mesh", "adaptive mesh, max Delta S 0.05, up to 3 passes, local hub length mesh 1 mm"])
        w.writerow(["outputs", "S11/S21/S22 touchstone, matched-port S21 proxy, local max |E| proxy at f_E"])

    with (OUT / "tissue_materials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["freq_key", "frequency_MHz", "tissue", "relative_permittivity", "conductivity_S_per_m", "density_kg_per_m3", "source_note"])
        for fk, data in TISSUES.items():
            for tissue, props in data.items():
                w.writerow([fk, FREQS[fk]["freq_mhz"], tissue, props.eps_r, props.sigma, props.rho, FREQS[fk]["source_note"]])


def postprocess(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sparams_raw.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case_id",
            "design",
            "node_class",
            "depth_mm",
            "regime",
            "freq_key",
            "frequency_MHz",
            "s11_db",
            "s21_db",
            "s12_db",
            "s22_db",
            "path_loss_db",
            "touchstone_file",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    field_rows = [r for r in rows if r["freq_key"] == "FE"]
    with (OUT / "field_proxy.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case_id",
            "design",
            "node_class",
            "depth_mm",
            "regime",
            "frequency_MHz",
            "emax_v_per_m",
            "q_proxy_sigma_e2_over_rho",
            "q_note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in field_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    by_key = {(r["node_class"], float(r["depth_mm"]), r["regime"], r["freq_key"]): r for r in rows}
    base_fd = by_key[("surface", 0.0, "Rest", "FD")]
    base_fe = by_key[("surface", 0.0, "Rest", "FE")]
    base_g = 10 ** (float(base_fe["s21_db"]) / 10.0)
    base_r = 10 ** (float(base_fd["s21_db"]) / 10.0)
    base_q = float(base_fe.get("q_proxy_sigma_e2_over_rho", float("nan")))
    base_eta = base_q / max(base_g, 1e-300)

    library = []
    for case in CASES:
        fd = by_key[(case.node_class, case.depth_mm, case.regime, "FD")]
        fe = by_key[(case.node_class, case.depth_mm, case.regime, "FE")]
        gd = 10 ** (float(fe["s21_db"]) / 10.0)
        rd = 10 ** (float(fd["s21_db"]) / 10.0)
        q = float(fe.get("q_proxy_sigma_e2_over_rho", float("nan")))
        eta = q / max(gd, 1e-300) if math.isfinite(q) else float("nan")
        library.append(
            {
                "case_id": case.case_id,
                "node_class": case.node_class,
                "depth_mm": case.depth_mm,
                "regime": case.regime,
                "s21_fd_db": fd["s21_db"],
                "s21_fe_db": fe["s21_db"],
                "pl_fd_db": -float(fd["s21_db"]),
                "pl_fe_db": -float(fe["s21_db"]),
                "kappa_r": rd / max(base_r, 1e-300),
                "kappa_g": gd / max(base_g, 1e-300),
                "kappa_chi": q / base_q if math.isfinite(q) and math.isfinite(base_q) and base_q > 0 else float("nan"),
                "kappa_p": base_eta / eta if math.isfinite(eta) and eta > 0 and math.isfinite(base_eta) else float("nan"),
                "data_source": "HFSS",
            }
        )

    with (OUT / "calibrated_library.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case_id",
            "node_class",
            "depth_mm",
            "regime",
            "s21_fd_db",
            "s21_fe_db",
            "pl_fd_db",
            "pl_fe_db",
            "kappa_r",
            "kappa_g",
            "kappa_chi",
            "kappa_p",
            "data_source",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in library:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SNP.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    if LOG.exists() and args.overwrite:
        LOG.unlink()

    write_setup_tables()
    log("Starting visual AEDT/HFSS calibration run")
    log(f"Project: {PROJECT}")

    if PROJECT.exists() and args.overwrite:
        # AEDT stores sidecar result folders; remove only the project file here.
        PROJECT.unlink()
        results_dir = Path(str(PROJECT) + "results")
        pyaedt_dir = PROJECT.with_suffix(".pyaedt")
        for path in (results_dir, pyaedt_dir):
            if path.exists() and path.is_dir() and str(path).startswith(str(ROOT)):
                import shutil

                shutil.rmtree(path, ignore_errors=True)

    cases = CASES[: args.limit_cases] if args.limit_cases else CASES
    rows: list[dict[str, object]] = []
    first_hfss = None

    for case in cases:
        for freq_key in ("FD", "FE"):
            freq = FREQS[freq_key]["freq_mhz"]
            design = f"C{case.case_id:02d}_{case.node_class}_{int(case.depth_mm)}mm_{case.regime}_{freq_key}"
            log(f"Building {design}")
            hfss = Hfss(
                project=str(PROJECT),
                design=design,
                solution_type="DrivenTerminal",
                version="2024.1",
                non_graphical=False,
                new_desktop=first_hfss is None,
                close_on_exit=False,
                remove_lock=True,
            )
            if first_hfss is None:
                first_hfss = hfss
            hfss.modeler.model_units = "mm"
            obs_name = build_design(hfss, case, freq_key)
            try:
                hfss.post.export_model_picture(
                    full_name=str(IMG / f"{design}_model.jpg"),
                    orientation="isometric",
                    width=1600,
                    height=1000,
                )
            except Exception as exc:
                log(f"Model image warning for {design}: {exc}")
            hfss.save_project()

            if args.build_only:
                continue

            log(f"Solving {design} at {freq} MHz")
            solved = hfss.analyze_setup("Setup1", cores=4, tasks=4, blocking=True)
            log(f"Solve status {design}: {solved}")
            if not solved:
                raise RuntimeError(f"HFSS solve failed for {design}")

            s2p = SNP / f"{design}.s2p"
            exported = hfss.export_touchstone(setup="Setup1", output_file=str(s2p), renormalization=True, impedance=50)
            log(f"Touchstone export {design}: {exported}")
            vals = parse_touchstone(s2p)
            emax, qproxy = field_proxy(hfss, obs_name, case, freq_key)
            rows.append(
                {
                    "case_id": case.case_id,
                    "design": design,
                    "node_class": case.node_class,
                    "depth_mm": case.depth_mm,
                    "regime": case.regime,
                    "freq_key": freq_key,
                    "frequency_MHz": freq,
                    "s11_db": vals["s11_db"],
                    "s21_db": vals["s21_db"],
                    "s12_db": vals["s12_db"],
                    "s22_db": vals["s22_db"],
                    "path_loss_db": -vals["s21_db"],
                    "touchstone_file": str(s2p),
                    "emax_v_per_m": emax,
                    "q_proxy_sigma_e2_over_rho": qproxy,
                    "q_note": "sigma*max(|E|)^2/rho in local Omega, scheduling proxy only",
                }
            )
            hfss.save_project()

    if rows:
        postprocess(rows)
        log("Post-processing complete")
        log(f"Wrote {OUT / 'sparams_raw.csv'}")
        log(f"Wrote {OUT / 'field_proxy.csv'}")
        log(f"Wrote {OUT / 'calibrated_library.csv'}")
    else:
        log("Build-only mode complete")
    log("HFSS calibration script finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
