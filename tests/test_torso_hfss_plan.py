from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "hfss" / "torso_main" / "scripts" / "run_torso_hfss.py"


def load_module():
    spec = importlib.util.spec_from_file_location("torso_hfss", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_solver_plan_deduplicates_review_frequency_requests():
    mod = load_module()

    analysis_rows = mod.analysis_frequency_rows()
    solver_rows = mod.solver_frequency_rows()

    assert len(mod.unique_cases()) == 21
    assert len(analysis_rows) == 57
    assert len(solver_rows) == 51
    assert len({(row["label"], row["freq_MHz"]) for row in solver_rows}) == 51


def test_main_states_are_clinically_motivated_torso_labels():
    mod = load_module()

    cases = {case.label: case for case in mod.MAIN_LIBRARY_CASES}

    assert list(cases) == [
        "t_surf_rest",
        "t_surf_sweat",
        "t_surf_loose",
        "t_surf_fail",
        "t_imp10_rest",
        "t_imp10_stress",
        "t_imp30_rest",
        "t_imp30_stress",
    ]
    assert cases["t_imp10_stress"].stress_mechanism == "fibrotic_encapsulation"
    assert cases["t_imp30_stress"].stress_mechanism == "fibrotic_encapsulation"
    assert cases["t_imp10_rest"].rx_x_mm == cases["t_imp10_stress"].rx_x_mm == 50.0
    assert cases["t_imp30_rest"].rx_x_mm == cases["t_imp30_stress"].rx_x_mm == 50.0
    assert cases["t_surf_loose"].contact_layer_mm == 0.3
    assert cases["t_surf_fail"].rx_lift_mm == 0.5


def test_torso_materials_and_solver_settings_capture_review_design():
    mod = load_module()

    assert mod.TORSO_RADIUS_MM == 150.0
    assert mod.TORSO_LENGTH_MM == 300.0
    assert mod.SKIN_THICKNESS_MM == 2.0
    assert mod.FAT_THICKNESS_MM == 5.0
    assert mod.MUSCLE_RADIUS_MM == 143.0
    assert mod.AIR_MARGIN_MM == 300.0
    assert mod.MATERIALS["skin_sweat"].conductivity == 1.10
    assert mod.MATERIALS["fibrosis"].conductivity == 0.20
    assert mod.MATERIALS["fibrosis"].permittivity == 60.0
    assert mod.SOLVER_SETTINGS["max_delta_s"] == 0.01
    assert mod.SOLVER_SETTINGS["max_passes"] == 8
    assert mod.SOLVER_SETTINGS["incident_power_w"] == 1.0
    assert math.isclose(mod.air_wavelength_fraction_at_40mhz(), 0.04, rel_tol=0.02)


def test_manifest_writer_outputs_required_review_tables(tmp_path):
    mod = load_module()

    mod.write_all_manifests(tmp_path)

    expected = {
        "case_manifest.csv",
        "analysis_frequency_plan.csv",
        "solver_frequency_plan.csv",
        "material_table.csv",
        "setup_table.csv",
        "physical_anchor_template.csv",
        "sar_chi_correlation_template.csv",
        "lit_cross_validation_template.csv",
        "torso_hfss_design_review.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert len(read_csv(tmp_path / "case_manifest.csv")) == 21
    assert len(read_csv(tmp_path / "analysis_frequency_plan.csv")) == 57
    assert len(read_csv(tmp_path / "solver_frequency_plan.csv")) == 51


def test_torso_layers_are_nested_without_overlap():
    mod = load_module()

    layers = mod.torso_layers()

    assert [(layer.name, layer.inner_radius_mm, layer.outer_radius_mm) for layer in layers] == [
        ("muscle_core", 0.0, 143.0),
        ("fat_shell", 143.0, 148.0),
        ("skin_shell", 148.0, 150.0),
    ]
    assert all(layer.inner_radius_mm < layer.outer_radius_mm for layer in layers)


def test_surface_loose_and_contact_failure_are_distinct_cases():
    mod = load_module()

    loose = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_surf_loose")
    failure = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_surf_fail")

    assert loose.rx_lift_mm == 0.0
    assert loose.contact_layer_mm > 0.0
    assert failure.rx_lift_mm == 0.50
    assert loose.condition == "surface_loose"
    assert failure.condition == "surface_contact_failure"


def test_implant_host_layers_follow_receiver_depth():
    mod = load_module()

    imp10 = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp10_rest")
    imp30 = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp30_rest")

    assert mod.host_tissue_layers_for_implant(imp10) == ["muscle_core"]
    assert mod.host_tissue_layers_for_implant(imp30) == ["muscle_core"]


def test_fibrotic_stress_geometry_adds_capsule_without_position_shift():
    mod = load_module()

    class FakeObject:
        def __init__(self, name: str, material: str = ""):
            self.name = name
            self.material = material
            self.model = True

    class FakeModeler:
        def __init__(self):
            self.model_units = "mm"
            self.boxes = []
            self.cylinders = []
            self.regions = []
            self.subtractions = []

        def create_box(self, origin, sizes, name: str, material: str):
            self.boxes.append({"origin": origin, "sizes": sizes, "name": name, "material": material})
            return FakeObject(name, material)

        def create_cylinder(self, *args, name: str, material: str, **kwargs):
            self.cylinders.append({"name": name, "material": material})
            return FakeObject(name, material)

        def create_rectangle(self, *args, name: str, **kwargs):
            return FakeObject(name)

        def create_region(self, pad_value, pad_type: str):
            self.regions.append((list(pad_value), pad_type))
            return FakeObject("Region", "air")

        def subtract(self, blank_list, tool_list, keep_originals=False):
            self.subtractions.append((list(blank_list), list(tool_list), keep_originals))
            return True

    class FakeHfss:
        def __init__(self):
            self.modeler = FakeModeler()
            self.boundaries = []
            self.ports = []

        def assign_radiation_boundary_to_objects(self, *args, **kwargs):
            self.boundaries.append((args, kwargs))
            return True

        def lumped_port(self, *args, **kwargs):
            self.ports.append((args, kwargs))
            return True

    hfss = FakeHfss()
    rest = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp30_rest")
    stress = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp30_stress")

    geometry = mod.create_case_geometry(hfss, stress)

    assert rest.rx_x_mm == stress.rx_x_mm == 50.0
    assert geometry["rx_z_mm"] == mod.receiver_z_for_case(stress)
    assert any(box["name"] == "fibrosis_capsule" and box["material"] == "hbc_fibrosis_proxy" for box in hfss.modeler.boxes)
    assert hfss.modeler.regions == [(["300mm"] * 6, "Absolute Offset")]
    assert hfss.boundaries[0][0][0].name == "Region"
    assert (["muscle_core"], ["fibrosis_capsule"], True) in hfss.modeler.subtractions


def test_implant10_geometry_subtracts_receiver_and_capsule_from_muscle():
    mod = load_module()

    class FakeObject:
        def __init__(self, name: str, material: str = ""):
            self.name = name
            self.material = material
            self.model = True

    class FakeModeler:
        def __init__(self):
            self.model_units = "mm"
            self.subtractions = []

        def create_box(self, origin, sizes, name: str, material: str):
            return FakeObject(name, material)

        def create_cylinder(self, *args, name: str, material: str, **kwargs):
            return FakeObject(name, material)

        def create_rectangle(self, *args, name: str, **kwargs):
            return FakeObject(name)

        def create_region(self, pad_value, pad_type: str):
            return FakeObject("Region", "air")

        def subtract(self, blank_list, tool_list, keep_originals=False):
            self.subtractions.append((list(blank_list), list(tool_list), keep_originals))
            return True

    class FakeHfss:
        def __init__(self):
            self.modeler = FakeModeler()

        def assign_radiation_boundary_to_objects(self, *args, **kwargs):
            return True

        def lumped_port(self, *args, **kwargs):
            return True

    hfss = FakeHfss()
    case = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp10_stress")

    mod.create_case_geometry(hfss, case)

    assert (["muscle_core"], ["fibrosis_capsule"], True) in hfss.modeler.subtractions
    assert (["fibrosis_capsule"], ["rx_sig", "rx_ref"], True) in hfss.modeler.subtractions
    assert (["muscle_core"], ["rx_sig", "rx_ref"], True) in hfss.modeler.subtractions


def test_select_cases_filters_by_study_and_labels_and_rejects_unknowns():
    mod = load_module()

    by_study = mod.select_cases(studies=["A_main_library"])
    by_label = mod.select_cases(case_labels=["t_surf_rest", "t_imp30_stress"])

    assert len(by_study) == 8
    assert [case.label for case in by_label] == ["t_surf_rest", "t_imp30_stress"]
    try:
        mod.select_cases(case_labels=["missing_case"])
    except ValueError as exc:
        assert "missing_case" in str(exc)
    else:
        raise AssertionError("unknown label should fail")


def test_electrode_pair_uses_explicit_port_sheet():
    mod = load_module()

    class FakeObject:
        def __init__(self, name: str):
            self.name = name

    class FakeModeler:
        def __init__(self):
            self.rectangles = []

        def create_box(self, origin, sizes, name: str, material: str):
            return FakeObject(name)

        def create_rectangle(self, plane, origin, sizes, name: str):
            self.rectangles.append((plane, origin, sizes, name))
            return FakeObject(name)

    class FakeHfss:
        def __init__(self):
            self.modeler = FakeModeler()
            self.ports = []

        def lumped_port(self, **kwargs):
            self.ports.append(kwargs)
            return True

    hfss = FakeHfss()

    mod.create_electrode_pair(hfss, "rx", 50.0, 149.5, 6.0, 3.0, 0.2, 4.0, "NodePort")

    assert hfss.modeler.rectangles == [("YZ", [50.0, -2.0, 149.4], [4.0, 0.2], "rx_port_sheet")]
    assert hfss.ports[0]["assignment"] == "rx_port_sheet"
    assert hfss.ports[0]["integration_line"] == [[50.0, -2.0, 149.5], [50.0, 2.0, 149.5]]


def test_create_setup_uses_single_point_sweeps_and_saves_energy_fields_only():
    mod = load_module()

    class FakeSetup:
        def __init__(self):
            self.props = {}
            self.updated = False

        def update(self):
            self.updated = True

    class FakeMesh:
        def __init__(self):
            self.length_meshes = []

        def assign_length_mesh(self, objects, **kwargs):
            self.length_meshes.append((list(objects), kwargs))
            return True

    class FakeHfss:
        def __init__(self):
            self.setup = FakeSetup()
            self.sweeps = []
            self.mesh = FakeMesh()

        def create_setup(self, name):
            self.setup.props["Name"] = name
            return self.setup

        def create_single_point_sweep(self, **kwargs):
            self.sweeps.append(kwargs)
            return True

    hfss = FakeHfss()
    case = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_surf_rest")

    mod.create_setup(hfss, case)

    assert hfss.setup.props["Frequency"] == "40.0MHz"
    assert hfss.setup.props["MaximumPasses"] == 8
    assert [sweep["freq"] for sweep in hfss.sweeps] == [1.0, 10.0, 13.56, 21.0, 40.0]
    save_flags = {sweep["freq"]: sweep["save_fields"] for sweep in hfss.sweeps}
    assert save_flags[13.56] is True
    assert save_flags[40.0] is False


def test_sample_points_use_125_point_cube_and_si_export_conversion():
    mod = load_module()
    case = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_imp10_rest")
    rx_z = mod.receiver_z_for_case(case)

    points = mod.sample_points_mm(case, rx_z, side_mm=5.0, n=5)

    assert len(points) == 125
    assert min(point[2] for point in points) < rx_z < max(point[2] for point in points)
    si_points = mod.sample_points_si(case, rx_z, side_mm=5.0, n=5)
    assert math.isclose(si_points[0][0], points[0][0] / 1000.0)


def test_touchstone_parser_handles_db_and_ri_formats(tmp_path):
    mod = load_module()
    db_path = tmp_path / "db.s2p"
    ri_path = tmp_path / "ri.s2p"
    db_path.write_text("# MHz S DB R 50\n13.56 -10 0 -60 30 -60 30 -12 0\n", encoding="utf-8")
    ri_path.write_text("# GHz S RI R 50\n0.04 0 0 0.001 0 0.001 0 0 0\n", encoding="utf-8")

    db_rows = mod.parse_touchstone(db_path)
    ri_rows = mod.parse_touchstone(ri_path)

    assert db_rows[0]["freq_MHz"] == 13.56
    assert math.isclose(db_rows[0]["S21_dB"], -60.0)
    assert math.isclose(db_rows[0]["S21_mag2"], 1.0e-6, rel_tol=1e-9)
    assert ri_rows[0]["freq_MHz"] == 40.0
    assert math.isclose(ri_rows[0]["S21_dB"], -60.0, abs_tol=1e-9)


def test_field_parser_and_stats_use_local_material():
    mod = load_module()

    case = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_surf_sweat")
    stats = mod.field_stats_from_magnitudes(case, [1.0, 2.0, 3.0, 4.0])

    assert stats["Ep95_sampled_V_per_m"] == 4.0
    assert math.isclose(stats["Q_p95_W_per_m3"], 1.10 * 16.0)


def test_run_and_export_rejects_empty_touchstone(tmp_path):
    mod = load_module()

    class FakeHfss:
        def __init__(self):
            self.active = []

        def set_active_design(self, name):
            self.active.append(name)

        def analyze_setup(self, *args, **kwargs):
            return True

        def export_touchstone(self, **kwargs):
            Path(kwargs["output_file"]).write_text("", encoding="utf-8")
            return True

    case = next(case for case in mod.MAIN_LIBRARY_CASES if case.label == "t_surf_rest")
    try:
        mod.run_and_export(FakeHfss(), [case], {case.label: {"rx_z_mm": mod.receiver_z_for_case(case)}}, out_dir=tmp_path)
    except RuntimeError as exc:
        assert "Touchstone parse failed" in str(exc)
    else:
        raise AssertionError("empty touchstone should fail")


def test_cli_accepts_build_and_case_filters():
    mod = load_module()

    args = mod.parse_args(["--build-only", "--case-labels", "t_surf_rest,t_imp30_stress", "--studies", "A_main_library"])

    assert args.build_only is True
    assert args.case_labels == "t_surf_rest,t_imp30_stress"
    assert args.studies == "A_main_library"


def test_build_project_keep_existing_reuses_design_without_rebuilding(tmp_path, monkeypatch):
    mod = load_module()
    project_path = tmp_path / "hbc_torso_main.aedt"
    project_path.write_text("existing", encoding="utf-8")

    class FakeHfss:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active_designs = []
            self.saved = False
            FakeHfss.instances.append(self)

        def set_active_design(self, name):
            self.active_designs.append(name)
            return True

        def save_project(self, *args, **kwargs):
            self.saved = True

    def fail_if_rebuilding(*args, **kwargs):
        raise AssertionError("existing project reuse must not recreate geometry")

    monkeypatch.setattr(mod, "PROJECT_PATH", project_path)
    monkeypatch.setattr(mod, "require_hfss", lambda: FakeHfss)
    monkeypatch.setattr(mod, "create_case_geometry", fail_if_rebuilding)

    cases = mod.select_cases(case_labels=["t_surf_rest"])
    hfss, geometry = mod.build_project(cases, overwrite=False)

    assert hfss.kwargs["project"] == str(project_path)
    assert hfss.kwargs["new_desktop"] is True
    assert hfss.active_designs == ["t_surf_rest"]
    assert geometry["t_surf_rest"]["rx_z_mm"] == mod.receiver_z_for_case(cases[0])
