import os
from pathlib import Path

from ansys.aedt.core import Hfss

ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
ROOT.mkdir(parents=True, exist_ok=True)

hfss = Hfss(
    project=str(ROOT / "pyaedt_smoke.aedt"),
    design="smoke_hfss",
    solution_type="DrivenTerminal",
    version="2024.1",
    non_graphical=False,
    new_desktop=True,
    close_on_exit=False,
)
hfss.modeler.model_units = "mm"
hfss.save_project()
print("SMOKE_OK", hfss.project_file)
