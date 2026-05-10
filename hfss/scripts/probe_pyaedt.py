from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from ansys.aedt.core import Hfss


ROOT = Path(os.environ.get("HBC_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    report_path = OUT_DIR / "environment_probe.txt"
    hfss = None
    try:
        hfss = Hfss(
            project=None,
            design="probe_hfss",
            solution_type="Terminal",
            version="2024.1",
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
            remove_lock=True,
        )
        version = getattr(hfss.desktop_class, "aedt_version_id", "unknown")
        report_path.write_text(
            "\n".join(
                [
                    "PyAEDT probe: OK",
                    f"python={sys.executable}",
                    f"aedt_version={version}",
                    f"project={hfss.project_name}",
                    f"design={hfss.design_name}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception:
        report_path.write_text("PyAEDT probe: FAILED\n" + traceback.format_exc(), encoding="utf-8")
        return 1
    finally:
        if hfss:
            hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
