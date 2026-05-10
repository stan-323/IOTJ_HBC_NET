import os
import sys

ROOT = os.environ.get("HBC_HFSS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import ScriptEnv
    ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
    oDesktop = ScriptEnv.GetDesktop()
    version = oDesktop.GetVersion()
    with open(os.path.join(OUT_DIR, "aedt_probe.txt"), "w") as f:
        f.write("AEDT initialized\n")
        f.write("version=%s\n" % version)
    oDesktop.QuitApplication()
except Exception as exc:
    with open(os.path.join(OUT_DIR, "aedt_probe_error.txt"), "w") as f:
        f.write("ERROR\n")
        f.write(str(exc))
    raise
