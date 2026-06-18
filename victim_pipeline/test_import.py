import os, importlib.util, traceback
path = os.path.join(os.path.dirname(__file__), "agents.py")
spec = importlib.util.spec_from_file_location("agents", path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print("Import success")
except Exception as e:
    print("Import failed:", e)
    traceback.print_exc()
