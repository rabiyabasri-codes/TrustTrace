import os
DEBUG = os.getenv("TRUSTTRACE_DEBUG", "0").lower() in ("1", "true", "yes")  # Enable via env var
