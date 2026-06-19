import os

# Centralized configuration flags
DEBUG = os.getenv("TRUSTTRACE_DEBUG", "false").lower() == "true"
LOG_SQL = DEBUG
LOG_IRS = DEBUG
LOG_TEMPLATE = DEBUG
LOG_CHROMADB = DEBUG
EXPERIMENT_MODE = False
