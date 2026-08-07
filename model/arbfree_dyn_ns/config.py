# ---------------------------------------------------------------------------
# VENDORED FILE — ADAPTED (the ONLY model file we modify vs upstream).
#
# Upstream: snejens/arbfree_dynamic_nelson_siegel @ 2a402a8 (2024-05-19).
# The original is preserved verbatim next to this file as
# `config_upstream.py.orig`. All other files in this package are byte-for-byte
# upstream.
#
# Why this file is patched: the upstream config hardcodes an author-specific
# absolute path ("/home/jens/Nextcloud/Documents/GBS/Thesis") and calls
# logging.basicConfig() with a log file under that path at import time, which
# raises FileNotFoundError on any other machine and makes the whole package
# unimportable. This file contains NO model math — only environment paths and
# logging setup — so adapting it does not alter model behaviour.
#
# Changes vs upstream:
#   * BASE_PATH is read from the ARBFREE_BASE_PATH env var, defaulting to a
#     writable per-user cache dir; the directory tree is created if missing.
#   * The joblib Memory cache location is derived from BASE_PATH.
#   * logging.basicConfig() no longer crashes when the target dir is absent.
# Public names (memory, BASE_PATH, PKL_BASE_PATH, DEFAULT_BUBA_PATH,
# MAIN_GRID) are unchanged so downstream imports keep working.
# ---------------------------------------------------------------------------
import logging
import os
from pathlib import Path

import numpy as np
from joblib import Memory

# BASE_PATH: overridable via env var; defaults to a writable per-user location.
BASE_PATH = Path(
    os.environ.get("ARBFREE_BASE_PATH", Path.home() / ".cache" / "arbfree-dyn-ns")
).expanduser()
PKL_BASE_PATH = BASE_PATH / "out-pkl"
DEFAULT_BUBA_PATH = BASE_PATH / "data" / "buba"

# Ensure the directory tree exists so downstream writes / logging succeed.
for _p in (BASE_PATH, PKL_BASE_PATH, DEFAULT_BUBA_PATH):
    _p.mkdir(parents=True, exist_ok=True)

memory = Memory(location=BASE_PATH / "joblib-cache", verbose=0)

MAIN_GRID = np.array([1, 2, 3, 5, 6, 8, 10, 12, 15, 20, 25], dtype=float)

# set up logging to file
logging.basicConfig(
    filename=BASE_PATH / "arbfree-dyn-ns.log",
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger("").addHandler(console)
