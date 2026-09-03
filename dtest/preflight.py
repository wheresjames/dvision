"""Dependency preflight for deterministic and process vision tests.

The supported environment is pinned in ``requirements-visiontests.txt``. The
groups are listed separately because a shell that cannot render or cannot open
shared memory may skip only those groups — the deterministic physics and
coordinate contract tests must still run.
"""

from __future__ import annotations

import importlib
import sys


# Needed by the coordinate/physics contract group. These may never be skipped.
DETERMINISTIC_MODULES = ("numpy", "pytest", "PIL")

# Needed to render the calibration scene and run the client frame adapters.
RENDERING_MODULES = ("panda3d", "cv2")

# Needed to launch real processes over the shared-memory transports.
PROCESS_MODULES = ("pymembus",)

REQUIRED_MODULES = DETERMINISTIC_MODULES + RENDERING_MODULES + PROCESS_MODULES

GROUPS = {
    "deterministic": DETERMINISTIC_MODULES,
    "rendering": RENDERING_MODULES,
    "process": PROCESS_MODULES,
}


def missing_dependencies(modules: tuple[str, ...] = REQUIRED_MODULES) -> list[str]:
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    return missing


def main() -> int:
    failed = False
    for group, modules in GROUPS.items():
        missing = missing_dependencies(modules)
        if missing:
            failed = True
            print(f"{group}: missing {', '.join(missing)}", file=sys.stderr)
        else:
            print(f"{group}: ok")
    if failed:
        print(
            "install requirements-visiontests.txt to complete the supported "
            "vision-test environment",
            file=sys.stderr,
        )
        return 1
    print("vision-test preflight: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
