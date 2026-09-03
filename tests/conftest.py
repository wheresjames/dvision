"""Dependency gating for the vision-test groups.

The supported environment is pinned in ``requirements-visiontests.txt`` and
`python -m dtest.preflight` reports anything missing. When a shell genuinely
lacks rendering or IPC support, only the renderer and process groups may be
skipped — the physics and coordinate contract tests must still run and still
gate a merge.
"""

import importlib

# Test module stem -> the modules it cannot run without.
_MODULE_REQUIREMENTS = {
    "test_dvision_calibration_render": ("panda3d", "cv2"),
    "test_dvision_process_transport": ("panda3d", "cv2", "pymembus"),
    "test_dvision_nightly": ("panda3d", "pymembus"),
    "test_dway_process": ("panda3d", "pymembus"),
    "test_dvision_perception_chain": ("panda3d", "cv2"),
}


def _missing(names: tuple[str, ...]) -> list[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    return missing


collect_ignore = []
_SKIPPED: dict[str, list[str]] = {}

for _stem, _required in _MODULE_REQUIREMENTS.items():
    _gap = _missing(_required)
    if _gap:
        collect_ignore.append(f"{_stem}.py")
        _SKIPPED[_stem] = _gap


def pytest_report_header(config) -> list[str]:
    if not _SKIPPED:
        return []
    return [
        f"vision-test group {stem} not collected: {', '.join(gap)} unavailable; "
        "install requirements-visiontests.txt"
        for stem, gap in _SKIPPED.items()
    ]
