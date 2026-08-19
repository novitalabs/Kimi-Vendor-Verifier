from __future__ import annotations

import sys
from pathlib import Path


SELF_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SELF_TEST_ROOT.parent
for path in (SELF_TEST_ROOT, REPO_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

pytest_plugins = ("tests.conftest", "pytest_artifacts")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live endpoint or captured-service check")
