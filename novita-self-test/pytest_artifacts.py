from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


_ARTIFACT_PATHS: dict[str, Path] | None = None
_ARTIFACT_BUFFER: list[dict] = []
_STARTED_AT: float | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--artifact-dir",
        default=os.environ.get("K3_ARTIFACT_DIR", ""),
        help="Write cases.jsonl and summary.json to this directory",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _ARTIFACT_PATHS, _STARTED_AT
    raw = str(config.getoption("artifact_dir") or "").strip()
    if not raw:
        return
    root = Path(raw).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_PATHS = {"cases": root / "cases.jsonl", "summary": root / "summary.json"}
    _STARTED_AT = time.time()


def _shorten(value: object, limit: int = 4096) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _ARTIFACT_PATHS is None:
        return
    if report.when == "setup" and not (report.skipped or report.failed):
        return
    if report.when == "teardown" and not report.failed:
        return
    if report.when not in {"setup", "call", "teardown"}:
        return
    _ARTIFACT_BUFFER.append(
        {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "phase": report.when,
            "duration_s": round(report.duration, 6),
            "longrepr": _shorten(getattr(report, "longrepr", None)),
            "caplog": _shorten(getattr(report, "caplog", None)),
            "sections": {name: _shorten(content) for name, content in report.sections},
        }
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    yield
    if _ARTIFACT_PATHS is None:
        return
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "other": 0}
    for record in _ARTIFACT_BUFFER:
        outcome = record.get("outcome", "other")
        counts[outcome] = counts.get(outcome, 0) + 1
    _ARTIFACT_PATHS["cases"].write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in _ARTIFACT_BUFFER),
        encoding="utf-8",
    )
    _ARTIFACT_PATHS["summary"].write_text(
        json.dumps(
            {
                "artifact_dir": str(_ARTIFACT_PATHS["cases"].parent),
                "exitstatus": exitstatus,
                "started_at": _STARTED_AT,
                "finished_at": time.time(),
                "counts": counts,
                "total_cases": len(_ARTIFACT_BUFFER),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
