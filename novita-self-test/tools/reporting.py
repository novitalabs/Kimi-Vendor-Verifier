"""Machine-readable run metadata and verdict artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET

STATUSES = {"passed", "failed", "blocked", "not_run", "deferred", "not_implemented"}
SENSITIVE_FLAGS = {"--api-key", "--authorization", "--auth-header"}


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
        elif item in SENSITIVE_FLAGS:
            redacted.append(item)
            redact_next = True
        elif any(item.startswith(flag + "=") for flag in SENSITIVE_FLAGS):
            redacted.append(item.split("=", 1)[0] + "=<redacted>")
        else:
            # A URL anywhere on the command line may carry credentials in its query
            # string, which no flag name would reveal. Strip query/fragment from all
            # of them rather than trusting the caller to pass keys via flags only.
            redacted.append(redact_url(item) if "://" in item else item)
    if redact_next:
        redacted.append("<missing>")
    return redacted


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_url(value: str | None) -> str | None:
    if not value:
        return value
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(root: Path) -> dict:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "status_sha256": None, "diff_sha256": None, "untracked_paths": []}
    return {
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_paths": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
    }
def run_metadata(
    root: Path,
    *,
    command: list[str],
    base_url: str | None,
    model: str | None,
    external_assets_manifest: str | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "started_at": utc_now(),
        "python": sys.version.split()[0],
        "repo_commit": git_commit(root),
        **git_state(root),
        "model": model,
        "base_url": redact_url(base_url),
        "base_url_sha256": sha256_text(base_url) if base_url else None,
        "deployment_id": os.environ.get("K3_DEPLOYMENT_ID"),
        "runtime_id": os.environ.get("K3_RUNTIME_ID"),
        "serve_args_file": os.environ.get("K3_SERVE_ARGS_FILE"),
        "source_commit": os.environ.get("K3_SOURCE_COMMIT"),
        "external_assets_manifest": external_assets_manifest or os.environ.get("K3_EXTERNAL_ASSETS"),
        "extra_header_names": sorted(json.loads(os.environ.get("KIMI_EXTRA_HEADERS_JSON", "{}")).keys()),
        "command": redact_command(command),
        "authorization_recorded": False,
    }




def junit_counts(path: Path) -> dict | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root)
    else:
        suites = [root]
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, 0))
    return counts
def phase_result(
    *,
    phase: str,
    status: str,
    command: list[str] | None = None,
    log: str | None = None,
    reason: str | None = None,
    artifacts: Iterable[str] = (),
    returncode: int | None = None,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid phase status: {status}")
    return {
        "phase": phase,
        "status": status,
        "returncode": returncode,
        "command": redact_command(command) if command is not None else None,
        "log": log,
        "reason": reason,
        "artifacts": list(artifacts),
        "started_at": utc_now(),
    }


def overall_status(phases: Iterable[dict], required_gates: Iterable[str] | None = None) -> str:
    required = set(required_gates or [])
    statuses = {
        phase.get("status")
        for phase in phases
        if not required or phase.get("phase") in required
    }
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses or "not_run" in statuses:
        return "incomplete"
    return "passed" if statuses and statuses <= {"passed"} else "incomplete"


def write_summary(report_dir: Path, *, metadata: dict, phases: list[dict], required_gates: list[str]) -> dict:
    present = {str(phase.get("phase")) for phase in phases}
    missing = sorted(set(required_gates) - present)
    status = "failed" if missing else overall_status(phases, required_gates)
    verdict = {
        "schema_version": 1,
        "status": status,
        "completed": status == "passed",
        "required_gates": required_gates,
        "missing_gates": missing,
        "phases": phases,
        "started_at": metadata.get("started_at"),
        "ended_at": utc_now(),
    }
    (report_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# K3 Acceptance Run",
        "",
        f"Status: **{status}**",
        f"Completed: **{verdict['completed']}**",
        "",
        "| Phase | Status | Tests | Return code | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for phase in phases:
        count = phase.get("test_counts", {}).get("tests", "")
        lines.append(f"| `{phase.get('phase')}` | `{phase.get('status')}` | {count} | {phase.get('returncode', '')} | {phase.get('reason') or ''} |")
    if missing:
        lines.extend(["", "Missing required gates:", *[f"- `{item}`" for item in missing]])
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict
