"""Diff two run directories' rewards.tsv side-by-side."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_rewards(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Return {task: (reward, n_errors)} from <run_dir>/rewards.tsv."""
    tsv = run_dir / "rewards.tsv"
    if not tsv.exists():
        # fall back: look for any vendor_smoke_rewards_*.tsv inside
        candidates = list(run_dir.rglob("vendor_smoke_rewards_*.tsv"))
        if candidates:
            tsv = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"no rewards.tsv under {run_dir}")
    out: dict[str, tuple[str, str]] = {}
    for line in tsv.read_text().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        task, reward, n_errors = parts[0], parts[1], parts[2]
        out[task] = (reward, n_errors)
    return out


def compare(run_a: str, run_b: str) -> int:
    a = _load_rewards(Path(run_a))
    b = _load_rewards(Path(run_b))
    tasks = sorted(set(a) | set(b))

    a_name = Path(run_a).name
    b_name = Path(run_b).name

    print(f"| task | {a_name} | {b_name} | delta |")
    print(f"|---|---|---|---|")
    a_mean = 0.0
    b_mean = 0.0
    a_count = 0
    b_count = 0
    for t in tasks:
        ra, ea = a.get(t, ("-", "-"))
        rb, eb = b.get(t, ("-", "-"))
        try:
            ra_f = float(ra)
            a_mean += ra_f
            a_count += 1
        except ValueError:
            ra_f = None
        try:
            rb_f = float(rb)
            b_mean += rb_f
            b_count += 1
        except ValueError:
            rb_f = None
        if ra_f is not None and rb_f is not None:
            d = rb_f - ra_f
            delta = "↑" if d > 0 else ("↓" if d < 0 else "=")
        else:
            delta = "-"
        print(f"| {t} | {ra} | {rb} | {delta} |")
    a_avg = a_mean / a_count if a_count else 0.0
    b_avg = b_mean / b_count if b_count else 0.0
    print(f"| **mean** | **{a_avg:.3f}** ({a_count}) | **{b_avg:.3f}** ({b_count}) | **{(b_avg - a_avg):+.3f}** |")
    return 0


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1], sys.argv[2]))
