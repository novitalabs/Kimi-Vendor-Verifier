"""Offline K3 service-period metric evaluation.

The module accepts already collected observations. It never sends traffic and
therefore can be used to review production or benchmark evidence safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass(frozen=True)
class TTFTGate:
    max_input_tokens: int
    p50_limit_s: float
    p90_limit_s: float


TTFT_GATES = (
    TTFTGate(128_000, 15, 25),
    TTFTGate(256_000, 20, 30),
    TTFTGate(512_000, 35, 50),
    TTFTGate(1_000_000, 60, 100),
)


def percentile(values: Iterable[float], percentile_rank: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0 < percentile_rank <= 100:
        raise ValueError("percentile rank must be in (0, 100]")
    return ordered[max(0, ceil(len(ordered) * percentile_rank / 100) - 1)]


def ttft_bucket(input_tokens: int) -> TTFTGate:
    for gate in TTFT_GATES:
        if input_tokens < gate.max_input_tokens:
            return gate
    raise ValueError(f"input token count is outside the K3 gates: {input_tokens}")


def evaluate_ttft(observations: Iterable[dict]) -> dict:
    buckets: dict[int, list[float]] = {gate.max_input_tokens: [] for gate in TTFT_GATES}
    for observation in observations:
        gate = ttft_bucket(int(observation["input_tokens"]))
        buckets[gate.max_input_tokens].append(float(observation["ttft_s"]))
    result = {}
    for gate in TTFT_GATES:
        values = buckets[gate.max_input_tokens]
        if not values:
            result[str(gate.max_input_tokens)] = {"status": "blocked", "count": 0}
            continue
        p50 = percentile(values, 50)
        p90 = percentile(values, 90)
        result[str(gate.max_input_tokens)] = {
            "status": "passed" if p50 < gate.p50_limit_s and p90 < gate.p90_limit_s else "failed",
            "count": len(values),
            "p50_s": p50,
            "p90_s": p90,
            "p50_limit_s": gate.p50_limit_s,
            "p90_limit_s": gate.p90_limit_s,
        }
    return result


def evaluate_otps(observations: Iterable[dict], minimum_otps: float = 30.0) -> dict:
    rows = list(observations)
    if not rows:
        return {"status": "blocked", "count": 0}
    bad = sum(float(row["otps"]) <= minimum_otps for row in rows)
    bad_fraction = bad / len(rows)
    return {
        "status": "passed" if bad_fraction <= 0.10 else "failed",
        "count": len(rows),
        "bad_count": bad,
        "bad_fraction": bad_fraction,
        "minimum_otps": minimum_otps,
    }


def classify_rate_limit(*, status: int, tokens_in_current_minute: int, agreed_tpm: int) -> str:
    """Return the contract class for one request at the configured threshold."""
    if tokens_in_current_minute > agreed_tpm:
        return "pass" if status == 429 else "fail"
    return "pass" if 200 <= status < 300 else "fail"
