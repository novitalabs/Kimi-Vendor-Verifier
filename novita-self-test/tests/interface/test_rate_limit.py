"""Opt-in live check for the K3 above-agreement 429 contract.

This test is intentionally disabled unless the caller supplies an explicit
traffic budget. It is not part of ordinary smoke or unit runs.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tools.metrics import classify_rate_limit


@pytest.mark.integration
def test_tpm_over_agreement_returns_429(pytestconfig: pytest.Config) -> None:
    if os.environ.get("K3_RUN_RATE_LIMIT") != "1":
        pytest.skip("set K3_RUN_RATE_LIMIT=1 with an explicit traffic budget")
    base_url = str(pytestconfig.getoption("base_url")).rstrip("/")
    api_key = str(pytestconfig.getoption("api_key"))
    model = str(pytestconfig.getoption("smoke_model"))
    agreed_tpm = int(os.environ.get("K3_AGREED_TPM", "0"))
    request_tokens = int(os.environ.get("K3_RATE_LIMIT_REQUEST_TOKENS", "0"))
    request_count = int(os.environ.get("K3_RATE_LIMIT_REQUEST_COUNT", "0"))
    if not api_key or not model or "example.invalid" in base_url:
        pytest.skip("set live K3 endpoint credentials")
    if agreed_tpm < 1 or request_tokens < 1 or request_count < 1:
        pytest.fail("K3_AGREED_TPM, K3_RATE_LIMIT_REQUEST_TOKENS, and K3_RATE_LIMIT_REQUEST_COUNT are required")
    if request_tokens * request_count <= agreed_tpm:
        pytest.fail("configured rate-limit burst does not exceed K3_AGREED_TPM")

    prompt = "token " * request_tokens
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"},
        "max_tokens": 1,
    }

    def send() -> int:
        with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60) as client:
            return client.post("/chat/completions", json=payload).status_code

    with ThreadPoolExecutor(max_workers=min(request_count, 32)) as pool:
        statuses = list(pool.map(lambda _: send(), range(request_count)))

    assert any(status == 429 for status in statuses), f"expected at least one 429 above agreed TPM, got {statuses}"
    for status in statuses:
        assert classify_rate_limit(
            status=status,
            tokens_in_current_minute=request_tokens * request_count,
            agreed_tpm=agreed_tpm,
        ) == "pass", f"unexpected status {status} for above-agreement burst"
