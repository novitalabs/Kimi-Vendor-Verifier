from __future__ import annotations

import json
import os

import httpx
import pytest

from tools.observability import validate_metrics_snapshot


pytestmark = pytest.mark.integration


def test_provider_deploy_id_and_trace_correlation(pytestconfig: pytest.Config) -> None:
    base_url = str(pytestconfig.getoption("base_url")).rstrip("/")
    api_key = str(pytestconfig.getoption("api_key"))
    model = str(pytestconfig.getoption("smoke_model"))
    if not api_key or not model or "example.invalid" in base_url:
        pytest.skip("set live K3 endpoint credentials")
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    headers = {"Authorization": f"Bearer {api_key}", "traceparent": traceparent}
    extra = os.environ.get("KIMI_EXTRA_HEADERS_JSON", "")
    if extra:
        headers.update({str(key): str(value) for key, value in json.loads(extra).items()})
    with httpx.Client(base_url=base_url, headers=headers, timeout=180) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "thinking": {"type": "enabled", "effort": "low"},
                "max_tokens": 16,
            },
        )
    assert response.status_code == 200, response.text[:1000]
    deploy_header = os.environ.get("KIMI_DEPLOY_ID_HEADER", "X-Msh-ProviderDeployId")
    assert response.headers.get(deploy_header), f"missing {deploy_header} response header"
    correlated = {name.lower() for name in response.headers}
    assert {"traceparent", "x-request-id", "x-trace-id"} & correlated, response.headers


def test_metrics_snapshot_contract_from_file(pytestconfig: pytest.Config) -> None:
    path = os.environ.get("KIMI_METRICS_SNAPSHOT")
    if not path:
        pytest.skip("set KIMI_METRICS_SNAPSHOT to a captured token metrics JSON")
    snapshot = json.loads(open(path, encoding="utf-8").read())
    model = str(pytestconfig.getoption("smoke_model"))
    result = validate_metrics_snapshot(snapshot, model_id=model)
    assert result["status"] == "passed", result
