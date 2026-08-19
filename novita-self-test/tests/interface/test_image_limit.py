from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest


@pytest.mark.integration
def test_accepts_1000_image_parts(pytestconfig: pytest.Config) -> None:
    if os.environ.get("K3_RUN_IMAGE_LIMIT") != "1":
        pytest.skip("set K3_RUN_IMAGE_LIMIT=1 to send the official 1000-image probe")
    base_url = str(pytestconfig.getoption("base_url")).rstrip("/")
    api_key = str(pytestconfig.getoption("api_key"))
    model = str(pytestconfig.getoption("smoke_model"))
    if not api_key or not model or "example.invalid" in base_url:
        pytest.skip("set live K3 endpoint credentials")
    fixture_path = Path(__file__).parents[2] / "fixtures" / "vendor-img-testcases-inhouse-3.jsonl"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8").splitlines()[0])
    content = fixture["transformed_request"]["messages"][-1]["content"]
    image_part = next(part for part in content if part.get("type") == "image_url")
    parts = [image_part] * 1000 + [{"type": "text", "text": "Count the images and reply with the number."}]
    payload = {
        "model": model,
        "messages":[{"role": "user", "content": parts}],
        "thinking": {"type": "enabled", "effort": "low"},
        "max_tokens": 32,
    }
    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=900) as client:
        response = client.post("/chat/completions", json=payload)
    assert response.status_code == 200, response.text[:1000]
