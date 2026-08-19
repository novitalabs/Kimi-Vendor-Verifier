import json
from pathlib import Path

from tools.replay_fixture import load_records, parse_body, request_for, safe_response_headers


def test_replay_response_headers_are_allowlisted() -> None:
    assert safe_response_headers(
        {"content-type": "application/json", "set-cookie": "secret", "x-request-id": "req"}
    ) == {"content-type": "application/json", "x-request-id": "req"}


def test_fixture_loader_preserves_official_case_counts() -> None:
    root = Path(__file__).parents[2] / "fixtures"
    image = load_records("image", root / "vendor-img-testcases-inhouse-3.jsonl")
    openclaw = load_records("openclaw", root / "9.openclaw_cases12.jsonl")
    assert len(image) == 25
    assert len(openclaw) == 12
    assert request_for("image", image[0], "kimi-k3")["model"] == "kimi-k3"
    assert request_for("openclaw", openclaw[0], "kimi-k3")["model"] == "kimi-k3"


def test_replay_parser_handles_json_and_sse() -> None:
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    parsed, observation = parse_body(json.dumps(payload).encode())
    assert parsed == payload
    assert observation["stream"] is False

    stream = b"\n".join(
        [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{\\\"x\\\":1}"}}]},"finish_reason":null}]}',
            b"data: [DONE]",
        ]
    )
    parsed, observation = parse_body(stream)
    assert parsed is not None
    assert observation["stream"] is True
    assert observation["done"] is True
    assert observation["tool_calls"][0]["name"] == "f"
