import json
from pathlib import Path


def _records(name: str) -> list[dict]:
    path = Path(__file__).parents[2] / "fixtures" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_official_k3_fixture_counts_and_finish_reasons() -> None:
    image = _records("vendor-img-testcases-inhouse-3.jsonl")
    openclaw = _records("9.openclaw_cases12.jsonl")
    assert len(image) == 25
    assert len(openclaw) == 12
    assert {row["expected_finish_reason"] for row in image} == {"stop", "tool_calls"}
    assert all(row["request"]["tools"] for row in openclaw)


def test_official_image_fixture_covers_schema_and_message_roles() -> None:
    schemas = set()
    roles = set()
    max_images = 0
    for row in _records("vendor-img-testcases-inhouse-3.jsonl"):
        request = row["transformed_request"]
        for message in request["messages"]:
            roles.add(message["role"])
            content = message.get("content")
            if not isinstance(content, list):
                continue
            count = 0
            for part in content:
                if part.get("type") != "image_url":
                    continue
                count += 1
                value = part.get("image_url")
                schemas.add("string" if isinstance(value, str) else "object")
            max_images = max(max_images, count)
    assert schemas == {"string", "object"}
    assert {"user", "tool"}.issubset(roles)
    assert max_images > 1
