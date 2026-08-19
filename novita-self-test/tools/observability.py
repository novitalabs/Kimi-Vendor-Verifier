"""Validate the minimum operational metrics shape promised by K3."""

from __future__ import annotations

REQUIRED_CATEGORIES = {"text_chat", "image_chat", "text_claw", "image_claw"}


def validate_metrics_snapshot(snapshot: dict, *, model_id: str) -> dict:
    categories = snapshot.get("categories")
    if not isinstance(categories, dict):
        return {"status": "failed", "reason": "categories must be an object"}
    missing = sorted(REQUIRED_CATEGORIES - set(categories))
    model = snapshot.get("model_id")
    errors = []
    if model != model_id:
        errors.append(f"model_id mismatch: expected {model_id!r}, got {model!r}")
    for name, value in categories.items():
        if not isinstance(value, dict):
            errors.append(f"category {name!r} must be an object")
            continue
        for field in ("prompt_tokens", "completion_tokens", "requests"):
            if not isinstance(value.get(field), int) or value[field] < 0:
                errors.append(f"category {name!r} field {field!r} must be a non-negative integer")
    if missing:
        errors.append(f"missing categories: {', '.join(missing)}")
    return {"status": "passed" if not errors else "failed", "missing": missing, "errors": errors}
