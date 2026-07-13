"""Preset loader — model presets under presets/*.json.

A preset describes a model + its endpoint configurations across tracks
(official / self). CLI --preset NAME + --track official|self picks the
combination.

See presets/README.md for schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Presets live in <repo_root>/presets/
PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def list_presets() -> list[str]:
    """Return preset names (without .json) sorted."""
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json") if p.is_file())


def load_preset(name: str) -> dict[str, Any]:
    """Load presets/<name>.json. Raise SystemExit with helpful message on failure."""
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        available = list_presets()
        sys.exit(
            f"[kbench] preset {name!r} not found under {PRESETS_DIR}.\n"
            f"available: {', '.join(available) if available else '(none)'}"
        )
    try:
        preset = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"[kbench] preset {path} has invalid JSON: {e}")
    if "tracks" not in preset:
        sys.exit(f"[kbench] preset {path} missing 'tracks' section")
    return preset


def resolve_preset_track(preset: dict[str, Any], track: str,
                        self_url: str = "", self_key: str = "",
                        self_headers: list[str] | None = None,
                        self_model: str = "") -> dict[str, Any]:
    """Resolve a preset+track into a flat run config.

    For track='official', reads all from preset. api_key gets resolved via
    user_config.resolve_api_key (supports $ENV / file path / literal / user
    keys.NAME).

    For track='self', preset only provides 'preserve_thinking' + recommendations;
    the endpoint tuple must come from CLI (--self-url --self-key --self-header
    --self-model).

    Returns a dict with:
        base_url, api_key, model, preserve_thinking, strict_thinking_spec,
        headers (dict)
    """
    from user_config import resolve_api_key

    if track not in preset.get("tracks", {}):
        sys.exit(
            f"[kbench] preset {preset.get('name')!r} has no track {track!r}. "
            f"available tracks: {list(preset['tracks'].keys())}"
        )

    result = {
        "preserve_thinking": preset.get("preserve_thinking", 0),
        # strict_thinking_spec: does the endpoint enforce Kimi K2.7 spec #3
        # (reject `thinking.type=disabled` and `keep=null` with 400)?
        # K2.7-code presets set true; K2.6 / lenient presets default false.
        "strict_thinking_spec": bool(preset.get("strict_thinking_spec", False)),
    }
    tcfg = preset["tracks"][track]

    if track == "official":
        result["base_url"] = tcfg["base_url"]
        result["api_key"] = resolve_api_key(tcfg.get("api_key", ""))
        result["model"] = tcfg["model_id"]
        result["headers"] = dict(tcfg.get("headers", {}))
    elif track == "self":
        # self track needs CLI-provided endpoint tuple
        if not self_url:
            sys.exit(
                "[kbench] --track self requires --self-url (self-deployed vLLM "
                "endpoint, e.g. http://host:port/v1)"
            )
        result["base_url"] = self_url
        result["api_key"] = resolve_api_key(self_key) if self_key else ""
        result["model"] = self_model or ""
        if not result["model"]:
            sys.exit(
                "[kbench] --track self requires --self-model (the model id "
                "your vLLM exposes, e.g. /models/Kimi-K2.6 or kimi-k27-code)"
            )
        result["headers"] = {}
        for h in (self_headers or []):
            if ":" not in h:
                sys.exit(f"--self-header {h!r}: must be KEY:VALUE")
            k, v = h.split(":", 1)
            result["headers"][k.strip()] = v.lstrip()
    else:
        sys.exit(f"[kbench] unknown track {track!r}, expected 'official' or 'self'")

    return result
