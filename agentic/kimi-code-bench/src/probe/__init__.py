"""Endpoint capability probe.

Sends a fixed set of curl-equivalent requests to detect:
1. Endpoint reachable + which model ids are exposed
2. Empty tools=[] is accepted (vLLM-int A-line fix)
3. Top-level thinking={type:enabled, keep:all} accepted
4. thinking.type='disabled' rejected (Kimi K2.7 spec #3)
5. thinking.keep=null rejected
6. Interleaved thinking enforcement (raise vs warn)

Returns a ProbeResult the caller can use to pick run mode:
- supports_top_level_thinking: True if case 3 returned 200
- enforces_kimi_validations: True if case 4 returned 400
- interleaved_thinking_strict: True if case 6 returned 400 (raise),
  False if 200 (warn-only or absent)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import socket
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT = 30


@dataclass
class CaseResult:
    name: str
    expected_http: int
    actual_http: int
    body: dict[str, Any] | None
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.actual_http == self.expected_http


@dataclass
class ProbeResult:
    base_url: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    models_listed: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.cases)

    @property
    def supports_top_level_thinking(self) -> bool:
        c = next((x for x in self.cases if x.name == "thinking_enabled_keep_all"), None)
        return c is not None and c.passed

    @property
    def enforces_kimi_validations(self) -> bool:
        c = next((x for x in self.cases if x.name == "thinking_disabled_rejected"), None)
        return c is not None and c.passed and c.actual_http == 400

    @property
    def interleaved_thinking_strict(self) -> bool:
        c = next((x for x in self.cases if x.name == "interleaved_missing_reasoning"), None)
        return c is not None and c.actual_http == 400

    def format(self) -> str:
        lines = [f"=== Probe: {self.base_url} (model={self.model}) ==="]
        if self.models_listed:
            lines.append(f"Models exposed: {', '.join(self.models_listed)}")
        for c in self.cases:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  [{mark}] {c.name}: expected {c.expected_http}, got {c.actual_http} {c.notes}")
        # Only print capability summary when spec cases are present; the
        # extended probe runs different cases and the capability flags would
        # falsely read "False" otherwise.
        spec_names = {"thinking_enabled_keep_all", "thinking_disabled_rejected",
                      "interleaved_missing_reasoning"}
        if any(c.name in spec_names for c in self.cases):
            lines.append("")
            lines.append("Detected capabilities:")
            lines.append(f"  - top-level thinking accepted: {self.supports_top_level_thinking}")
            lines.append(f"  - K2.7 strict validation on:   {self.enforces_kimi_validations}")
            lines.append(f"  - interleaved thinking strict: {self.interleaved_thinking_strict}")
        return "\n".join(lines)


def _post(url: str, body: dict, headers: dict, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, dict | None]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req) as r:
            try:
                return r.status, json.loads(r.read())
            except json.JSONDecodeError:
                return r.status, None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except (json.JSONDecodeError, Exception):
            return e.code, None
    except Exception:
        return 0, None


def _get(url: str, headers: dict, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers=headers)
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req) as r:
            try:
                return r.status, json.loads(r.read())
            except json.JSONDecodeError:
                return r.status, None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except (json.JSONDecodeError, Exception):
            return e.code, None
    except Exception:
        return 0, None


def probe_endpoint(
    base_url: str,
    api_key: str,
    infer_id: str = "",
    model: str = "kimi-k26",
    extra_headers: dict[str, str] | None = None,
    strict_mode: bool = True,
) -> ProbeResult:
    """Probe endpoint capabilities.

    strict_mode:
      True  -> Kimi K2.7-code spec: `thinking.type=disabled` and
               `keep=null` MUST return 400. Endpoints that return 200
               fail the probe.
      False -> Kimi K2.6 / lenient endpoints: 200 on those cases is
               correct behavior. The probe still records the actual
               HTTP code but scores 200 as passing.

    Callers wire this from the preset: K2.7-code presets pass True;
    K2.6 / opensource presets pass False. See run_probe / cmd_probe.
    """
    base_url = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if infer_id:
        headers["X-Infer-ID"] = infer_id
    if extra_headers:
        headers.update(extra_headers)
    chat_url = f"{base_url}/chat/completions"

    result = ProbeResult(base_url=base_url, model=model)

    # 0. /v1/models reachability
    code, body = _get(f"{base_url}/models", headers)
    result.cases.append(CaseResult("models_alive", 200, code, body))
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        result.models_listed = [m.get("id", "?") for m in body["data"]]

    # 1. baseline chat
    code, _ = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }, headers)
    result.cases.append(CaseResult("baseline_chat", 200, code, None))

    # 2. empty tools=[] (A-line fix)
    code, _ = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "max_tokens": 8,
    }, headers)
    result.cases.append(CaseResult("empty_tools_accepted", 200, code, None,
                                   notes="vLLM-int A-line fix; OpenAI default behavior"))

    # 3. top-level thinking enabled+keep=all
    code, _ = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "keep": "all"},
        "max_tokens": 8,
    }, headers)
    result.cases.append(CaseResult("thinking_enabled_keep_all", 200, code, None,
                                   notes="spec #2: top-level thinking field"))

    # 4. thinking disabled (spec #3 expects 400 on K2.7-strict, 200 on lenient)
    code, _ = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
        "max_tokens": 8,
    }, headers)
    expected = 400 if strict_mode else 200
    result.cases.append(CaseResult("thinking_disabled_rejected", expected, code, None,
                                   notes=("spec #3: strict endpoints reject with 400; "
                                          "lenient (K2.6-family) accept with 200")))

    # 5. keep=null
    code, _ = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "keep": None},
        "max_tokens": 8,
    }, headers)
    result.cases.append(CaseResult("keep_null_rejected", expected, code, None,
                                   notes=("spec #3: strict endpoints reject with 400; "
                                          "lenient (K2.6-family) accept with 200")))

    # 6. interleaved thinking with missing reasoning
    # spec #6 in strict mode: 400; in warn mode (v2c): 200 + log
    interleaved_body = {
        "model": model,
        "thinking": {"type": "enabled", "keep": "all"},
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "y"},
        ],
        "max_tokens": 8,
    }
    code, _ = _post(chat_url, interleaved_body, headers)
    # We DON'T mark this as required-passing; record whichever and let the
    # caller decide. expected=200 here just so it shows green when warn-only.
    result.cases.append(CaseResult("interleaved_missing_reasoning", 200, code, None,
                                   notes="spec #6: 400 = strict, 200 = warn-only"))

    # 7. interleaved thinking with reasoning_content (always 200)
    good_body = dict(interleaved_body)
    good_body["messages"] = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "reasoning_content": "thinking about it...",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "y"},
    ]
    code, _ = _post(chat_url, good_body, headers)
    result.cases.append(CaseResult("interleaved_with_reasoning", 200, code, None,
                                   notes="spec #6 happy path"))

    return result


def _post_raw(url: str, body: dict, headers: dict, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str]:
    """Like _post but returns raw response text (for streaming probes)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def probe_extended(
    base_url: str,
    api_key: str,
    infer_id: str = "",
    model: str = "kimi-k26",
    extra_headers: dict[str, str] | None = None,
) -> ProbeResult:
    """Conformance probes beyond the 8 spec cases.

    Adds 6 wire-protocol-style checks that matter for vendor evaluation but are
    not covered by the spec validators alone:
      A. streaming chunks parse cleanly + final finish_reason emitted
      B. tool_call round-trip: tools={one}, tool_choice=auto, expect tool_calls
         + finish_reason="tool_calls" + JSON-parseable arguments
      C. max_tokens=4 truncation: finish_reason="length"
      D. thinking enabled emits non-empty reasoning_content
      E. prompt caching: send the same long-ish prefix twice, expect cache hit
         on the second usage.prompt_tokens_details
      F. long prompt (~16K tokens) doesn't fail
    """
    base_url = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if infer_id:
        headers["X-Infer-ID"] = infer_id
    if extra_headers:
        headers.update(extra_headers)
    chat_url = f"{base_url}/chat/completions"
    result = ProbeResult(base_url=base_url, model=model)

    # A. Streaming
    code, raw = _post_raw(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, headers, timeout=60)
    if code == 200:
        chunks = [ln[6:] for ln in raw.splitlines()
                  if ln.startswith("data: ") and ln != "data: [DONE]"]
        parsed = []
        bad = 0
        for c in chunks:
            try: parsed.append(json.loads(c))
            except: bad += 1
        finish = ""
        usage = None
        for p in parsed:
            for ch in (p.get("choices") or []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if p.get("usage"):
                usage = p["usage"]
        result.cases.append(CaseResult(
            "streaming_chunks", 200, code, None,
            notes=f"chunks={len(parsed)} bad_json={bad} finish={finish} usage={'yes' if usage else 'no'}",
        ))
    else:
        result.cases.append(CaseResult("streaming_chunks", 200, code, None,
                                       notes=f"HTTP {code}: {raw[:120]}"))

    # B. Tool call round-trip
    code, body = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "Call get_weather for Tokyo."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 256,
    }, headers)
    if code == 200 and isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message", {}) or {}
            tcs = msg.get("tool_calls") or []
            finish = choices[0].get("finish_reason", "")
            arg_ok = False
            if tcs:
                try:
                    json.loads(tcs[0].get("function", {}).get("arguments", "{}"))
                    arg_ok = True
                except Exception:
                    pass
            note = f"tool_calls={len(tcs)} finish={finish} args_parse={arg_ok}"
            passed = bool(tcs) and arg_ok and finish in ("tool_calls", "stop")
            result.cases.append(CaseResult(
                "tool_call_roundtrip", 200, code if passed else -1, None, notes=note,
            ))
        else:
            result.cases.append(CaseResult("tool_call_roundtrip", 200, -1, None,
                                           notes="no choices in response"))
    else:
        result.cases.append(CaseResult("tool_call_roundtrip", 200, code, None,
                                       notes=f"body: {str(body)[:120]}"))

    # C. max_tokens=4 truncation
    code, body = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "Count slowly from one to twenty."}],
        "max_tokens": 4,
    }, headers)
    if code == 200 and isinstance(body, dict):
        finish = (body.get("choices") or [{}])[0].get("finish_reason", "")
        passed = finish == "length"
        result.cases.append(CaseResult(
            "max_tokens_truncation", 200, code if passed else -1, None,
            notes=f"finish={finish} (want 'length')",
        ))
    else:
        result.cases.append(CaseResult("max_tokens_truncation", 200, code, None))

    # D. thinking enabled → reasoning_content present
    code, body = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
        "thinking": {"type": "enabled", "keep": "all"},
        "max_tokens": 256,
    }, headers)
    if code == 200 and isinstance(body, dict):
        msg = (body.get("choices") or [{}])[0].get("message", {}) or {}
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        passed = len(reasoning) > 20
        result.cases.append(CaseResult(
            "thinking_emits_reasoning", 200, code if passed else -1, None,
            notes=f"reasoning_len={len(reasoning)} (want >20)",
        ))
    else:
        result.cases.append(CaseResult("thinking_emits_reasoning", 200, code, None))

    # E. prompt caching
    long_prefix = (
        "You are a careful assistant.\n\n"
        + ("Please remember this list: " + ", ".join(f"item-{i}" for i in range(120)) + ".\n") * 3
    )
    body1 = {
        "model": model,
        "messages": [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "How many items did I list? Answer with a single number."},
        ],
        "max_tokens": 8,
    }
    code1, b1 = _post(chat_url, body1, headers)
    code2, b2 = _post(chat_url, body1, headers)
    cache_hit = False
    note_extra = ""
    if code1 == 200 and code2 == 200 and isinstance(b2, dict):
        usage2 = b2.get("usage") or {}
        det = usage2.get("prompt_tokens_details") or {}
        cached = det.get("cached_tokens", 0) or 0
        cache_hit = cached > 0
        note_extra = f"second_cached_tokens={cached}"
    result.cases.append(CaseResult(
        "prompt_caching", 200, 200 if cache_hit else -1, None,
        notes=f"{note_extra} (want >0; -1 if no cache signal)" if note_extra else "no usage info",
    ))

    # F. long prompt smoke
    big = "Repeat this sentence in your mind: " + ("foo bar " * 2000)  # ~16K chars
    code, body = _post(chat_url, {
        "model": model,
        "messages": [{"role": "user", "content": big + "\nNow say 'ok'."}],
        "max_tokens": 8,
    }, headers, timeout=60)
    result.cases.append(CaseResult(
        "long_prompt_smoke", 200, code, None,
        notes=f"~{len(big) // 4} tokens user msg",
    ))

    return result
