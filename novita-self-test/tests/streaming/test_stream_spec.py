"""
Verify the vendor's streaming Chat Completion output conforms to the spec
documented in 「流式 Chat Completion 输入和输出接口规格」.

Reads the raw SSE stream via httpx (not the openai SDK) so we can assert on
wire-level details that the SDK normalizes away — `data:` prefix, presence /
absence of optional keys, `delta = {}` end frame, `choices = []` usage frame,
the `reasoning_content=""` thinking boundary frame, etc.

Sections referenced in inline messages refer to the spec doc.

Spec ↔ test mapping (P0.N / P1.M / P2.K numbers from the spec list).
The authoritative machine-readable copy lives in the acceptance console's
`acceptance/rows.yaml` under row I09 (`spec_items`); the table below is a reader's
aid kept in sync with it:

  P0.1  include_usage 汇总帧                    → test_stream_spec_include_usage
  P0.2  usage 字段类型 (reasoning_tokens /
        cached_tokens 明细在 include_usage 汇总
        帧和结束帧均强制)                       → test_stream_spec_include_usage +
                                                    test_stream_spec_basic
  P0.3  工具调用续块仅含 index+arguments        → test_stream_spec_tool_calls
  P0.4  结束帧 choices[0].usage 携带            → test_stream_spec_basic (via helper)
  P0.5  include_internal_content + token_ids    → test_stream_spec_include_internal_content
  P0.6  SSE 帧格式与终止标记                    → test_stream_spec_basic (via helper)
  P0.7  顶层必含字段                            → test_stream_spec_basic (via helper)
  P0.8  choice 必含字段                         → test_stream_spec_basic (via helper)
  P0.9  首帧 delta.role                         → test_stream_spec_basic (via helper)
  P0.10 正文增量 delta.content 类型             → test_stream_spec_basic + test_stream_spec_frame_mutual_exclusion
  P0.11 UTF-8 字节拼接安全                       → test_stream_spec_utf8_safety
  P0.12 reasoning_content 是字符串增量          → test_stream_spec_thinking (P0 floor) + test_stream_spec_frame_mutual_exclusion
  P0.13 工具调用首块结构                         → test_stream_spec_tool_calls
  P0.14 工具调用 index + arguments 拼接         → test_stream_spec_tool_calls
  P0.15 finish_reason 标准值                    → test_stream_spec_basic (via helper)
  P0.16 n>1 多候选                              → test_stream_spec_n_gt_1
  P0.17 进流前错误                              → test_stream_spec_prefill_error
  P0.18 usage 统计响应头                         → test_stream_spec_usage_headers

  P1.1  stop 序列范围                           → test_stream_spec_stop_sequence
  P1.2  [DONE] 收尾                             → test_stream_spec_done_marker
  P1.3  顶层字段跨帧恒等                         → test_stream_spec_top_level_field_constancy
  P1.4  created 类型                            → test_stream_spec_top_level_field_constancy
  P1.5  首帧 delta.content 键必现               → test_stream_spec_first_frame_shape
  P1.6  role 仅首帧出现                         → test_stream_spec_first_frame_shape
  P1.7  thinking 结束边界帧                     → test_stream_spec_thinking (P1 focus)
  P1.8  function.arguments 键始终出现           → test_stream_spec_tool_calls (helper 隐式)
  P1.9  工具调用 index 聚簇                     → test_stream_spec_tool_call_index_clustering
  P1.10 结束帧 delta == {}                      → test_stream_spec_end_frame_shape
  P1.11 每个 index 的 finish_reason 仅一次      → test_stream_spec_end_frame_shape
  P1.12 内容/结束分帧                            → test_stream_spec_end_frame_shape (隐式)
  P1.13 扩展 finish_reason 视为异常             → (未覆盖:难人工触发)
  P1.14 汇总帧字段恒等                           → test_stream_spec_summary_frame_field_constancy
  P1.15 中间帧 usage 为 null                    → test_stream_spec_middle_frame_usage_null
  P1.16 流式中不发只含 error 的 SSE 帧          → (未覆盖:难人工触发)
  P1.17 logprobs 输出结构                        → test_stream_spec_logprobs

  P2.1  id 24-hex ObjectId 格式                 → test_stream_spec_id_format
  P2.2  tool_call.id `<name>_<idx>` 命名        → test_stream_spec_tool_call_id_naming
  P2.3  默认不发汇总帧                           → test_stream_spec_no_summary_by_default

  额外: 帧内 content/reasoning/tool_calls 互斥 → test_stream_spec_frame_mutual_exclusion
"""

from __future__ import annotations

import json
import re
import typing as t

import httpx
import pytest

# ===========================================================================
# Standalone single-file harness — no conftest.py, no external repo.
# Deps: pip install pytest httpx
# (optional: pytest-rerunfailures enables @pytest.mark.flaky reruns)
#
# Config via environment variables:
#   BASE_URL              vendor base url, e.g. https://api.example.com/v1 (required)
#   API_KEY               vendor API key (required)
#   SMODEL                model id under test (required)
#   STREAM_SPEC_PRIORITY  comma-separated priority filter, e.g. P0,P1;
#                         empty = run all; filtered-out tests show as skipped
#
# Run:  BASE_URL=... API_KEY=... SMODEL=... pytest test_stream_spec.py
# ===========================================================================

import os
import warnings

# No conftest.py exists to register the custom marks used below; silence the
# resulting unknown-mark warnings (installing pytest-rerunfailures registers
# `flaky` and enables its reruns).
warnings.filterwarnings("ignore", message="Unknown pytest.mark.*")

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
_SMODEL = os.getenv("SMODEL", "")
_PRIORITY_FILTER = {
    p.strip().upper()
    for p in os.getenv("STREAM_SPEC_PRIORITY", "").split(",")
    if p.strip()
}

THINKING_ONLY_MODELS = {"kimi-k2.7-code", "kimi-k2.7-code-highspeed"}


@pytest.fixture(autouse=True)
def _priority_gate(request):
    if not _PRIORITY_FILTER:
        return
    marker = request.node.get_closest_marker("priority")
    level = str(marker.args[0]).upper() if marker and marker.args else None
    if level is not None and level not in _PRIORITY_FILTER:
        pytest.skip(f"priority {level} filtered out by STREAM_SPEC_PRIORITY")


@pytest.fixture(scope="session")
def model():
    # Keep this fixture side-effect free so priority-filtered tests can skip
    # without requiring live endpoint credentials.
    return _SMODEL


@pytest.fixture(scope="session")
def is_vendor():
    # Standalone mode always targets a vendor base-url.
    return True


@pytest.fixture(scope="session")
def thinking_only_models():
    return set(THINKING_ONLY_MODELS)


@pytest.fixture(scope="session")
def skip_non_thinking_model():
    def _skip(model: str) -> None:
        if model in THINKING_ONLY_MODELS:
            pytest.skip(f"skip non-thinking case for thinking-only model: {model}")
    return _skip


@pytest.fixture(scope="function")
def hclient():
    if not BASE_URL or not API_KEY:
        raise pytest.UsageError("BASE_URL and API_KEY env vars are required")
    client = httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {API_KEY}", **json.loads(os.getenv("KIMI_EXTRA_HEADERS_JSON", "{}"))},
        timeout=60,
    )
    yield client
    client.close()



# §6.1 standard + §6.2 extended finish_reason values.
STANDARD_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}
EXTENDED_FINISH_REASONS = {
    "unexpected_state",
    "malformed_byte_sequence",
    "engine_overloaded",
    "server_interrupted",
    "repeat",
    "unknown",
}
ALL_FINISH_REASONS = STANDARD_FINISH_REASONS | EXTENDED_FINISH_REASONS

# §4.1 id format: chatcmpl-<24 hex>.
ID_PATTERN = re.compile(r"^chatcmpl-[0-9a-f]{24}$")

# §5.4.1 tool_call.id format: <function_name>_<global_index>.
TOOL_CALL_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*_\d+$")


class SSEFrame(t.NamedTuple):
    """One decoded SSE event with both the raw line and parsed JSON payload."""

    raw: str
    payload: t.Optional[dict]
    is_done: bool


def _iter_sse_frames(response: httpx.Response) -> t.Iterator[SSEFrame]:
    """Yield each `data: ...` event in spec order.

    Asserts the §3 SSE framing contract (each line begins with `data: `, JSON
    payload is parseable, `[DONE]` is exactly `[DONE]`).
    """
    for line in response.iter_lines():
        if not line:
            continue
        assert line.startswith("data: "), (
            f"§3 violation: every SSE event must start with 'data: ', got: {line!r}"
        )
        body = line[len("data: ") :]
        if body == "[DONE]":
            yield SSEFrame(raw=line, payload=None, is_done=True)
            return
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            pytest.fail(f"§3 violation: data frame is not valid JSON: {body!r} ({exc})")
        assert isinstance(payload, dict), (
            f"§4 violation: each chunk must be a JSON object, got {type(payload).__name__}"
        )
        yield SSEFrame(raw=line, payload=payload, is_done=False)


def _request_stream(
    hclient: httpx.Client,
    *,
    model: str,
    messages: list,
    thinking_type: str = "disabled",
    tools: t.Optional[list] = None,
    include_usage: bool = False,
    max_tokens: int = 1024,
    extra: t.Optional[dict] = None,
):
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking_type},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)

    # httpx.Client.stream() returns a context manager that streams the body and
    # auto-closes the response on exit; Client.send(req, stream=True) does NOT.
    return hclient.stream("POST", "/chat/completions", json=body)


def _assert_top_level_constants(payload: dict, baseline: t.Optional[dict]) -> dict:
    """Validate §4 top-level fields and confirm they stay constant across frames.

    §4.1 (`id` 24-hex ObjectId format) is P2 and is NOT enforced here — it has
    its own dedicated test so it does not short-circuit P0 / P1 checks for
    vendors whose `id` uses a different shape but is otherwise compliant.
    """

    # §4 — these four are non-optional on every chunk.
    for key in ("id", "object", "created", "model"):
        assert key in payload, f"§4 violation: chunk missing required field {key!r}"

    # id existence and string type — P0. Regex (P2 §4.1) is checked separately.
    assert isinstance(payload["id"], str) and payload["id"], (
        f"§4 violation: id must be a non-empty string, got {payload['id']!r}"
    )
    assert payload["object"] == "chat.completion.chunk", (
        f"§4.2 violation: object must be 'chat.completion.chunk', got {payload['object']!r}"
    )
    assert isinstance(payload["created"], int), (
        f"§4.3 violation: created must be an integer Unix-second, got {type(payload['created']).__name__}"
    )
    assert isinstance(payload["model"], str) and payload["model"], (
        f"§4.4 violation: model must be a non-empty string, got {payload['model']!r}"
    )

    if baseline is None:
        return {k: payload[k] for k in ("id", "object", "created", "model")}

    for k, v in baseline.items():
        assert payload[k] == v, (
            f"§4 violation: top-level field {k!r} must be constant across frames "
            f"(expected {v!r}, got {payload[k]!r})"
        )
    return baseline


def _assert_usage_shape(usage: dict, context: str, *, require_details: bool = False) -> None:
    """Validate usage-object required fields and integer types.

    Rules:
    - `prompt_tokens` / `completion_tokens` / `total_tokens` are always required
      integers (P0.2 / P0.4 — applies wherever `usage` appears).
    - `completion_tokens_details.reasoning_tokens` and
      `prompt_tokens_details.cached_tokens` are required on complete usage
      objects, including the **per-choice end frame** and the
      **`include_usage` summary frame**. Pass `require_details=True` when
      validating those frames.
    """
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert key in usage and type(usage[key]) is int, (
            f"P0.2 violation: {context}.{key} must be a required integer, "
            f"got {usage.get(key)!r}"
        )

    if not require_details:
        return

    completion_details = usage.get("completion_tokens_details")
    assert isinstance(completion_details, dict), (
        f"P0.1/P0.2 violation: {context}.completion_tokens_details must be a required object, "
        f"got {completion_details!r}"
    )
    reasoning_tokens = completion_details.get("reasoning_tokens")
    assert type(reasoning_tokens) is int, (
        f"P0.1/P0.2 violation: {context}.completion_tokens_details.reasoning_tokens "
        f"must be a required integer; return 0 when there are no reasoning tokens, "
        f"got {reasoning_tokens!r}"
    )

    prompt_details = usage.get("prompt_tokens_details")
    assert isinstance(prompt_details, dict), (
        f"P0.1/P0.2 violation: {context}.prompt_tokens_details must be a required object, "
        f"got {prompt_details!r}"
    )
    cached_tokens = prompt_details.get("cached_tokens")
    assert type(cached_tokens) is int, (
        f"P0.1/P0.2 violation: {context}.prompt_tokens_details.cached_tokens "
        f"must be a required integer; return 0 when there are no cached tokens, "
        f"got {cached_tokens!r}"
    )

def _consume_stream(
    response: httpx.Response,
    *,
    expect_thinking: bool,
    expect_tool_calls: bool,
    expect_usage_summary: bool,
) -> dict:
    """Walk the SSE stream and assert **P0-only** conformance.

    P1 / P2 rules (first-frame `content` key, end-frame `delta == {}`,
    `finish_reason` once per idx, `role` only in first frame, `[DONE]` marker,
    reasoning boundary frame, mutual-exclusion of increment kinds, no-backwards
    clustering of tool_call.index (monotonic +1 itself is P0 and IS enforced
    below), tool_call.id naming, no-summary-by-default, middle-frame usage
    null, cross-frame constancy of top-level fields, etc.) are
    intentionally NOT enforced here — they have their own dedicated tests, so
    a P1 or P2 deviation never short-circuits P0 visibility.

    Returns a stats dict for callers, including `raw_frames` for dedicated
    P1 / P2 tests that need to re-inspect the stream.
    """
    assert response.status_code == 200, (
        f"non-200 status before streaming starts: {response.status_code}, body: {response.read()!r}"
    )
    ctype = response.headers.get("content-type", "")
    assert "text/event-stream" in ctype, (
        f"§1 violation: stream response must use content-type text/event-stream, got {ctype!r}"
    )

    baseline: t.Optional[dict] = None
    saw_done = False
    saw_usage_summary = False
    finished_when_summary_arrived: t.Optional[set[int]] = None
    # Per-content-frame collected text (for §5.2 UTF-8 / content checks by callers).
    content_pieces: dict[int, list[str]] = {}
    # Raw payloads for dedicated P1 / P2 tests.
    raw_frames: list[dict] = []

    # Per-candidate (per choices[].index) bookkeeping.
    seen_first_frame: dict[int, bool] = {}
    seen_finish: dict[int, str] = {}
    saw_reasoning_nonnull: dict[int, bool] = {}
    saw_reasoning_boundary: dict[int, bool] = {}
    content_chunks: dict[int, int] = {}

    # Per-(candidate, tool_call.index) bookkeeping.
    tool_call_started: dict[tuple[int, int], dict] = {}
    tool_call_arg_pieces: dict[tuple[int, int], list[str]] = {}
    # Per-candidate next expected tool_call.index (§5.4 P0 — start at 0, +1).
    tool_call_next_index: dict[int, int] = {}

    for frame in _iter_sse_frames(response):
        if frame.is_done:
            saw_done = True
            break

        payload = frame.payload
        raw_frames.append(payload)
        baseline = _assert_top_level_constants(payload, baseline)

        assert "choices" in payload, "§4 violation: choices field must be present"
        choices = payload["choices"]
        assert isinstance(choices, list), "§4 violation: choices must be a list"

        # §7.1 — usage summary frame (P0).
        if not choices:
            assert "usage" in payload and isinstance(payload["usage"], dict), (
                "§7.1 violation: when choices=[] the frame must carry a usage object"
            )
            usage = payload["usage"]
            # Summary frame requires the full detail objects (P0.1 + P0.2).
            _assert_usage_shape(usage, "summary usage", require_details=True)
            saw_usage_summary = True
            # §9 P0 — snapshot which candidates had already finished at the moment
            # the summary frame arrived (callers use this to verify §9 ordering).
            finished_when_summary_arrived = set(seen_finish.keys())
            continue

        # §9 P0 — each frame is single-choice element, even with n>1.
        assert len(choices) == 1, (
            f"§9 violation: every SSE frame must contain exactly one choice element "
            f"(even with n>1, candidates interleave one-element frames), got {len(choices)}"
        )
        choice = choices[0]
        assert isinstance(choice, dict), "§5 violation: choice must be an object"

        # §5 P0 — always-present fields on a choice.
        assert "index" in choice, "§5 violation: choices[].index must always be present (even when 0)"
        idx = choice["index"]
        assert isinstance(idx, int) and idx >= 0, (
            f"§5 violation: choices[].index must be a non-negative int, got {idx!r}"
        )
        assert "delta" in choice, "§5 violation: choices[].delta must always be present"
        delta = choice["delta"]
        assert isinstance(delta, dict), "§5 violation: choices[].delta must be an object"
        assert "finish_reason" in choice, (
            "§5 violation: choices[].finish_reason must always be present (null or string)"
        )
        finish_reason = choice["finish_reason"]

        is_first_frame_for_idx = not seen_first_frame.get(idx, False)

        # §5.1 P0 — first frame MUST carry delta.role='assistant'. (The content-key
        # requirement and role-only-first-frame rules are P1; they live in their
        # own dedicated tests.)
        if is_first_frame_for_idx:
            seen_first_frame[idx] = True
            assert delta.get("role") == "assistant", (
                f"§5.1 violation: first frame for index {idx} must carry delta.role='assistant', "
                f"got delta={delta!r}"
            )
            # §5.1 P0 — the first frame is not an end frame; finish_reason must be null.
            assert finish_reason is None, (
                f"§5.1 violation: first frame for index {idx} must carry finish_reason=null, "
                f"got {finish_reason!r}"
            )
            continue

        # End frame — P0 pieces only (delta=={} and finish_reason-only-once are P1).
        if finish_reason is not None:
            seen_finish[idx] = finish_reason
            # §6 P0 — finish_reason must be in the allowed set.
            assert finish_reason in ALL_FINISH_REASONS, (
                f"§6 violation: unknown finish_reason {finish_reason!r}; expected one of {sorted(ALL_FINISH_REASONS)}"
            )
            # §5.5/§7 P0 — end frame's choices[0].usage must always be present
            # with required integer fields and token-detail fields.
            usage_on_end = choice.get("usage")
            assert isinstance(usage_on_end, dict), (
                f"§5.5/§7 violation: end frame must carry choices[0].usage object, got {usage_on_end!r}"
            )
            _assert_usage_shape(
                usage_on_end,
                "end-frame choices[0].usage",
                require_details=True,
            )
            continue

        # Increment frames — P0 pieces only (mutual-exclusion §5.0 is P1;
        # boundary-frame §5.3 ordering is P1; index-clustering §5.4 is P1;
        # tool_call.id naming §5.4.1 is P2; middle-frame-usage-null §7 is P1).

        # §5.2 P0 — delta.content must be a string when present.
        if "content" in delta:
            assert isinstance(delta["content"], str), (
                f"§5.2 violation: delta.content must be a string increment, got {type(delta['content']).__name__}"
            )
            content_chunks[idx] = content_chunks.get(idx, 0) + 1
            content_pieces.setdefault(idx, []).append(delta["content"])

        # §5.3 P0 — delta.reasoning_content must be a string when present.
        # P0.12 floor: at least one frame with non-null reasoning_content;
        # an empty string is allowed and still counts.
        if "reasoning_content" in delta:
            rc = delta["reasoning_content"]
            assert isinstance(rc, str), (
                f"§5.3 violation: delta.reasoning_content must be a string, got {type(rc).__name__}"
            )
            if rc == "":
                saw_reasoning_boundary[idx] = True
            saw_reasoning_nonnull[idx] = True

        # §5.4 P0 — tool_calls frames: shape checks (non-empty list, tc.index,
        # first-chunk fields, subseq-chunk cleanliness, string arguments).
        if "tool_calls" in delta:
            tcs = delta["tool_calls"]
            assert isinstance(tcs, list) and tcs, (
                f"§5.4 violation: delta.tool_calls must be a non-empty list, got {tcs!r}"
            )
            for tc in tcs:
                assert isinstance(tc, dict), "§5.4 violation: each tool_call must be an object"
                assert "index" in tc and isinstance(tc["index"], int), (
                    f"§5.4 violation: tool_calls[].index must be a non-negative int, got {tc.get('index')!r}"
                )
                tc_idx = tc["index"]
                assert tc_idx >= 0, f"§5.4 violation: tool_calls[].index must be ≥ 0, got {tc_idx}"

                key = (idx, tc_idx)
                fn = tc.get("function")
                assert fn is None or isinstance(fn, dict), (
                    f"§5.4 violation: tool_calls[].function must be an object if present, got {fn!r}"
                )

                if key not in tool_call_started:
                    # §5.4 P0 — tool_call.index must start at 0 and increase by +1
                    # per candidate. (No-backwards clustering is P1 — dedicated test.)
                    expected_tc_idx = tool_call_next_index.get(idx, 0)
                    assert tc_idx == expected_tc_idx, (
                        f"§5.4 violation: tool_calls[].index must start at 0 and increase "
                        f"monotonically by +1 per candidate, expected {expected_tc_idx}, "
                        f"got {tc_idx} (choice {idx})"
                    )
                    tool_call_next_index[idx] = expected_tc_idx + 1
                    # §5.4.1 P0 — first chunk fields (id, type, function.name, function.arguments).
                    assert "id" in tc, (
                        f"§5.4.1 violation: first tool_call chunk for index {tc_idx} must carry id, got {tc!r}"
                    )
                    assert "type" in tc and tc["type"] == "function", (
                        f"§5.4.1 violation: first tool_call chunk must carry type='function', got {tc.get('type')!r}"
                    )
                    assert fn is not None and "name" in fn and fn["name"], (
                        f"§5.4.1 violation: first tool_call chunk must carry function.name, got function={fn!r}"
                    )
                    assert "arguments" in fn, (
                        f"§5.4.1 violation: first tool_call chunk must carry function.arguments key "
                        f"(even when empty), got function={fn!r}"
                    )
                    assert fn["arguments"] == "", (
                        f"§5.4.1 violation: first tool_call chunk function.arguments must be an "
                        f"empty string (arguments stream in subsequent chunks), got {fn['arguments']!r}"
                    )
                    # (§5.4.1 tool_call.id naming `<name>_<idx>` is P2 — see dedicated test.)
                    tool_call_started[key] = {"id": tc.get("id"), "name": fn["name"]}
                    tool_call_arg_pieces[key] = [""]
                else:
                    # §5.4.2 P0 — subsequent chunks: only index + function.arguments.
                    assert set(tc.keys()) <= {"index", "function"}, (
                        f"§5.4.2 violation: subsequent tool_call chunks must only carry index and "
                        f"function keys, got {sorted(tc.keys())} in {tc!r}"
                    )
                    assert "id" not in tc, (
                        f"§5.4.2 violation: subsequent tool_call chunks must not carry id, got {tc!r}"
                    )
                    assert "type" not in tc, (
                        f"§5.4.2 violation: subsequent tool_call chunks must not carry type, got {tc!r}"
                    )
                    assert fn is not None, (
                        f"§5.4.2 violation: subsequent tool_call chunks must carry function.arguments, got {tc!r}"
                    )
                    assert set(fn.keys()) <= {"arguments"}, (
                        f"§5.4.2 violation: subsequent tool_call chunks must only carry function.arguments, "
                        f"got function={fn!r}"
                    )
                    assert "arguments" in fn, (
                        f"§5.4.2 violation: function.arguments key must always be present, got function={fn!r}"
                    )
                    assert isinstance(fn["arguments"], str), (
                        f"§5.4.2 violation: function.arguments must be a string, got {type(fn['arguments']).__name__}"
                    )
                    tool_call_arg_pieces[key].append(fn["arguments"])

    # §5.5 P0 — every candidate must have emitted a finish_reason (once-only
    # constraint is checked in a dedicated P1 test).
    assert seen_first_frame, "stream produced zero frames"
    for idx in seen_first_frame:
        assert idx in seen_finish, (
            f"§5.5 violation: candidate index {idx} never emitted a finish_reason"
        )

    # §5.3 P0 — when thinking is expected, at least one frame must carry a
    # non-null reasoning_content (an empty string is allowed and counts).
    # (Boundary-frame requirement is P1 — see dedicated test.)
    if expect_thinking:
        assert any(saw_reasoning_nonnull.values()), (
            "§5.3 violation: expected reasoning_content frames (non-null; empty "
            "string allowed) but stream had none"
        )

    # §5.4 P0 — when a tool call is expected, at least one must have been emitted
    # and concatenated `arguments` must be valid JSON.
    if expect_tool_calls:
        assert tool_call_started, (
            "§5.4 violation: expected tool_calls but stream emitted none"
        )
        for key, pieces in tool_call_arg_pieces.items():
            full = "".join(pieces)
            try:
                json.loads(full)
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"§5.4 violation: concatenated tool_call arguments for {key} are not valid JSON: "
                    f"{full!r} ({exc})"
                )

    # §7.1 P0 — include_usage=true MUST produce a `choices=[]` summary frame.
    # (§7.2, the reverse — no summary when include_usage=false — is P2, see
    # dedicated test.)
    if expect_usage_summary:
        assert saw_usage_summary, (
            "§7.1 violation: include_usage=true requested but no choices=[] usage summary frame arrived"
        )

    # (§3 [DONE] presence is P1 — see dedicated test.)

    return {
        "candidates": list(seen_first_frame.keys()),
        "finish_reasons": dict(seen_finish),
        "saw_reasoning_nonnull": dict(saw_reasoning_nonnull),
        "saw_reasoning_boundary": dict(saw_reasoning_boundary),
        "content_chunks": dict(content_chunks),
        "content_pieces": {k: list(v) for k, v in content_pieces.items()},
        "full_content": {k: "".join(v) for k, v in content_pieces.items()},
        "tool_calls": {k: v for k, v in tool_call_started.items()},
        "tool_call_arguments": {k: "".join(v) for k, v in tool_call_arg_pieces.items()},
        "saw_usage_summary": saw_usage_summary,
        "finished_when_summary_arrived": finished_when_summary_arrived,
        "saw_done": saw_done,
        "raw_frames": raw_frames,
    }


# ============================================================================
# P0 — Protocol-level acceptance tests
# ============================================================================
# Each test below targets at least one P0 rule. Failures here mean the vendor
# response is unparseable, drops data, or otherwise breaks core client logic.
# Tests are listed in ascending spec-section order.
# ============================================================================


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_basic(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — vanilla streaming, no thinking. Baseline rules.

    **Spec:** P0.6 (SSE 帧格式) · P0.7 (顶层必含字段) · P0.8 (choice 必含字段) ·
    P0.9 (首帧 delta.role) · P0.10 (content 是字符串) · P0.15 (finish_reason 值域) ·
    P0.4 (结束帧 choices[0].usage,要求 prompt/completion/total_tokens 整数及
    reasoning_tokens / cached_tokens 明细) —— 通过 `_consume_stream` helper 综合校验。

    P0 checks: §3 SSE framing / §4 top-level fields / §4.2 object value /
    §5 always-present fields (index/delta/finish_reason) / §5.1 first-frame role /
    §5.2 content is string / §6.1 finish_reason values / §7 usage shape.
    P1 checks: §3 [DONE] / §5.1 content key / §5.5 end-frame delta == {} /
    §5.5 finish_reason once / §5.6 content/finish split.
    P2 checks: §4.1 id format / §7.2 default no summary frame.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; covered by the thinking case")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with the single word: hello."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        max_tokens=64,
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=False,
            expect_tool_calls=False,
            expect_usage_summary=False,
        )

    assert stats["candidates"] == [0], (
        f"expected single candidate index 0, got {stats['candidates']}"
    )
    assert stats["finish_reasons"][0] in STANDARD_FINISH_REASONS, (
        f"§6 violation: expected a standard finish_reason, got {stats['finish_reasons'][0]!r}"
    )
    assert stats["content_chunks"].get(0, 0) > 0, (
        "expected at least one delta.content frame on a normal completion"
    )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_utf8_safety(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — §5.2 — server must intercept and stitch incomplete UTF-8
    byte sequences so every `delta.content` is a complete, valid Unicode string.

    **Spec:** P0.11 (UTF-8 字节拼接安全)

    A naive vendor that splits multibyte chars across frames will either inject
    U+FFFD (replacement character) or break JSON parsing entirely.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; covered indirectly by other thinking cases")

    messages = [
        {"role": "system", "content": "You are a helpful multilingual assistant."},
        {
            "role": "user",
            # Force a mix of multibyte CJK + emojis to maximize chance of mid-token splits.
            "content": "请逐字复述这段文本(不要加任何额外解释):你好,世界 🌍 한국어 日本語 — café résumé 🚀✨",
        },
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        max_tokens=256,
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=False,
            expect_tool_calls=False,
            expect_usage_summary=False,
        )

    pieces = stats["content_pieces"].get(0, [])
    assert pieces, "expected at least one content frame"

    # Each individual content frame must be valid Unicode with no replacement chars.
    for i, piece in enumerate(pieces):
        assert "�" not in piece, (
            f"§5.2 violation: content frame #{i} contains U+FFFD replacement char "
            f"(vendor likely split a UTF-8 byte sequence across frames): {piece!r}"
        )
        # Round-trip: must be encodable as UTF-8 (always true for valid Python str,
        # but this also catches lone surrogates that some servers leak via \uD8XX escapes).
        try:
            piece.encode("utf-8")
        except UnicodeEncodeError as exc:
            pytest.fail(
                f"§5.2 violation: content frame #{i} not encodable as UTF-8 "
                f"(probable lone surrogate from raw token leak): {piece!r} ({exc})"
            )

    full = stats["full_content"].get(0, "")
    assert "�" not in full, (
        f"§5.2 violation: concatenated content contains U+FFFD: {full!r}"
    )
    # Verify at least one multibyte char actually appears, so we know the server processed
    # one — otherwise the test is vacuous (vendor may have refused / responded in pure ASCII).
    assert any(ord(c) > 0x7F for c in full), (
        f"§5.2 vacuous: response had no multibyte characters; can't verify UTF-8 stitching. full={full!r}"
    )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_tool_calls(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** (with P2 noise on `tool_call.id` naming) — §5.4.

    **Spec:** P0.3 (续块仅含 index + arguments) · P0.13 (首块结构) ·
    P0.14 (index 单调递增 + arguments 拼接为合法 JSON)

    P0 checks: §5.4.1 first-chunk shape (index+id+type+function.name+arguments="") /
    §5.4.2 subsequent-chunk shape (only index+arguments) / §5.4.2 arguments is string /
    §5.4 index monotonic from 0 / §5.4 concatenated arguments is valid JSON.
    P1 checks: §5.4.1 function.arguments key always present / §5.4 index clustered.
    P2 checks: §5.4.1 `tool_call.id` formatted as `<func_name>_<idx>` (Moonshot extension).
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    # Use thinking=disabled where supported to keep the stream short; otherwise enable it.
    use_thinking = "enabled" if model in thinking_only_models else "disabled"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a tool-using assistant. Always call the provided function to answer weather questions.",
        },
        {"role": "user", "content": "What's the weather in Beijing right now? Use the tool."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type=use_thinking,
        tools=tools,
        max_tokens=512,
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=(use_thinking == "enabled"),
            expect_tool_calls=True,
            expect_usage_summary=False,
        )

    assert stats["finish_reasons"][0] == "tool_calls", (
        f"expected finish_reason='tool_calls', got {stats['finish_reasons'][0]!r}"
    )
    # All tool calls registered their first chunk → ids should be present.
    for (cand_idx, tc_idx), meta in stats["tool_calls"].items():
        assert meta["name"] == "get_weather", (
            f"unexpected tool name for choice {cand_idx} tool_call {tc_idx}: {meta!r}"
        )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_include_usage(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — §7.1 — `include_usage=true` must add a tail `choices=[]`
    summary frame after the end frame, with full-request token statistics.

    **Spec:** P0.1 (include_usage 汇总帧) · P0.2 (usage 字段类型)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model.find("2.5") != -1:
        pytest.skip(f"{model} later than k2.5 will have such key!!! {model} dose not have such feature.")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; the include_usage rule is shape-only")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say 'ok'."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        include_usage=True,
        max_tokens=32,
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=False,
            expect_tool_calls=False,
            expect_usage_summary=True,
        )

    assert stats["saw_usage_summary"], (
        "§7.1 violation: include_usage=true should produce a `choices=[]` summary frame"
    )
    # §7.1 P0 — the summary frame must arrive AFTER all candidates have finished
    # (n=1 here; the n>1 ordering case is covered in test_stream_spec_n_gt_1).
    assert stats["finished_when_summary_arrived"] == {0}, (
        f"§7.1 violation: usage summary frame must arrive after all candidates finished; "
        f"finished candidates at summary arrival: {stats['finished_when_summary_arrived']}"
    )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_n_gt_1(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — §9 — `n>1` multi-candidate: each `choices[].index` runs an
    independent full sequence (first frame → content → end frame), every frame still
    carries exactly one `choices` entry, and (if `include_usage`) the summary frame
    arrives only AFTER all candidates have finished.

    **Spec:** P0.16 (n>1 多候选)

    Vendors that reject `n>1` (some streaming + tool servers do) are skipped on HTTP
    4xx — the spec lists `n` as optional, not mandatory."""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; n>1 case uses thinking=disabled")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with one short sentence about the sea."},
    ]
    n_requested = 2
    # NOTE: `_request_stream` only builds the (not-yet-entered) context manager —
    # no I/O happens until the `with` block is entered, so the transport-level
    # try/except must wrap the `with`, not the constructor call.
    try:
        with _request_stream(
            hclient,
            model=model,
            messages=messages,
            thinking_type="disabled",
            include_usage=True,
            max_tokens=64,
            extra={"n": n_requested},
        ) as response:
            if response.status_code != 200:
                body = response.read()
                # spec §10.1: pre-stream errors carry {error: {...}}; if the vendor refused n>1,
                # treat that as "n>1 not supported here" rather than a spec failure.
                pytest.skip(f"vendor rejected n={n_requested}: HTTP {response.status_code} body={body!r}")
            stats = _consume_stream(
                response,
                expect_thinking=False,
                expect_tool_calls=False,
                expect_usage_summary=True,
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"vendor rejected n>1 request at transport layer: {exc}")

    candidates = sorted(stats["candidates"])
    assert candidates == list(range(n_requested)), (
        f"§9 violation: expected candidate indices {list(range(n_requested))}, got {candidates}"
    )
    for idx in range(n_requested):
        assert idx in stats["finish_reasons"], (
            f"§9 violation: candidate index {idx} never emitted a finish_reason"
        )
        # §5.5 — each candidate's finish_reason appears exactly once. (_consume_stream already
        # enforces this; here we additionally check the per-candidate end frame surfaced.)
        assert stats["finish_reasons"][idx] in STANDARD_FINISH_REASONS, (
            f"§6 violation: candidate {idx} finish_reason={stats['finish_reasons'][idx]!r} not standard"
        )

    # §9 — the usage summary frame must arrive after ALL candidates have finished.
    finished_when_summary = stats["finished_when_summary_arrived"]
    assert finished_when_summary is not None, (
        "§9 violation: expected an `include_usage` summary frame after all candidates finished"
    )
    assert finished_when_summary == set(range(n_requested)), (
        f"§9 violation: usage summary frame arrived before all candidates finished; "
        f"only {sorted(finished_when_summary)} had finished out of {n_requested} requested"
    )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_prefill_error(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
):
    """**Priority: P0** — §10.1 — request-level errors (before the stream begins)
    MUST return a non-200 HTTP response with `{"error": {message, type, code?, param?}}`
    JSON, NOT an SSE event-stream. Triggered here by sending a clearly invalid `model`.

    **Spec:** P0.17 (进流前错误)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    # An obviously invalid model name should trip the vendor's pre-flight validation.
    bogus_model = "this-model-does-not-exist-on-any-vendor-zzz999"
    body = {
        "model": bogus_model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 8,
    }
    response = hclient.post("/chat/completions", json=body)

    assert response.status_code != 200, (
        f"§10.1 violation: invalid model should fail pre-flight with non-200, got {response.status_code} "
        f"body={response.text!r}"
    )
    ctype = response.headers.get("content-type", "")
    assert "text/event-stream" not in ctype, (
        f"§10.1 violation: pre-stream errors must NOT use event-stream content-type, got {ctype!r}"
    )
    try:
        payload = response.json()
    except ValueError as exc:
        pytest.fail(f"§10.1 violation: error body must be JSON, got {response.text!r} ({exc})")

    assert isinstance(payload, dict) and "error" in payload, (
        f"§10.1 violation: error body must be {{\"error\": {{...}}}}, got {payload!r}"
    )
    err = payload["error"]
    assert isinstance(err, dict), f"§10.1 violation: error must be an object, got {type(err).__name__}"
    assert "message" in err and isinstance(err["message"], str) and err["message"], (
        f"§10.1 violation: error.message must be a non-empty string, got {err.get('message')!r}"
    )
    assert "type" in err and isinstance(err["type"], str) and err["type"], (
        f"§10.1 violation: error.type must be a non-empty string, got {err.get('type')!r}"
    )
    if "code" in err and err["code"] is not None:
        assert isinstance(err["code"], (str, int)), (
            f"§10.1 violation: error.code must be string or number when present, got {type(err['code']).__name__}"
        )
    if "param" in err and err["param"] is not None:
        assert isinstance(err["param"], str), (
            f"§10.1 violation: error.param must be a string when present, got {type(err['param']).__name__}"
        )


# ============================================================================
# P2 — Convention / identifier-format tests
# ============================================================================
# These rules are conventions or recommended identifier formats. Clients can
# treat the field as an opaque value and still work; vendor compliance is
# nice-to-have, not required for functional integration.
# ============================================================================


@pytest.mark.priority("P2")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_id_format(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P2** — §4.1 — `id` should be `chatcmpl-<24 hex ObjectId>`,
    so chunk ids sort lexicographically by creation time.

    **Spec:** P2.1 (id 24-hex ObjectId 格式)

    This test runs an independent walk of the stream (not via `_consume_stream`)
    so a P2 id-format deviation never short-circuits the P0 / P1 checks in the
    other tests. A vendor whose id is opaque-but-stable is functionally fine;
    only the lexicographic-time-sort use case is impaired.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    messages = [
        {"role": "user", "content": "Say 'ok'."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        max_tokens=32,
    ) as response:
        assert response.status_code == 200, (
            f"non-200 before streaming starts: {response.status_code}, body: {response.read()!r}"
        )

        first_id: t.Optional[str] = None
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            chunk_id = payload.get("id")
            assert isinstance(chunk_id, str), (
                f"§4 violation: id must be a string, got {chunk_id!r}"
            )
            if first_id is None:
                first_id = chunk_id
                assert ID_PATTERN.match(chunk_id), (
                    f"§4.1 violation: id must match 'chatcmpl-<24 hex ObjectId>' "
                    f"so ids sort lexicographically by creation time, "
                    f"got {chunk_id!r}"
                )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_thinking(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
):
    """**Priority: P1** (with P0 floor on `reasoning_content` being a string) — §5.3.

    **Spec:** P0.12 (reasoning_content 是字符串增量) + P1.7 (`reasoning_content=""`
    结束边界帧,且边界之后才允许 content)

    P0 checks: `delta.reasoning_content` is a string increment when present
        (Moonshot extension — OpenAI's reasoning models don't expose CoT at all).
    P1 checks: thinking-enabled streams emit a `reasoning_content=""` END-OF-THINKING
        boundary frame (Moonshot extension); `delta.content` only appears AFTER that
        boundary (Moonshot extension).
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    messages = [
        {"role": "system", "content": "You are a careful assistant. Think step by step before answering."},
        {"role": "user", "content": "What is 17 * 23? Briefly."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="enabled",
        max_tokens=512,
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=True,
            expect_tool_calls=False,
            expect_usage_summary=False,
        )

    assert stats["saw_reasoning_nonnull"].get(0), (
        "expected reasoning_content frames (non-null; empty string allowed) "
        "with thinking enabled"
    )
    if model.find("2.5") != -1:
        # k2.5 may not have reasoning boundary
        return
    assert stats["saw_reasoning_boundary"].get(0), (
        "§5.3 violation: reasoning_content='' boundary frame missing"
    )

    # §5.3 P1 — boundary frame POSITION: it must come after the last non-empty
    # reasoning_content and before the first delta.content, and no
    # reasoning_content may appear once the boundary has passed.
    boundary_frame: t.Optional[int] = None
    saw_nonempty_before_boundary = False
    for i, payload in enumerate(stats["raw_frames"]):
        choices = payload.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        if not isinstance(ch, dict):
            continue
        delta = ch.get("delta") or {}
        if "reasoning_content" in delta:
            if boundary_frame is not None:
                pytest.fail(
                    f"§5.3 violation: reasoning_content appeared at frame#{i} AFTER the "
                    f"end-of-thinking boundary frame#{boundary_frame}: {delta!r}"
                )
            if delta["reasoning_content"] == "":
                boundary_frame = i
            else:
                saw_nonempty_before_boundary = True
        if boundary_frame is None and delta.get("content"):
            pytest.fail(
                f"§5.3 violation: delta.content appeared at frame#{i} BEFORE the "
                f"reasoning_content='' boundary frame: {delta!r}"
            )
    assert boundary_frame is not None and saw_nonempty_before_boundary, (
        "§5.3 violation: boundary frame must come after at least one non-empty "
        "reasoning_content frame (a `reasoning_content=''` smuggled into the first "
        "frame does not count)"
    )


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_continuous_reasoning_usage(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
):
    """Continuous usage must expose one monotonic K3 reasoning-token count.

    The count grows on reasoning chunks, freezes across later tool chunks, and
    matches the terminal choice, terminal top-level usage, and summary usage.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Think briefly, then always call the provided weather tool.",
            },
            {
                "role": "user",
                "content": "Use the tool to check the weather in Beijing.",
            },
        ],
        thinking_type="enabled",
        tools=tools,
        include_usage=True,
        max_tokens=512,
        extra={
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
                "include_internal_content": True,
            }
        },
    ) as response:
        assert response.status_code == 200, response.read()
        payloads = [
            frame.payload
            for frame in _iter_sse_frames(response)
            if not frame.is_done and frame.payload is not None
        ]

    data_frames = [payload for payload in payloads if payload.get("choices")]
    summary = next(
        (payload for payload in payloads if payload.get("choices") == []),
        None,
    )
    terminal = next(
        (
            payload
            for payload in data_frames
            if payload["choices"][0].get("finish_reason") is not None
        ),
        None,
    )
    assert terminal is not None, "expected a terminal choice frame"
    assert summary is not None, "expected an include_usage summary frame"

    top_counts: list[int] = []
    for index, payload in enumerate(data_frames):
        usage = payload.get("usage")
        assert isinstance(usage, dict), (
            f"continuous usage frame#{index} must carry top-level usage, got {usage!r}"
        )
        _assert_usage_shape(usage, f"continuous usage frame#{index}", require_details=True)
        top_counts.append(usage["completion_tokens_details"]["reasoning_tokens"])
    assert all(left <= right for left, right in zip(top_counts, top_counts[1:])), (
        f"continuous reasoning_tokens must be monotonic, got {top_counts}"
    )

    reasoning_frames = [
        payload
        for payload in data_frames
        if payload["choices"][0].get("delta", {}).get("reasoning_content")
    ]
    assert reasoning_frames, "expected at least one non-empty reasoning chunk"
    assert all(
        payload["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0
        for payload in reasoning_frames
    ), "non-empty reasoning chunks must report positive cumulative reasoning_tokens"

    terminal_choice_usage = terminal["choices"][0].get("usage")
    terminal_top_usage = terminal.get("usage")
    summary_usage = summary.get("usage")
    for context, usage in (
        ("terminal choice", terminal_choice_usage),
        ("terminal top-level", terminal_top_usage),
        ("summary", summary_usage),
    ):
        assert isinstance(usage, dict), f"{context} usage missing: {usage!r}"
        _assert_usage_shape(usage, f"{context} usage", require_details=True)

    final_reasoning = terminal_choice_usage["completion_tokens_details"]["reasoning_tokens"]
    assert final_reasoning > 0
    assert terminal_top_usage["completion_tokens_details"]["reasoning_tokens"] == final_reasoning
    assert summary_usage["completion_tokens_details"]["reasoning_tokens"] == final_reasoning

    boundary_index = next(
        (
            index
            for index, payload in enumerate(data_frames)
            if payload["choices"][0].get("delta", {}).get("reasoning_content") == ""
        ),
        None,
    )
    assert boundary_index is not None, "expected reasoning_content='' boundary"
    tool_frames = [
        payload
        for payload in data_frames[boundary_index + 1 :]
        if payload["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert tool_frames, "expected tool-call chunks after the reasoning boundary"
    assert all(
        payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == final_reasoning
        for payload in tool_frames
    ), "tool-call chunks must preserve the final cumulative reasoning_tokens"


# ----------------------------------------------------------------------------
# P1 — dedicated tests carved out of `_consume_stream` so a P1 deviation
# never short-circuits P0 visibility. Each test performs its own request +
# stream walk (via _iter_sse_frames) and enforces one focused rule group.
# ----------------------------------------------------------------------------


def _walk_basic_stream(
    hclient: httpx.Client, model: str, *, thinking_type: str = "disabled",
) -> list[dict]:
    """Send a minimal request and return the list of raw SSE payload dicts
    (excluding the [DONE] marker). Used by the small dedicated P1 / P2 tests
    that only need to inspect frames, not enforce the full P0 contract."""
    with _request_stream(
        hclient,
        model=model,
        messages=[
            {"role": "user", "content": "Reply with the single word: ok."},
        ],
        thinking_type=thinking_type,
        max_tokens=64,
    ) as response:
        assert response.status_code == 200, (
            f"non-200 status: {response.status_code}, body: {response.read()!r}"
        )
        frames: list[dict] = []
        for f in _iter_sse_frames(response):
            if f.is_done:
                return frames
            assert f.payload is not None
            frames.append(f.payload)
        return frames


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_top_level_field_constancy(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §4 — `id` / `object` / `created` / `model` values must
    remain identical across every frame of the same completion. Also §4.3:
    `created` must be a **non-boolean integer**.

    **Spec:** P1.3 (顶层字段跨帧恒等) + P1.4 (`created` 类型整数)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    frames = _walk_basic_stream(hclient, model)
    assert frames, "expected at least one SSE frame"
    first = frames[0]
    for i, f in enumerate(frames):
        for key in ("id", "object", "created", "model"):
            assert f.get(key) == first.get(key), (
                f"§4 violation: {key!r} not constant across frames "
                f"(frame#0={first.get(key)!r} vs frame#{i}={f.get(key)!r})"
            )
        # §4.3
        c = f.get("created")
        assert type(c) is int, (
            f"§4.3 violation: created must be an integer Unix second (not bool / float / str), "
            f"got {type(c).__name__} at frame#{i}: {c!r}"
        )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_done_marker(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §3 — normal-completed stream must end with
    `data: [DONE]\\n\\n`. Fatal errors may omit it; a clean run should not.

    **Spec:** P1.2 (`[DONE]` 收尾)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    with _request_stream(
        hclient,
        model=model,
        messages=[{"role": "user", "content": "Say ok."}],
        thinking_type="disabled",
        max_tokens=32,
    ) as response:
        assert response.status_code == 200
        saw_done = False
        for f in _iter_sse_frames(response):
            if f.is_done:
                saw_done = True
                break
    assert saw_done, (
        "§3 violation: normally-finished stream ended without 'data: [DONE]'"
    )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_stop_sequence(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — a `stop` sequence must truncate the `delta.content`
    stream: generation ends with `finish_reason="stop"`, the stop sequence
    itself must not appear in the concatenated output, and content after the
    stop sequence must be cut.

    **Spec:** P1.1 (stop 序列范围)

    Run with thinking disabled so the assertion targets `delta.content`
    truncation directly (the spec's "stop does not affect reasoning_content /
    tool_calls" clause is not exercised here — those scenarios are hard to
    trigger deterministically)."""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; stop-sequence case uses thinking=disabled")

    stop_word = "STOPWORD"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Output exactly the following text and nothing else: hello {stop_word} world",
        },
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        max_tokens=128,
        extra={"stop": [stop_word]},
    ) as response:
        stats = _consume_stream(
            response,
            expect_thinking=False,
            expect_tool_calls=False,
            expect_usage_summary=False,
        )

    full = stats["full_content"].get(0, "")
    assert stats["finish_reasons"].get(0) == "stop", (
        f"P1.1 violation: expected finish_reason='stop' after hitting the stop sequence, "
        f"got {stats['finish_reasons'].get(0)!r} (content={full!r})"
    )
    assert stop_word not in full, (
        f"P1.1 violation: stop sequence {stop_word!r} must not appear in the output, got {full!r}"
    )
    assert "world" not in full, (
        f"P1.1 violation: content after the stop sequence must be truncated, got {full!r}"
    )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_first_frame_shape(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §5.1 — the first frame per candidate must include the
    `delta.content` key (value may be `""` or `null`) so clients can always
    read `chunk.delta.content` unconditionally. `role` must appear **only** in
    that first frame; subsequent frames must not re-emit it.

    **Spec:** P1.5 (首帧 `delta.content` 键必现) + P1.6 (`role` 仅首帧出现)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    frames = _walk_basic_stream(hclient, model)
    first_frame_seen: dict[int, bool] = {}
    for i, f in enumerate(frames):
        choices = f.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        if not isinstance(ch, dict):
            continue
        idx = ch.get("index")
        if idx is None:
            continue
        delta = ch.get("delta") or {}
        if not first_frame_seen.get(idx, False):
            first_frame_seen[idx] = True
            assert "content" in delta, (
                f"§5.1 violation: first frame for index {idx} must include delta.content key "
                f"(empty allowed), got delta={delta!r} at frame#{i}"
            )
            assert delta["content"] in ("", None), (
                f"§5.1 violation: first-frame delta.content must be '' or null, "
                f"got {delta['content']!r} at frame#{i}"
            )
        else:
            assert "role" not in delta, (
                f"§5.1 violation: delta.role must only appear in the first frame, "
                f"got delta={delta!r} at frame#{i} (index {idx})"
            )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_end_frame_shape(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §5.5 / §5.6 — the end frame (frame with non-null
    `finish_reason`) must have `delta == {}` (empty object, not `{"content": null}`
    or similar), and each candidate's `finish_reason` must appear **exactly once**.

    **Spec:** P1.10 (结束帧 `delta == {}`) + P1.11 (每个 index 的 `finish_reason`
    仅一次) + P1.12 (内容/结束分帧 — 隐式,end-frame delta 为空自然保证不与
    非空 finish_reason 同帧)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    frames = _walk_basic_stream(hclient, model)
    seen_finish: dict[int, str] = {}
    for i, f in enumerate(frames):
        choices = f.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        if not isinstance(ch, dict):
            continue
        fr = ch.get("finish_reason")
        if fr is None:
            continue
        idx = ch.get("index")
        # §5.5 — only one finish_reason per idx.
        assert idx not in seen_finish, (
            f"§5.5 violation: finish_reason for index {idx} appeared twice "
            f"(first={seen_finish[idx]!r}, then={fr!r} at frame#{i})"
        )
        seen_finish[idx] = fr
        # §5.5/§5.6 — end frame delta must be empty {}.
        delta = ch.get("delta")
        assert delta == {}, (
            f"§5.5/§5.6 violation: end frame delta must be empty {{}} "
            f"(no {{'content': null}} or similar), got {delta!r} at frame#{i}"
        )
    assert seen_finish, "expected at least one end frame with non-null finish_reason"


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_frame_mutual_exclusion(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §5.0 — a single non-end frame should carry only ONE of
    `content` / `reasoning_content` / `tool_calls`, not a mix. Tested with
    thinking enabled so both content and reasoning_content have opportunity.

    **Spec:** P0.10 / P0.12 单增量帧约束(§5.0 单帧只一种增量)。
    也覆盖 P1.8(`function.arguments` 键始终出现)相关的续块清洁度隐式检查。"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    with _request_stream(
        hclient,
        model=model,
        messages=[
            {"role": "system", "content": "Think step by step."},
            {"role": "user", "content": "What is 3 + 4? Briefly."},
        ],
        thinking_type="enabled",
        max_tokens=256,
    ) as response:
        assert response.status_code == 200
        for i, frame in enumerate(_iter_sse_frames(response)):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            choices = payload.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason") is not None:
                continue  # end frame — mutual exclusion doesn't apply
            delta = ch.get("delta") or {}
            present = [k for k in ("content", "reasoning_content", "tool_calls") if k in delta]
            assert len(present) <= 1, (
                f"§5.0 violation: a single non-end frame should carry only one of "
                f"content / reasoning_content / tool_calls, got {present} in delta={delta!r} "
                f"at frame#{i}"
            )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_middle_frame_usage_null(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §7 — non-end / non-summary frames must NOT carry usage
    (top-level or per-choice). Vendors that stamp `usage: {prompt_tokens: 0, ...}`
    on every frame can trick clients into double-counting cost.

    **Spec:** P1.15 (中间帧 `usage` 为 `null`)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    frames = _walk_basic_stream(hclient, model)
    for i, f in enumerate(frames):
        choices = f.get("choices") or []
        if not choices:
            continue  # summary frame — usage IS allowed here
        ch = choices[0]
        if not isinstance(ch, dict):
            continue
        if ch.get("finish_reason") is not None:
            continue  # end frame — usage IS allowed here
        # Middle frame — usage must be null / missing.
        assert ch.get("usage") in (None, {}), (
            f"§7 violation: middle frame#{i} carries per-choice usage: {ch.get('usage')!r}"
        )
        assert f.get("usage") is None, (
            f"§7 violation: middle frame#{i} carries top-level usage: {f.get('usage')!r}"
        )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_tool_call_index_clustering(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §5.4 — within one candidate, chunks for the same
    `tool_calls[].index` must be **clustered**: once a different index starts,
    the previous one must not receive more chunks. And indices increase
    monotonically from 0 by +1.

    **Spec:** P1.9 (工具调用 index 聚簇)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    use_thinking = "enabled" if model in thinking_only_models else "disabled"
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        },
    }]
    with _request_stream(
        hclient,
        model=model,
        messages=[
            {"role": "system", "content": "Always call the tool for weather."},
            {"role": "user", "content": "Weather in Beijing? Use the tool."},
        ],
        thinking_type=use_thinking,
        tools=tools,
        max_tokens=256,
    ) as response:
        if response.status_code != 200:
            pytest.skip(f"tool_call scenario returned {response.status_code}")
        seen_indices: list[int] = []
        active: t.Optional[int] = None
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            choices = payload.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                tc_idx = tc.get("index")
                if not isinstance(tc_idx, int):
                    continue
                if tc_idx not in seen_indices:
                    if seen_indices:
                        assert tc_idx == seen_indices[-1] + 1, (
                            f"§5.4 violation: tool_calls[].index must be monotonic +1, "
                            f"saw {seen_indices} then {tc_idx}"
                        )
                    else:
                        assert tc_idx == 0, (
                            f"§5.4 violation: tool_calls[].index must start at 0, got {tc_idx}"
                        )
                    seen_indices.append(tc_idx)
                # Clustering: once we've moved to a new active index, the previous
                # must not receive more chunks.
                if active is not None and tc_idx != active:
                    assert tc_idx > active, (
                        f"§5.4 violation: tool_call.index went backwards "
                        f"(active {active}, then {tc_idx}) — indices must be clustered"
                    )
                active = tc_idx


# ----------------------------------------------------------------------------
# P2 — dedicated tests for identifier-format / convention rules that a
# functional client can ignore. Kept isolated so P2 deviations never leak
# into P0 / P1 test failures.
# ----------------------------------------------------------------------------


@pytest.mark.priority("P2")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_tool_call_id_naming(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P2** — §5.4.1 — `tool_call.id` should be formatted as
    `<function_name>_<global_index>` (e.g. `get_weather_0`), NOT the raw
    tokenizer form `functions.<name>:<idx>`. Clients that treat the id as
    opaque still work; the naming matters for id-round-trip use cases where
    the id is fed back as `tool_call_id` in history.

    **Spec:** P2.2 (`tool_call.id` 命名 `<func_name>_<idx>`)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")

    use_thinking = "enabled" if model in thinking_only_models else "disabled"
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        },
    }]
    with _request_stream(
        hclient,
        model=model,
        messages=[
            {"role": "system", "content": "Always call the tool for weather."},
            {"role": "user", "content": "Weather in Beijing? Use the tool."},
        ],
        thinking_type=use_thinking,
        tools=tools,
        max_tokens=256,
    ) as response:
        if response.status_code != 200:
            pytest.skip(f"tool_call scenario returned {response.status_code}")
        first_tc_id: t.Optional[str] = None
        fn_name: t.Optional[str] = None
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}) if isinstance(choices[0], dict) else {}
            for tc in delta.get("tool_calls") or []:
                if "id" in tc and first_tc_id is None:
                    first_tc_id = tc["id"]
                    fn = tc.get("function") or {}
                    fn_name = fn.get("name")
            if first_tc_id is not None:
                break
    assert isinstance(first_tc_id, str), (
        f"expected first tool_call chunk to carry an id string, got {first_tc_id!r}"
    )
    assert TOOL_CALL_ID_PATTERN.match(first_tc_id), (
        f"§5.4.1 violation: tool_call.id should match '<function_name>_<global_index>' "
        f"(e.g. 'get_weather_0'), got {first_tc_id!r} "
        f"(vendor is likely leaking the raw tokenizer form 'functions.<name>:<idx>')"
    )
    if fn_name:
        assert first_tc_id.startswith(fn_name + "_"), (
            f"§5.4.1 violation: tool_call.id must start with function name prefix "
            f"(name={fn_name!r}, id={first_tc_id!r})"
        )


@pytest.mark.priority("P2")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_no_summary_by_default(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P2** — §7.2 — when the request does NOT set
    `stream_options.include_usage=true`, the vendor must not spontaneously
    emit a `choices=[]` usage summary frame. Auto-emitting it can trip
    clients that use `choices=[]` as an end-of-stream marker.

    **Spec:** P2.3 (默认不发汇总帧)"""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    with _request_stream(
        hclient,
        model=model,
        messages=[{"role": "user", "content": "Say ok."}],
        thinking_type="disabled",
        max_tokens=32,
        # NOTE: intentionally NOT passing stream_options.include_usage.
    ) as response:
        assert response.status_code == 200
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            choices = payload.get("choices")
            assert choices != [], (
                f"§7.2 violation: request did not set include_usage=true but vendor "
                f"emitted a `choices=[]` summary frame: {payload!r}"
            )

    # Explicit include_usage=false must behave the same as omitting stream_options.
    with _request_stream(
        hclient,
        model=model,
        messages=[{"role": "user", "content": "Say ok."}],
        thinking_type="disabled",
        max_tokens=32,
        extra={"stream_options": {"include_usage": False}},
    ) as response:
        assert response.status_code == 200
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            assert payload.get("choices") != [], (
                f"§7.2 violation: request set include_usage=false but vendor "
                f"emitted a `choices=[]` summary frame: {payload!r}"
            )


# ----------------------------------------------------------------------------
# Extra dedicated P1 test that requires an `include_usage=true` stream (kept
# separate from the P2 no-summary-by-default test to avoid double-purposing).
# ----------------------------------------------------------------------------


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_summary_frame_field_constancy(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §7.1 tail — the `include_usage` summary frame
    (`choices=[]` + full-request `usage`) must carry the SAME
    `id` / `object` / `created` / `model` values as every other frame in the
    stream.

    **Spec:** P1.14 (汇总帧 `id` / `object` / `created` / `model` 与流内其他帧
    一致)

    A vendor that stamps the summary frame with a fresh id or different `created`
    would break clients that group all frames of one completion by top-level id.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled")

    with _request_stream(
        hclient,
        model=model,
        messages=[{"role": "user", "content": "Say ok."}],
        thinking_type="disabled",
        include_usage=True,
        max_tokens=32,
    ) as response:
        assert response.status_code == 200
        baseline: t.Optional[dict] = None
        summary_frame: t.Optional[dict] = None
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            if baseline is None:
                # Establish baseline on the very first data frame.
                baseline = {k: payload.get(k) for k in ("id", "object", "created", "model")}
            if payload.get("choices") == []:
                summary_frame = payload
                break

    assert baseline is not None, "expected at least one non-summary frame first"
    assert summary_frame is not None, (
        "§7.1 violation: include_usage=true was requested but no `choices=[]` summary "
        "frame arrived — P1.14 cannot be evaluated"
    )
    for key in ("id", "object", "created", "model"):
        assert summary_frame.get(key) == baseline.get(key), (
            f"P1.14 violation: summary frame's {key!r} differs from the rest of the stream "
            f"(baseline={baseline.get(key)!r}, summary={summary_frame.get(key)!r})"
        )


@pytest.mark.priority("P1")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_logprobs(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P1** — §8 — when `logprobs=true`, frames carrying real
    content (and the end frame) carry `choices[0].logprobs` with the
    OpenAI-standard shape; the role-only first frame must NOT carry logprobs;
    each entry's `top_logprobs` length must follow the request's `top_logprobs`.

    **Spec:** P1.17 (logprobs 输出结构)

    Vendors must support logprobs: rejecting the option with a request-level
    error is a failure, not a skip; accepting it with HTTP 200 but never
    emitting logprobs is also a failure — that silently drops a requested field.
    The reverse (no `logprobs` in the request → no logprobs on the wire) is
    checked with a plain follow-up request.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; logprobs case uses thinking=disabled")

    top_n = 2
    # NOTE: `_request_stream` only builds the (not-yet-entered) context manager —
    # no I/O happens until the `with` block is entered, so the transport-level
    # try/except must wrap the `with`, not the constructor call.
    logprob_frames = 0
    try:
        with _request_stream(
            hclient,
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: hello."}],
            thinking_type="disabled",
            max_tokens=32,
            extra={"logprobs": True, "top_logprobs": top_n},
        ) as response:
            if response.status_code != 200:
                body = response.read()
                pytest.fail(
                    f"P1.17 violation: vendor must support logprobs, got "
                    f"HTTP {response.status_code} body={body!r}"
                )

            first_frame_seen = False
            for frame_i, frame in enumerate(_iter_sse_frames(response)):
                if frame.is_done:
                    break
                payload = frame.payload
                assert payload is not None
                choices = payload.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if not isinstance(ch, dict):
                    continue
                if not first_frame_seen:
                    first_frame_seen = True
                    # §8 — the role-only first frame must NOT carry logprobs.
                    assert ch.get("logprobs") in (None, {}), (
                        f"§8 violation: role-only first frame must not carry logprobs, "
                        f"got {ch.get('logprobs')!r}"
                    )
                    continue
                lp = ch.get("logprobs")
                if lp in (None, {}):
                    continue
                logprob_frames += 1
                # §8 — shape: {"content": [{"token", "logprob", "bytes", "top_logprobs"}, ...]}
                assert isinstance(lp, dict), (
                    f"§8 violation: choices[].logprobs must be an object, "
                    f"got {type(lp).__name__} at frame#{frame_i}"
                )
                content = lp.get("content")
                assert isinstance(content, list) and content, (
                    f"§8 violation: logprobs.content must be a non-empty list, "
                    f"got {content!r} at frame#{frame_i}"
                )
                for j, entry in enumerate(content):
                    where = f"logprobs.content[{j}] at frame#{frame_i}"
                    assert isinstance(entry, dict), (
                        f"§8 violation: {where} must be an object, got {entry!r}"
                    )
                    token = entry.get("token")
                    assert isinstance(token, str), (
                        f"§8 violation: {where}.token must be a string, got {token!r}"
                    )
                    logprob = entry.get("logprob")
                    assert logprob is None or isinstance(logprob, (int, float)), (
                        f"§8 violation: {where}.logprob must be a number or null, got {logprob!r}"
                    )
                    # §8 / OpenAI schema: the bytes key is REQUIRED; its value
                    # is a list of ints or null (e.g. special tokens).
                    assert "bytes" in entry, (
                        f"§8 violation: {where} must carry a bytes key (list of ints or "
                        f"null), keys={list(entry.keys())}"
                    )
                    bts = entry["bytes"]
                    if bts is not None:
                        assert isinstance(bts, list) and all(type(b) is int for b in bts), (
                            f"§8 violation: {where}.bytes must be a list of ints, got {bts!r}"
                        )
                        assert bytes(bts) == token.encode("utf-8"), (
                            f"§8 violation: {where}.bytes must be the UTF-8 encoding of the token "
                            f"(token={token!r}, bytes={bts!r})"
                        )
                    top = entry.get("top_logprobs")
                    assert isinstance(top, list), (
                        f"§8 violation: {where}.top_logprobs must be a list when the request "
                        f"sets top_logprobs, got {top!r}"
                    )
                    assert len(top) == top_n, (
                        f"§8 violation: {where}.top_logprobs length must follow the request "
                        f"(top_logprobs={top_n}), got {len(top)}"
                    )
                    for k, cand in enumerate(top):
                        assert isinstance(cand, dict) and isinstance(cand.get("token"), str), (
                            f"§8 violation: {where}.top_logprobs[{k}] must carry a token string, "
                            f"got {cand!r}"
                        )
                        cand_lp = cand.get("logprob")
                        assert cand_lp is None or isinstance(cand_lp, (int, float)), (
                            f"§8 violation: {where}.top_logprobs[{k}].logprob must be a number "
                            f"or null, got {cand_lp!r}"
                        )
                        # §8 / OpenAI schema: TopLogprob also requires the bytes key
                        # (list of ints or null).
                        assert "bytes" in cand, (
                            f"§8 violation: {where}.top_logprobs[{k}] must carry a bytes key "
                            f"(list of ints or null), got {cand!r}"
                        )
                        cb = cand["bytes"]
                        if cb is not None:
                            assert isinstance(cb, list) and all(type(b) is int for b in cb), (
                                f"§8 violation: {where}.top_logprobs[{k}].bytes must be a list "
                                f"of ints, got {cb!r}"
                            )
    except httpx.HTTPError as exc:
        pytest.skip(f"vendor rejected logprobs request at transport layer: {exc}")

    assert logprob_frames > 0, (
        "§8 violation: logprobs=true was accepted (HTTP 200) but no frame carried "
        "choices[0].logprobs — vendor must either reject the option or emit logprobs"
    )

    # §8 reverse — without logprobs=true, no frame may carry logprobs.
    frames = _walk_basic_stream(hclient, model)
    for i, f in enumerate(frames):
        for ch in f.get("choices") or []:
            if isinstance(ch, dict):
                assert ch.get("logprobs") in (None, {}), (
                    f"§8 violation: request did not set logprobs=true but frame#{i} "
                    f"carries logprobs: {ch.get('logprobs')!r}"
                )


# ----------------------------------------------------------------------------
# P0.18 — usage 统计响应头
# ----------------------------------------------------------------------------


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_usage_headers(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — P0.18 — every response must carry the
    `X-Msh-Usage-Prompt-Tokens` and `X-Msh-Usage-Cached-Tokens` headers so the
    client can account prompt / cached tokens even when the final usage never
    arrives (e.g. the connection drops mid-stream). When the final usage IS
    received, the headers must equal `usage.prompt_tokens` and
    `usage.prompt_tokens_details.cached_tokens` respectively.
    """
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; header case uses thinking=disabled")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply in one short English sentence about the sea."},
    ]
    with _request_stream(
        hclient,
        model=model,
        messages=messages,
        thinking_type="disabled",
        max_tokens=64,
        include_usage=True,
    ) as response:
        assert response.status_code == 200, (
            f"unexpected non-200: {response.status_code} body={response.read()!r}"
        )
        headers = response.headers
        usage = None
        for frame in _iter_sse_frames(response):
            if frame.is_done:
                break
            payload = frame.payload
            assert payload is not None
            if payload.get("usage"):
                usage = payload["usage"]
            for choice in payload.get("choices") or []:
                if isinstance(choice, dict) and choice.get("usage"):
                    usage = choice["usage"]

    prompt_header = headers.get("x-msh-usage-prompt-tokens")
    cached_header = headers.get("x-msh-usage-cached-tokens")
    assert prompt_header is not None and cached_header is not None, (
        "P0.18 violation: response must carry X-Msh-Usage-Prompt-Tokens and "
        f"X-Msh-Usage-Cached-Tokens headers; got prompt={prompt_header!r} "
        f"cached={cached_header!r}"
    )
    for name, value in (("X-Msh-Usage-Prompt-Tokens", prompt_header),
                        ("X-Msh-Usage-Cached-Tokens", cached_header)):
        assert value.isdigit(), (
            f"P0.18 violation: {name} must be a non-negative integer string, got {value!r}"
        )

    assert usage is not None, "expected a final usage frame (include_usage=true)"
    expected_prompt = usage["prompt_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    expected_cached = details.get("cached_tokens", 0)
    assert int(prompt_header) == expected_prompt, (
        f"P0.18 violation: X-Msh-Usage-Prompt-Tokens={prompt_header} != "
        f"usage.prompt_tokens={expected_prompt}"
    )
    assert int(cached_header) == expected_cached, (
        f"P0.18 violation: X-Msh-Usage-Cached-Tokens={cached_header} != "
        f"usage.prompt_tokens_details.cached_tokens={expected_cached}"
    )




@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_usage_headers_non_stream(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """P0.18 also applies to non-stream responses, not only SSE responses."""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; header case uses thinking=disabled")

    response = hclient.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one short sentence about the sea."}],
            "thinking": {"type": "disabled"},
            "max_tokens": 64,
        },
    )
    assert response.status_code == 200, response.text[:1000]
    payload = response.json()
    usage = payload.get("usage")
    assert isinstance(usage, dict), f"P0.18 requires non-stream usage, got {payload!r}"
    _assert_usage_shape(usage, "non-stream usage", require_details=True)

    expected = {
        "x-msh-usage-prompt-tokens": usage["prompt_tokens"],
        "x-msh-usage-cached-tokens": usage["prompt_tokens_details"]["cached_tokens"],
    }
    for name, value in expected.items():
        actual = response.headers.get(name)
        assert actual is not None and actual.isdigit(), (
            f"P0.18 violation: non-stream response must carry {name} as a non-negative integer, "
            f"got {actual!r}"
        )
        assert int(actual) == value, (
            f"P0.18 violation: non-stream {name}={actual} != final usage value {value}"
        )


# ============================================================================
# Moonshot extensions (spec P0.5)
# ============================================================================
# These tests target Moonshot extension contract layers. They walk the raw
# SSE stream directly (not via _consume_stream) so unrelated spec violations
# don't mask extension failures.
# ============================================================================


@pytest.mark.priority("P0")
@pytest.mark.flaky(reruns=2, reruns_delay=2)
def test_stream_spec_include_internal_content(
    hclient: httpx.Client,
    model: str,
    is_vendor: bool,
    skip_non_thinking_model,
    thinking_only_models: set,
):
    """**Priority: P0** — Moonshot extension: when
    `stream_options.include_internal_content=true`, increment frames must
    additionally carry `delta.internal_content.token_ids` — a non-empty list of
    non-negative integers corresponding to the tokens that produced the
    simultaneous `content` / `reasoning_content` / `tool_calls` increment.

    **Spec:** P0.5 (`stream_options.include_internal_content` 扩展)

    Vendors must implement this extension: rejecting the option with a
    request-level error is a failure, not a skip."""
    if not is_vendor:
        pytest.skip("only run against vendor base-url")
    if model in thinking_only_models:
        pytest.skip(f"{model} does not accept thinking=disabled; covered indirectly elsewhere")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply in one short English sentence about the sea."},
    ]
    # NOTE: `_request_stream` only builds the (not-yet-entered) context manager —
    # no I/O happens until the `with` block is entered, so the transport-level
    # try/except must wrap the `with`, not the constructor call.
    try:
        with _request_stream(
            hclient,
            model=model,
            messages=messages,
            thinking_type="disabled",
            max_tokens=64,
            extra={"stream_options": {"include_internal_content": True}},
        ) as response:
            if response.status_code != 200:
                body = response.read()
                pytest.fail(
                    f"P0.5 violation: vendor must support "
                    f"stream_options.include_internal_content, got "
                    f"HTTP {response.status_code} body={body!r}"
                )

            # Walk the SSE stream directly (not via _consume_stream) so unrelated spec
            # violations elsewhere — e.g. §4.1 id format, §5.1 first-frame content key —
            # don't mask whether the internal_content extension was actually honored.
            frames: list[dict] = []
            # Frames carrying a real increment but NO internal_content (violations).
            missing_ic: list[tuple[int, list[str]]] = []
            for frame_i, frame in enumerate(_iter_sse_frames(response)):
                if frame.is_done:
                    break
                payload = frame.payload
                assert payload is not None
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                # A frame "carries an increment" when it holds a non-empty content /
                # reasoning_content string or a tool_calls list.
                increment_keys = [k for k in ("content", "reasoning_content", "tool_calls") if delta.get(k)]
                if increment_keys and "internal_content" not in delta:
                    missing_ic.append((frame_i, increment_keys))
                if "internal_content" in delta:
                    frames.append({
                        "candidate_index": choice.get("index"),
                        "internal_content": delta["internal_content"],
                        "co_keys": [k for k in ("content", "reasoning_content", "tool_calls") if k in delta],
                        "finish_reason": choice.get("finish_reason"),
                    })
    except httpx.HTTPError as exc:
        pytest.skip(f"vendor rejected include_internal_content request at transport layer: {exc}")

    assert frames, (
        "extension violation: stream_options.include_internal_content=true was accepted "
        "(HTTP 200) but no frame carried delta.internal_content — vendor must "
        "emit token_ids on increment frames"
    )

    # At least one content increment frame must carry internal_content alongside it.
    content_co_frames = [f for f in frames if "content" in f["co_keys"]]
    assert content_co_frames, (
        "extension violation: expected delta.internal_content to appear on at least one "
        "frame that also carries delta.content (co-occurrence with the increment it describes); "
        f"observed co-keys per frame: {[f['co_keys'] for f in frames]}"
    )

    # EVERY increment frame must carry internal_content — not just some of them.
    assert not missing_ic, (
        "extension violation: every frame carrying a content / reasoning_content / "
        "tool_calls increment must also carry delta.internal_content; "
        f"missing on frames (frame_index, increment_keys): {missing_ic}"
    )

    # Shape check on every collected internal_content payload.
    for i, f in enumerate(frames):
        ic = f["internal_content"]
        assert isinstance(ic, dict), (
            f"extension violation: delta.internal_content must be an object, "
            f"got {type(ic).__name__} at frame #{i}: {ic!r}"
        )
        assert "token_ids" in ic, (
            f"extension violation: delta.internal_content must contain a token_ids key "
            f"at frame #{i}: keys={list(ic.keys())}"
        )
        token_ids = ic["token_ids"]
        # token_ids is a list of non-negative integers (the tokens that produced
        # this increment; usually 1 element per chunk but vendor may batch).
        assert isinstance(token_ids, list), (
            f"extension violation: delta.internal_content.token_ids must be a list, "
            f"got {type(token_ids).__name__} at frame #{i}: {token_ids!r}"
        )
        # When the frame carries a content/reasoning/tool_calls increment, token_ids
        # must be non-empty.
        if f["co_keys"]:
            assert token_ids, (
                f"extension violation: token_ids must be a non-empty list when accompanying "
                f"an increment (co_keys={f['co_keys']}), got [] at frame #{i}"
            )
        for j, tid in enumerate(token_ids):
            assert isinstance(tid, int) and not isinstance(tid, bool), (
                f"extension violation: token_ids[{j}] must be an integer, "
                f"got {type(tid).__name__} at frame #{i}: {tid!r}"
            )
            assert tid >= 0, (
                f"extension violation: token_ids[{j}] must be ≥ 0, got {tid} at frame #{i}"
            )
