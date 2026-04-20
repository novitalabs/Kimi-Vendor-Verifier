"""
Verify Kimi-K2.5 interleaved thinking validation before running benchmarks.

When the server is launched with a reasoning parser (interleaved thinking),
requests where the last message is role=tool must have reasoning_content in
the preceding assistant message. Otherwise the server should reject the
request with 400 Bad Request.
"""

import argparse
import os
import sys

import httpx
from openai import BadRequestError, OpenAI


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "unit"],
            },
        },
    }
]

TOOL_CALLS = [
    {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        },
    }
]


def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=60.0),
    )


def build_messages(include_reasoning: bool) -> list[dict]:
    assistant_message: dict = {
        "role": "assistant",
        "tool_calls": TOOL_CALLS,
    }
    if include_reasoning:
        assistant_message["reasoning_content"] = (
            "The user wants weather in Fahrenheit for SF. "
            "I should call the get_weather tool."
        )
    return [
        {
            "role": "user",
            "content": "What's the weather in Fahrenheit like in San Francisco?",
        },
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "The current temperature in San Francisco, CA is 72°F.",
        },
    ]


def make_request(
    client: OpenAI,
    model: str,
    include_reasoning: bool,
) -> tuple[bool, str]:
    """Send second-turn tool-result request. Returns (success, message)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(include_reasoning),
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=128,
        )
        content = (
            response.choices[0].message.content[:50]
            if response.choices and response.choices[0].message.content
            else "no content"
        )
        return True, f"OK: {content}"
    except BadRequestError as e:
        return False, f"Rejected(400): {e.message}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


def test_accepted_with_reasoning(client: OpenAI, model: str) -> tuple[bool, str]:
    """Request WITH reasoning_content should succeed (200 OK)."""
    success, msg = make_request(client, model, include_reasoning=True)
    if success:
        return True, "✓ PASS: request with reasoning_content accepted"
    return False, f"❌ FAIL: request with reasoning_content rejected: {msg}"


def test_rejected_without_reasoning(client: OpenAI, model: str) -> tuple[bool, str]:
    """Request WITHOUT reasoning_content should be rejected (400)."""
    success, msg = make_request(client, model, include_reasoning=False)
    if success:
        return (
            False,
            "❌ FAIL: request without reasoning_content should be rejected but succeeded",
        )
    if "interleaved thinking" not in msg.lower():
        return (
            False,
            f"❌ FAIL: expected error about interleaved thinking, got: {msg}",
        )
    return True, "✓ PASS: request without reasoning_content correctly rejected"


def run_verification(
    base_url: str,
    api_key: str,
    model: str,
    test_reject: bool = True,
    test_accept: bool = True,
) -> bool:
    """Run full verification. Returns True if all tests pass."""
    client = get_client(base_url, api_key)

    print(f"\n{'='*60}")
    print("Interleaved thinking validation")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}\n")

    all_passed = True
    results = []

    if test_accept:
        print("[1] Test request WITH reasoning_content (should be accepted)...")
        passed, msg = test_accepted_with_reasoning(client, model)
        results.append(passed)
        if not passed:
            all_passed = False
        print(f"    {msg}")

    if test_reject:
        print("\n[2] Test request WITHOUT reasoning_content (should be rejected)...")
        passed, msg = test_rejected_without_reasoning(client, model)
        results.append(passed)
        if not passed:
            all_passed = False
        print(f"    {msg}")

    print(f"\n{'='*60}")
    passed_count = sum(results)
    status = "✓ ALL PASSED" if all_passed else "❌ SOME FAILED"
    print(f"Result: {status} ({passed_count}/{len(results)})")
    print(f"{'='*60}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify Kimi-K2.5 interleaved thinking validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_interleaved_thinking.py --model kimi-k2.5
  python verify_interleaved_thinking.py --model kimi-k2.5 --base-url http://localhost:8000/v1
  python verify_interleaved_thinking.py --model kimi-k2.5 --only-reject
""",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        help="API base URL (default: $KIMI_BASE_URL or https://api.moonshot.cn/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KIMI_API_KEY"),
        help="API key (default: $KIMI_API_KEY)",
    )
    parser.add_argument(
        "--only-reject",
        action="store_true",
        help="Only test request without reasoning_content is rejected",
    )
    parser.add_argument(
        "--only-accept",
        action="store_true",
        help="Only test request with reasoning_content is accepted",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set KIMI_API_KEY env var or use --api-key")
        sys.exit(1)

    test_reject = not args.only_accept
    test_accept = not args.only_reject

    all_passed = run_verification(
        args.base_url,
        args.api_key,
        args.model,
        test_reject=test_reject,
        test_accept=test_accept,
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
