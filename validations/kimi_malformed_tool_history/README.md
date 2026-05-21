# Kimi Malformed Tool History Replay

These sanitized OpenAI-compatible chat completion requests cover malformed
historical `assistant.tool_calls[*].function.arguments` shapes that previously
caused Kimi serving to fail before generation.

The covered bad-history shapes are:

- missing `function.arguments`
- truncated JSON strings such as `{"queries":["..."`
- truncated code payloads such as `{"code":"...`
- truncated URL/data-source payloads
- valid JSON that decodes to a non-object value, such as an array

Run the replay against a Kimi OpenAI-compatible endpoint:

```bash
python validations/run_kimi_malformed_tool_history.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Kimi-K2.6-mix-b300 \
  --output-dir /tmp/kimi-malformed-tool-history-replay
```

For PPIO/OpenAI-compatible routing with provider headers:

```bash
python validations/run_kimi_malformed_tool_history.py \
  --url https://api.ppio.com/openai/v1/chat/completions \
  --api-key "$KIMI_API_KEY" \
  --provider kimi-k26-mix-polaris \
  --model moonshotai/kimi-k2.6-h \
  --output-dir /tmp/kimi-malformed-tool-history-replay
```

Expected result: every case returns HTTP 2xx and no streaming
`data: {"error": ...}` frame. Non-Kimi models are intentionally not covered by
this replay fixture because they should keep stricter OpenAI validation
behavior.
