#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# smoke_test_v0_1_1.sh - four checks against the API on node0. Run on node0 after /health returns 200.
# Beam reasons before it answers, and reasoning counts against max_tokens, so budgets here are generous.
set -u
API="${API:-http://127.0.0.1:8600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="${TMPDIR:-/tmp}"
echo "1. health (want 200): $(curl -s -o /dev/null -w '%{http_code}' "$API/health")"

echo "2. tool call (want finish_reason=tool_calls and one get_weather call with city Paris):"
curl -s "$API/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"beam","max_tokens":400,"temperature":0,
 "tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}],
 "messages":[{"role":"user","content":"What is the weather in Paris right now? Use celsius."}]}' \
 | python3 -c "import json,sys; r=json.load(sys.stdin)['choices'][0]; print('  finish_reason=%s calls=%s' % (r['finish_reason'], [(c['function']['name'], c['function']['arguments']) for c in (r['message'].get('tool_calls') or [])]))"

echo "3. arithmetic canary (want 391):"
curl -s "$API/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"beam","max_tokens":800,"temperature":0,
 "messages":[{"role":"user","content":"What is 17 multiplied by 23? Reply with the number only."}]}' \
 | python3 -c "import json,sys; r=json.load(sys.stdin); print('  content=%r tokens=%s' % ((r['choices'][0]['message'].get('content') or '').strip(), r['usage']['completion_tokens']))"

echo "4. long-context needle (want TANGERINE-7X4Q; takes about 30 s):"
python3 "$HERE/make_long_prompt_v0_2_0.py" "$TMP/beam_long.json" >/dev/null
curl -s "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$TMP/beam_long.json" \
 | python3 -c "import json,sys; r=json.load(sys.stdin); print('  content=%r prompt_tokens=%s' % ((r['choices'][0]['message'].get('content') or '').strip(), r['usage']['prompt_tokens']))"
