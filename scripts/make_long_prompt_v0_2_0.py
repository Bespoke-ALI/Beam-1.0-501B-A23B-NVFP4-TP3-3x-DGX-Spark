#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_long_prompt_v0_2_0.py - write a long synthetic service manual (roughly 40-50K tokens; the smoke test prints the exact count) with one planted fact, as a chat request.

Expected answer from the model: TANGERINE-7X4Q. Deterministic (fixed seed). Standard library only.
Usage: python3 make_long_prompt_v0_2_0.py [OUT.json]   (default /tmp/beam_long.json), then
       curl -s 127.0.0.1:8600/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/beam_long.json
"""
import json, random, sys

random.seed(3)
CLAUSES = [
    "Section %d. Before servicing the compressor, isolate the unit from its electrical supply, lock out the main switch, "
    "and release all stored air pressure through the drain valve until the tank gauge reads zero.",
    "Section %d. Inspect the intake filter every 500 operating hours and replace it when the pressure drop across the "
    "filter exceeds the limit shown on the service label, or at least once every twelve months of operation.",
    "Section %d. Check the oil level only when the unit has been stopped for at least ten minutes, with the compressor "
    "standing level, and top up with the grade of oil listed in the specification table if the level is low.",
    "Section %d. Verify that the safety relief valve opens at its rated pressure by operating the test lever with the "
    "tank pressurised, and replace the valve if it does not discharge freely or fails to reseat after the test.",
]
secs = [random.choice(CLAUSES) % i for i in range(1, 900)]
secs.insert(540, "Section 540A. After replacing the discharge pressure sensor, the technician must enter calibration code "
                 "TANGERINE-7X4Q on the controller; no other code is accepted.")
doc = "\n\n".join(secs)
body = {"model": "beam", "max_tokens": 600, "temperature": 0,
        "messages": [{"role": "user", "content": "Read the following service manual carefully.\n\n" + doc +
                      "\n\nQuestion: What calibration code must be entered after replacing the discharge pressure sensor? "
                      "Answer with the code only."}]}
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/beam_long.json"
json.dump(body, open(out, "w"))
print("wrote %s, %d characters" % (out, len(body["messages"][0]["content"])))
