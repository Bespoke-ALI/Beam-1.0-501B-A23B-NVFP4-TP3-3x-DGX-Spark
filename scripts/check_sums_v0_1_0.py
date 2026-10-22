#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_sums_v0_1_0.py - cross-check the padded checkpoint's SHA256SUMS against PADDING_MANIFEST.json.

The padder records the SHA-256 of every shard it writes in PADDING_MANIFEST.json. SHA256SUMS is computed
afterwards from the files on disk. Agreement shows nothing changed a shard between writing and hashing.
Prints SUMS_VS_MANIFEST=OK or FAIL.   Usage: python3 check_sums_v0_1_0.py /srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
"""
import json, os, sys


def main():
    d = sys.argv[1]
    sums = {}
    for line in open(os.path.join(d, "SHA256SUMS")):
        h, name = line.split(None, 1)
        sums[name.strip().lstrip("*").removeprefix("./")] = h
    man = json.load(open(os.path.join(d, "PADDING_MANIFEST.json")))["shards"]
    bad = [f for f, v in man.items() if sums.get(f) != v["sha256"]]
    print("SHA256SUMS lines=%d manifest shards=%d mismatched=%d" % (len(sums), len(man), len(bad)))
    for f in bad[:10]:
        print("  MISMATCH", f, sums.get(f), man[f]["sha256"])
    print("SUMS_VS_MANIFEST=" + ("OK" if not bad and len(man) > 0 else "FAIL"))
    sys.exit(0 if (not bad and len(man) > 0) else 1)


if __name__ == "__main__":
    main()
