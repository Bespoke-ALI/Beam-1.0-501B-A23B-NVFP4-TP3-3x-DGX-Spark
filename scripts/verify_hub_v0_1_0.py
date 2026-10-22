#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_hub_v0_1_0.py - check a downloaded Hugging Face snapshot against the pinned tree listing.

Every file in the listing must exist with the listed size. Large (LFS) files must match their SHA-256;
small files must match their git blob SHA-1. Prints one line per file, then VERIFY_RESULT=PASS or FAIL.

Usage: python3 verify_hub_v0_1_0.py --dir /srv/models/Beam-1.0-501B-A23B-NVFP4 --manifests /srv/models/_manifests
The manifests folder holds REVISION and tree_<revision>.json, written in the download step of the README.
Needs only the Python standard library.
"""
import argparse, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed


def check(dest, e):
    p = os.path.join(dest, e["path"])
    if not os.path.exists(p):
        return e["path"], "MISSING", ""
    if os.path.getsize(p) != e["size"]:
        return e["path"], "SIZE", "%d!=%d" % (os.path.getsize(p), e["size"])
    lfs = (e.get("lfs") or {}).get("oid")
    if lfs:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return e["path"], ("OK" if h.hexdigest() == lfs else "SHA256_MISMATCH"), "sha256"
    h = hashlib.sha1(b"blob %d\0" % e["size"])
    with open(p, "rb") as f:
        h.update(f.read())
    return e["path"], ("OK" if h.hexdigest() == e.get("oid") else "GITSHA1_MISMATCH"), "git-sha1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="downloaded snapshot folder")
    ap.add_argument("--manifests", required=True, help="folder with REVISION and tree_<rev>.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rev = open(os.path.join(a.manifests, "REVISION")).read().strip()
    tree = [e for e in json.load(open(os.path.join(a.manifests, "tree_%s.json" % rev))) if e.get("type") == "file"]
    t0, bad, n = time.time(), [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(check, a.dir, e) for e in tree]):
            path, st, how = fut.result()
            n += 1
            if st != "OK":
                bad.append((path, st, how))
            print("%s %2d/%d %-16s %s %s" % (time.strftime("%H:%M:%S"), n, len(tree), st, how, path), flush=True)
    extra = sorted(set(os.listdir(a.dir)) - {e["path"].split("/")[0] for e in tree} - {".cache"})
    print("VERIFY revision=%s files=%d bad=%d extra_files=%s seconds=%d" % (rev, len(tree), len(bad), extra or "none", time.time() - t0))
    print("VERIFY_RESULT=" + ("PASS" if not bad else "FAIL " + str(bad[:5])))


if __name__ == "__main__":
    main()
