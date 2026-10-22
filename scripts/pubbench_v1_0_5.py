#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""pubbench_v1_0_5.py - four public benchmarks against an OpenAI-compatible endpoint serving Beam-1.0.

Benchmarks and what each is compared with:
  gpqa      GPQA Diamond, 198 questions, OpenAI simple-evals prompt and answer pattern -> Reflection NVFP4 89.3%
  ifbench   IFBench test set, official AllenAI scorer (prompt-level loose and strict)  -> Reflection NVFP4 79.2%
  gsm8k     GSM8K test, first 200 questions                                             -> diffbot EXL3 97.5%
  humaneval HumanEval, 164 problems, pass@1, code run in a network-less container       -> diffbot EXL3 97.6%
Sampling for all: temperature 0.7, top_p 0.99 (Reflection's stated evaluation settings). Scores use the final answer only
(Beam's reasoning comes back separately and is never scored). 4 requests at once by default (--workers).
Answers are streamed, so a proxy with an idle timeout in front of the server does not cut off long answers.

Settings come from environment variables (defaults in brackets):
  BEAM_URL           chat completions URL   [http://127.0.0.1:8600/v1/chat/completions]
  BEAM_MODEL         served model name      [beam]
  BEAM_API_KEY_FILE  file holding a bearer key; unset = no Authorization header
  PUBBENCH_DIR       working folder         [~/pubbench]

Token caps: each benchmark has a primary cap (gpqa 32768, others 16384). Answers that hit it are scored as wrong.
To measure how much the cap matters, re-ask only those questions with a higher cap; the primary result is untouched.
Caps can be raised in rounds: each round re-asks the questions still cut off after every lower-cap round.
  python3 pubbench_v1_0_5.py run ifbench --cap 32768 --only-truncated
  python3 pubbench_v1_0_5.py score ifbench --cap 32768   (primary answers, re-asked ones substituted)
  python3 pubbench_v1_0_5.py run ifbench --cap 65536 --only-truncated   (only those still cut off at 32768)
  python3 pubbench_v1_0_5.py score ifbench --cap 65536   (primary, then the 32768 round, then the 65536 round)
A re-asked question is a fresh attempt, not a continuation of the cut-off answer.
Every score is also written to results/<bench>_score[_cap<N>].json.
GPQA's authors ask that its questions not be published online: share GPQA scores, never its prompts or answers.

Steps (each prints a PASS/FAIL gate line):
  python3 pubbench_v1_0_5.py fetch                     download the four datasets into $PUBBENCH_DIR/data, with hashes
  python3 pubbench_v1_0_5.py check                     confirm the endpoint answers (and the key works, if set)
  python3 pubbench_v1_0_5.py run gsm8k [--limit N]     generate (resumable: re-running skips finished items, retries errors)
  python3 pubbench_v1_0_5.py score gsm8k               score, with a 95% interval and the reference figure
A timestamped progress line with an ETA prints every 60 s while a run is going.
"""
import argparse, csv, gzip, hashlib, json, math, os, random, re, shutil, subprocess, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VERSION = "1.0.5"
ROOT = Path(os.environ.get("PUBBENCH_DIR", str(Path.home() / "pubbench")))
DATA, RES = ROOT / "data", ROOT / "results"
URL = os.environ.get("BEAM_URL", "http://127.0.0.1:8600/v1/chat/completions")
MODEL = os.environ.get("BEAM_MODEL", "beam")
KEYFILE = os.environ.get("BEAM_API_KEY_FILE")
SAMPLING = {"temperature": 0.7, "top_p": 0.99}
SOURCES = {
    "gsm8k": ("https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl", "gsm8k_test.jsonl"),
    "humaneval": ("https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz"),
    "gpqa": ("https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv", "gpqa_diamond.csv"),
    "ifbench": ("https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl", "IFBench_test.jsonl"),
}
EXPECT = {"gsm8k": 1319, "humaneval": 164, "gpqa": 198}            # ifbench: count printed, not gated
# First 16 hex characters of each dataset file's SHA-256, as used for the published results (Oct 20, 2026).
# fetch fails if a source has changed since, so later runs score exactly the same questions.
EXPECT_SHA16 = {"gsm8k": "3730d312f6e34405", "humaneval": "b796127e635a67f9", "gpqa": "41d1213cd7a49986",
                "ifbench": "d2ada7da94a38cfe"}
IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"     # scorer commit used for the published results
REFERENCE = {"gpqa": ("Reflection, NVFP4 model card", 89.3), "ifbench": ("Reflection, NVFP4 model card", 79.2),
             "gsm8k": ("diffbot EXL3 3bpw card, first 200", 97.5), "humaneval": ("diffbot EXL3 3bpw card", 97.6)}
MAX_TOKENS = {"gpqa": 32768, "ifbench": 16384, "gsm8k": 16384, "humaneval": 16384}
GPQA_TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
                 "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
                 "\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")
GPQA_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
# Override with exact digests to reproduce the published scores (the digests used are recorded in each score summary).
EXEC_IMAGE = os.environ.get("PUBBENCH_EXEC_IMAGE", "python:3.12-slim")
IFB_IMAGE = os.environ.get("PUBBENCH_IFBENCH_IMAGE", "python:3.12")


def now():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------- data
def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (url, fn) in SOURCES.items():
        p = DATA / fn
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "pubbench/" + VERSION})
            with urllib.request.urlopen(req, timeout=120) as r, open(str(p) + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(str(p) + ".part", p)
        n = len(load_raw(name))
        want = EXPECT.get(name)
        h16 = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        good = (want is None or n == want) and h16 == EXPECT_SHA16[name]
        ok &= good
        print("%-9s %-22s %9d bytes  sha256 %s (want %s)  items %d%s  %s" % (name, fn, p.stat().st_size, h16,
              EXPECT_SHA16[name], n, "" if want is None else " (want %d)" % want, "OK" if good else "BAD"))
    print("FETCH_" + ("PASS" if ok else "FAIL"))


def load_raw(name):
    p = DATA / SOURCES[name][1]
    if name == "gpqa":
        return list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def items(name):
    raw = load_raw(name)
    out = []
    if name == "gsm8k":
        for i, r in enumerate(raw[:200]):
            q = r["question"] + "\n\nSolve the problem step by step. End with a final line of the form 'Answer: <number>'."
            out.append({"id": "gsm8k-%03d" % i, "messages": [{"role": "user", "content": q}],
                        "gold": r["answer"].split("####")[-1].strip()})
    elif name == "gpqa":
        rng = random.Random(0)                         # simple-evals: one RNG, one permutation per example, in order
        for i, r in enumerate(raw):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            perm = rng.sample(range(4), 4)
            ch = [choices[j] for j in perm]
            q = GPQA_TEMPLATE.format(Question=r["Question"], A=ch[0], B=ch[1], C=ch[2], D=ch[3])
            out.append({"id": "gpqa-%03d" % i, "messages": [{"role": "user", "content": q}], "gold": "ABCD"[ch.index(r["Correct Answer"])]})
    elif name == "humaneval":
        for r in raw:
            q = ("Complete the following Python function. Reply with the complete function, including the signature and "
                 "any imports it needs, in a single ```python code block.\n\n" + r["prompt"])
            out.append({"id": r["task_id"].replace("/", "-"), "messages": [{"role": "user", "content": q}],
                        "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]})
    elif name == "ifbench":
        for r in raw:
            out.append({"id": "ifbench-%s" % r["key"], "messages": [{"role": "user", "content": r["prompt"]}], "prompt": r["prompt"]})
    return out


# ---------------------------------------------------------------- model calls
def headers():
    h = {"Content-Type": "application/json"}
    if KEYFILE:
        k = Path(KEYFILE).expanduser().read_text().strip()
        if not k:
            raise SystemExit("ABORT: empty key file %s" % KEYFILE)
        h["Authorization"] = "Bearer " + k
    return h


def call(messages, max_tokens, idle_timeout=600):
    """Streamed request. idle_timeout is the longest silence allowed between chunks, not a limit on the whole answer."""
    body = json.dumps({"model": MODEL, "messages": messages, **SAMPLING, "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, data=body, headers=headers())
    t0 = time.time()
    content, rchars, finish, usage = [], 0, None, {}
    with urllib.request.urlopen(req, timeout=idle_timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                rchars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {"content": "".join(content), "reasoning_chars": rchars, "finish_reason": finish, "usage": usage,
            "max_tokens": max_tokens, "seconds": round(time.time() - t0, 1)}


def check():
    murl = URL.rsplit("/chat/completions", 1)[0] + "/models"
    req = urllib.request.Request(murl, headers=headers())
    models = [m.get("id") for m in json.load(urllib.request.urlopen(req, timeout=30)).get("data", [])]
    print("models:", models)
    r = call([{"role": "user", "content": "What is 17 multiplied by 23? Reply with the number only."}], 800, idle_timeout=300)
    ok = MODEL in models and "391" in r["content"] and r["usage"].get("completion_tokens")
    print("canary (streamed): content=%r finish=%s tokens=%s %.1fs" % (r["content"].strip(), r["finish_reason"], r["usage"].get("completion_tokens"), r["seconds"]))
    print("CHECK_" + ("PASS" if ok else "FAIL"))


def resp_path(name, cap=None):
    return RES / ("%s_responses%s.jsonl" % (name, "" if cap is None else "_cap%d" % cap))


def rerun_caps(name):
    """Caps of the re-run rounds on disk for this benchmark, lowest first."""
    caps = []
    for f in RES.glob("%s_responses_cap*.jsonl" % name):
        try:
            caps.append(int(f.stem.rsplit("_cap", 1)[1]))
        except ValueError:
            pass
    return sorted(caps)


def read_jsonl(p):
    out = {}
    if p.exists():
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                out[r["id"]] = r                 # the last good record for an id wins
    return out


def run(name, limit=None, workers=4, cap=None, only_truncated=False):
    p = DATA / SOURCES[name][1]
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()[:16] != EXPECT_SHA16[name]:
        raise SystemExit("ABORT: %s is missing or differs from the published dataset; run fetch first" % p)
    RES.mkdir(parents=True, exist_ok=True)
    max_tokens = cap or MAX_TOKENS[name]
    out = resp_path(name, cap)
    done = set(read_jsonl(out))
    pool = items(name)
    if only_truncated:
        if cap is None:
            raise SystemExit("--only-truncated needs --cap")
        lower = [c for c in rerun_caps(name) if c < cap]
        base = responses(name, lower[-1] if lower else None)
        cut = {i for i, r in base.items() if r.get("finish_reason") == "length"}
        print("%s %d answers still cut off after the %s round(s); re-asking those at %d" % (
            now(), len(cut), ", ".join(str(c) for c in [MAX_TOKENS[name]] + lower), cap))
        pool = [it for it in pool if it["id"] in cut]
    todo = [it for it in pool if it["id"] not in done]
    if limit:
        todo = todo[:limit]
    total, t0, lock = len(todo), time.time(), threading.Lock()
    state = {"n": 0, "err": 0, "tok": 0}
    print("%s RUN %s: %d to do (%d already done), %d at a time, max_tokens %d, results -> %s" % (
        now(), name, total, len(done), workers, max_tokens, out))
    stop = threading.Event()

    def ticker():
        while not stop.wait(60):
            with lock:
                n, e, tok = state["n"], state["err"], state["tok"]
            el = time.time() - t0
            eta = (el / n * (total - n)) if n else 0
            print("  [%s] %d/%d done, %d errors, %.0f output tok/s, elapsed %dm, ETA %dm" % (
                now(), n, total, e, tok / el if el else 0, el // 60, eta // 60), flush=True)

    threading.Thread(target=ticker, daemon=True).start()

    def one(it):
        try:
            r = call(it["messages"], max_tokens)
            rec = {"id": it["id"], **r}
        except Exception as e:
            rec = {"id": it["id"], "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        return rec

    with ThreadPoolExecutor(workers) as ex, open(out, "a") as f:
        for fut in as_completed([ex.submit(one, it) for it in todo]):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                state["n"] += 1
                state["err"] += 1 if rec.get("error") else 0
                state["tok"] += (rec.get("usage") or {}).get("completion_tokens") or 0
    stop.set()
    print("%s RUN_DONE %s: %d finished, %d errors (re-run the same command to retry errors), %dm" % (
        now(), name, state["n"], state["err"], (time.time() - t0) // 60))


# ---------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d * 100, (c + h) / d * 100)


def responses(name, cap=None):
    """Primary answers; with cap, answers re-run at that cap replace the primary ones for the same items.
    An answer cut off at its token cap is scored as empty (wrong), whatever partial text came back."""
    out = read_jsonl(resp_path(name))
    if cap is not None:
        for c in [c for c in rerun_caps(name) if c <= cap]:   # rounds applied lowest cap first
            out.update(read_jsonl(resp_path(name, c)))
    for r in out.values():
        if r.get("finish_reason") == "length":
            r["partial_chars"] = len(r.get("content") or "")
            r["content"] = ""
    return out


def num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def last_python_block(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


SUMMARY = {}


def image_digest(img):
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", img],
                             capture_output=True, text=True).stdout.strip()
        return out or img
    except Exception:
        return img


def report(name, k, n, extra="", kind="main"):
    lo, hi = wilson(k, n)
    src, ref = REFERENCE[name]
    within = lo <= ref <= hi
    print("%s SCORE %s: %d/%d = %.1f%% (95%% interval %.1f-%.1f)%s" % (now(), name, k, n, 100 * k / n if n else 0, lo, hi, extra))
    print("  reference %.1f%% (%s): %s" % (ref, src, "INSIDE our interval - consistent" if within else "OUTSIDE our interval - investigate"))
    SUMMARY.setdefault("scores", {})[kind] = {"correct": k, "total": n, "percent": round(100 * k / n, 2) if n else 0,
                                              "wilson95": [round(lo, 2), round(hi, 2)], "reference": {"source": src, "percent": ref},
                                              "consistent": within}


def score(name, cap=None):
    its = {it["id"]: it for it in items(name)}
    rs = responses(name, cap)
    trunc = sum(1 for r in rs.values() if r.get("finish_reason") == "length")
    partial = sum(1 for r in rs.values() if r.get("partial_chars"))
    missing = [i for i in its if i not in rs]
    rounds = [c for c in rerun_caps(name) if cap is not None and c <= cap]
    rerun = {c: len(read_jsonl(resp_path(name, c))) for c in rounds}
    caps = sorted({r.get("max_tokens") or MAX_TOKENS[name] for r in rs.values()})
    SUMMARY.clear()
    SUMMARY.update({"benchmark": name, "pubbench": VERSION, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": MODEL,
                    "sampling": SAMPLING, "primary_cap": MAX_TOKENS[name], "rerun_cap": cap, "rerun_rounds": {str(c): n for c, n in rerun.items()},
                    "caps_in_use": caps, "responses": len(rs), "missing": len(missing), "cut_off": trunc})
    if cap is not None:
        print("%s %s: primary cap %d; re-asked and substituted: %s" % (now(), name, MAX_TOKENS[name],
              ", ".join("%d at cap %d" % (n, c) for c, n in rerun.items()) or "none"))
    print("%s %s: %d responses, %d missing, %d cut off at max_tokens (scored as wrong; %d of them had partial text, discarded)" % (
        now(), name, len(rs), len(missing), trunc, partial))
    SUMMARY["cut_off_with_partial_text"] = partial
    if name == "gsm8k":
        k = 0
        for i, it in its.items():
            r = rs.get(i)
            if not r:
                continue
            m = re.findall(r"(?i)answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", r["content"]) or re.findall(r"-?[\d,]*\.?\d+", r["content"])
            got, want = (num(m[-1]) if m else None), num(it["gold"])
            k += 1 if (got is not None and want is not None and abs(got - want) < 1e-6) else 0
        report(name, k, len(its), "" if not missing else " (missing items count as wrong)")
    elif name == "gpqa":
        k = 0
        for i, it in its.items():
            r = rs.get(i)
            m = re.search(GPQA_PATTERN, r["content"]) if r else None
            k += 1 if (m and m.group(1).upper() == it["gold"]) else 0
        report(name, k, len(its), "" if not missing else " (missing items count as wrong)")
    elif name == "humaneval":
        wd = ROOT / "humaneval_exec"
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        for i, it in its.items():
            code = last_python_block(rs[i]["content"]) if i in rs else ""
            if ("def " + it["entry_point"]) not in code:
                code = it["prompt"] + code
            (wd / (i + ".py")).write_text(code + "\n\n" + it["test"] + "\n\ncheck(%s)\n" % it["entry_point"])
        runner = ("import glob,subprocess,os\n"
                  "for f in sorted(glob.glob('/w/*.py')):\n"
                  "    try:\n"
                  "        p=subprocess.run(['python3',f],capture_output=True,timeout=15,cwd='/tmp')\n"
                  "        s='PASS' if p.returncode==0 else 'FAIL'\n"
                  "    except subprocess.TimeoutExpired:\n"
                  "        s='TIMEOUT'\n"
                  "    print(os.path.basename(f)[:-3],s,flush=True)\n")
        (ROOT / "he_runner.py").write_text(runner)
        print("  running %d programs in a network-less, read-only container (%s), 15 s limit each" % (len(its), EXEC_IMAGE))
        SUMMARY["container_image"] = image_digest(EXEC_IMAGE)
        cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m", "--memory", "2g",
               "--cpus", "2", "--pids-limit", "256", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "-v", "%s:/w:ro" % wd, "-v", "%s:/runner.py:ro" % (ROOT / "he_runner.py"), EXEC_IMAGE, "python3", "/runner.py"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        res = dict(l.split() for l in p.stdout.splitlines() if len(l.split()) == 2)
        if p.returncode != 0 and not res:
            print(p.stderr[-2000:])
            print("SCORE_FAIL humaneval: the container did not run"); return
        k = sum(1 for v in res.values() if v == "PASS")
        print("  results: %d PASS, %d FAIL, %d TIMEOUT" % (k, sum(v == "FAIL" for v in res.values()), sum(v == "TIMEOUT" for v in res.values())))
        report(name, k, len(its))
    elif name == "ifbench":
        wd = ROOT / ("ifbench_eval" if cap is None else "ifbench_eval_cap%d" % cap)
        wd.mkdir(parents=True, exist_ok=True)
        with open(wd / "beam-responses.jsonl", "w") as f:
            for i, it in its.items():
                f.write(json.dumps({"prompt": it["prompt"], "response": rs[i]["content"] if i in rs else ""}) + "\n")
        shutil.copy(DATA / SOURCES["ifbench"][1], wd / "IFBench_test.jsonl")
        script = ("set -e; mkdir /opt/IFBench; cd /opt/IFBench; git init -q; git remote add origin https://github.com/allenai/IFBench; "
                  "git fetch -q --depth 1 origin " + IFBENCH_COMMIT + "; git checkout -q FETCH_HEAD; "
                  "git rev-parse HEAD > /w/ifbench_commit.txt; pip install -q -r requirements.txt; "
                  "python -m spacy download -q en_core_web_sm >/dev/null 2>&1 || true; "
                  "pip freeze > /w/ifbench_pip_freeze.txt; "
                  "python -c \"import nltk; [nltk.download(x, quiet=True) for x in ('punkt','punkt_tab')]\"; "
                  "python -m run_eval --input_data=/w/IFBench_test.jsonl --input_response_data=/w/beam-responses.jsonl --output_dir=/w/eval")
        print("  scoring with AllenAI's official IFBench code in a throwaway %s container (downloads the scorer and its packages)" % IFB_IMAGE)
        p = subprocess.run(["docker", "run", "--rm", "-v", "%s:/w" % wd, IFB_IMAGE, "bash", "-c", script], capture_output=True, text=True)
        print("\n".join(p.stdout.splitlines()[-25:]))
        if p.returncode != 0:
            print(p.stderr[-3000:]); print("SCORE_FAIL ifbench: the official scorer did not finish"); return
        for kind in ("loose", "strict"):
            fs = list((wd / "eval").glob("*%s*.jsonl" % kind))
            if not fs:
                print("  no %s result file found" % kind); continue
            rows = [json.loads(l) for l in open(fs[0]) if l.strip()]
            k = sum(1 for r in rows if r.get("follow_all_instructions"))
            print("  prompt-level %s:" % kind)
            report(name, k, len(rows), " [%s]" % kind, kind=kind)
        commit = (wd / "ifbench_commit.txt").read_text().strip() if (wd / "ifbench_commit.txt").exists() else "?"
        SUMMARY["scorer_commit"] = commit
        SUMMARY["container_image"] = image_digest(IFB_IMAGE)
        fz = wd / "ifbench_pip_freeze.txt"
        SUMMARY["scorer_packages"] = [l.strip() for l in open(fz)] if fz.exists() else []
        print("  scorer packages recorded: %d (spaCy model %s)" % (len(SUMMARY["scorer_packages"]),
              next((l for l in SUMMARY["scorer_packages"] if l.lower().startswith("en-core-web-sm")), "not found")))
        print("  scorer commit:", commit)
    for f, src in [(DATA / SOURCES[name][1], "dataset_sha256")]:
        SUMMARY[src] = hashlib.sha256(f.read_bytes()).hexdigest()
    sp = RES / ("%s_score%s.json" % (name, "" if cap is None else "_cap%d" % cap))
    sp.write_text(json.dumps(SUMMARY, indent=2) + "\n")
    print("  summary -> %s" % sp)
    print("SCORE_DONE " + name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "check", "run", "score"])
    ap.add_argument("bench", nargs="?", choices=list(SOURCES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4, help="requests at once (default 4)")
    ap.add_argument("--cap", type=int, help="token cap for a re-run (results kept separately from the primary run)")
    ap.add_argument("--only-truncated", action="store_true", help="with --cap: re-run only answers that hit the primary cap")
    a = ap.parse_args()
    print("pubbench v%s  endpoint %s" % (VERSION, URL))
    if a.step == "fetch":
        fetch()
    elif a.step == "check":
        check()
    elif not a.bench:
        raise SystemExit("name a benchmark: " + ", ".join(SOURCES))
    elif a.step == "run":
        run(a.bench, a.limit, a.workers, a.cap, a.only_truncated)
    else:
        score(a.bench, a.cap)


if __name__ == "__main__":
    main()
