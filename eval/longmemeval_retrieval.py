#!/usr/bin/env python3
"""LLM-free retrieval evaluation of VCC search on LongMemEval_S.

LongMemEval (Wu et al., ICLR 2025, arXiv:2410.10813) gives, per question, a haystack of
~50 chat sessions and marks the turns that contain the evidence (`has_answer`). This
script writes every session as its own chat JSONL (VCC's generic `chat` adapter), runs
the real VCC pipeline (compile_pass + grep_search, exactly what `--grep` prints), maps
each printed hit back to its session and turn, and measures whether evidence shows up
near the top of the output.

Query: the question's content words as a case-insensitive alternation, which is what an
agent typing `--grep` would do without paraphrasing (a deliberately weak, LLM-free query).

Orderings compared on the same matched blocks:
  recency   VCC default: newest first
  coverage  naive: number of distinct query words in the block (ties newest first)
  bm25      VCC --rank

Metrics (abstention questions, which have no evidence, are skipped):
  turn_any@k   any evidence turn among the first k printed blocks
  turn_all@k   fraction of evidence turns among the first k blocks
  sess@k       fraction of evidence sessions with a block among the first k
  matched      fraction of evidence turns the regex matches at all (upper bound)

Usage:
  python eval/longmemeval_retrieval.py path/to/longmemeval_s_cleaned.json [--n N] [--out results.json]
Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT).
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "conversation-compiler", "scripts"))
import VCC  # noqa: E402

KS = (5, 10, 30)
STOP = set("""a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing done down during each
few for from further get got had has have having he her here hers herself him himself his how
i if in into is it its itself just let me more most much my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that
the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours
yourself yourselves also many make made like one two tell know think thing things remember
recall mentioned mention previous previously last time times did ago ever""".split())

HIT_RE = re.compile(r"^\((.+?):(\d+)-(\d+)\) \[(\w+)\]")


def iso(date_str):
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00Z'."""
    d = datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", date_str).strip(), "%Y/%m/%d %H:%M")
    return d.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def query_regex(question):
    words = []
    for w in re.findall(r"[A-Za-z0-9']+", question.lower()):
        w = w.strip("'")
        if len(w) >= 3 and w not in STOP and w not in words:
            words.append(w)
    return ("(?i)" + "|".join(re.escape(w) for w in words)) if words else None, words


def run_question(q, workdir):
    """Compile the haystack, search, and return per-ordering ranked (session, turn) lists."""
    sessions = list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]))
    sessions.sort(key=lambda s: iso(s[1]))                    # oldest first, like mtime order
    results, where = [], {}                                   # (relpath, start) -> (sid, turn)
    for sid, date, turns in sessions:
        path = os.path.join(workdir, f"{sid}.jsonl")
        ts = iso(date)
        with open(path, "w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps({"role": t["role"], "content": t["content"], "timestamp": ts}) + "\n")
        res = VCC.compile_pass(path, quiet=True, harness="chat")
        for fp, ir, *_ in res:
            rel = VCC._rel_path(fp)
            nonempty = [i for i, t in enumerate(turns) if t["content"]]
            secs = sorted({o["_sec"] for o in ir if o.get("_sec") is not None})
            assert len(secs) == len(nonempty), (sid, len(secs), len(nonempty))
            sec_turn = dict(zip(secs, nonempty))
            for o in ir:
                if o.get("searchable"):
                    where[(rel, o["start_line"] + 1)] = (sid, sec_turn[o["_sec"]])
        results.extend(res)

    rx, words = query_regex(q["question"])
    if rx is None:
        return None, words
    pattern = re.compile(rx)
    ranked = {}
    for name, rank in (("recency", False), ("bm25", True)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            VCC.grep_search(results, pattern, rank=rank)
        hits = []
        for line in buf.getvalue().split("\n"):
            m = HIT_RE.match(line)
            if m:
                hits.append(where[(m.group(1), int(m.group(2)))])
        ranked[name] = hits
    # naive coverage ranking over the same matched blocks (stable: ties stay newest first)
    text_of = {}
    for fp, ir, *_ in results:
        rel = VCC._rel_path(fp)
        for o in ir:
            if o.get("searchable"):
                text_of[where[(rel, o["start_line"] + 1)]] = "\n".join(o["content"]).lower()
    ranked["coverage"] = sorted(ranked["recency"],
                                key=lambda h: -sum(1 for w in words if w in text_of[h]))
    return ranked, words


def score(hits, ev_turns, ev_sess):
    out = {}
    for k in KS:
        top = hits[:k]
        out[f"turn_any@{k}"] = float(any(h in ev_turns for h in top)) if ev_turns else None
        out[f"turn_all@{k}"] = (sum(1 for t in ev_turns if t in top) / len(ev_turns)) if ev_turns else None
        out[f"sess@{k}"] = len({h[0] for h in top} & ev_sess) / len(ev_sess) if ev_sess else None
    out["matched"] = (sum(1 for t in ev_turns if t in set(hits)) / len(ev_turns)) if ev_turns else None
    out["n_hits"] = len(hits)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data")
    ap.add_argument("--n", type=int, default=0, help="first N questions only")
    ap.add_argument("--out", help="write per-question results as JSON")
    a = ap.parse_args()

    with open(a.data, encoding="utf-8") as f:
        data = json.load(f)
    data = [q for q in data if not q["question_id"].endswith("_abs")]
    if a.n:
        data = data[:a.n]

    rows, t0 = [], time.time()
    cwd = os.getcwd()
    for i, q in enumerate(data):
        ev_turns = {(sid, ti) for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"])
                    for ti, t in enumerate(turns) if t.get("has_answer")}
        ev_sess = set(q["answer_session_ids"])
        work = tempfile.mkdtemp(prefix="vcc_lme_")
        try:
            os.chdir(work)                      # hits print paths relative to cwd
            ranked, words = run_question(q, work)
        finally:
            os.chdir(cwd)
            shutil.rmtree(work, ignore_errors=True)
        row = {"id": q["question_id"], "type": q["question_type"], "words": words}
        if ranked is not None:
            for name, hits in ranked.items():
                row[name] = score(hits, ev_turns, ev_sess)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(data)} questions, {time.time() - t0:.0f}s", file=sys.stderr)

    names = ("recency", "coverage", "bm25")
    metrics = [f"turn_any@{k}" for k in KS] + [f"turn_all@{k}" for k in KS] + [f"sess@{k}" for k in KS]
    print(f"LongMemEval_S retrieval, {len(rows)} questions (abstention excluded), "
          f"{time.time() - t0:.0f}s\n")
    print(f"{'metric':<14}" + "".join(f"{n:>10}" for n in names))
    for m in metrics + ["matched"]:
        print(f"{m:<14}" + "".join(f"{mean(r[n][m] for r in rows if n in r):>10.3f}" for n in names))
    print(f"{'hits/question':<14}" + "".join(f"{mean(r[n]['n_hits'] for r in rows if n in r):>10.0f}"
                                           for n in names))
    print("\nturn_any@10 by question type:")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items()):
        print(f"  {t:<26} n={len(rs):<4}" + "".join(
            f"{n}={mean(r[n]['turn_any@10'] for r in rs if n in r):.3f}  " for n in names))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
