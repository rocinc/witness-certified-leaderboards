"""LLM relevance judging for WITNESS testbeds (DeepSeek, Anthropic-compatible endpoint), cached and resumable.

A content-moderation refusal from the API is an outcome, not an error: it is recorded with status "refused" and no
label, and the evaluator treats it as its own judge state.  Transient failures are retried; anything else raises.
Cache key sha1(model|system|prompt) is identical to the DuRetrieval runs, so their cache is reused read-only.
Env (600-permission file outside the repo): DS_API_KEY, DS_BASE_URL; model given on the command line.
--en selects the English prompt (D2); the default is the Chinese prompt (D1, D3).
usage: python scripts/data/judge.py <items.jsonl> <out.jsonl> <model> <shared_cache.jsonl> [n_threads] [--en]
"""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from witness import judge_prompt  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
args = [a for a in sys.argv[1:] if a != "--en"]
if "--en" in sys.argv:
    PASSAGE_CHARS, SYSTEM, TEMPLATE = judge_prompt.PASSAGE_CHARS_EN, judge_prompt.SYSTEM_EN, judge_prompt.TEMPLATE_EN
else:
    PASSAGE_CHARS, SYSTEM, TEMPLATE = judge_prompt.PASSAGE_CHARS, judge_prompt.SYSTEM, judge_prompt.TEMPLATE
items_path, out_path, MODEL, shared = args[:4]
n_threads = int(args[4]) if len(args) > 4 else 16
URL = os.environ["DS_BASE_URL"] + "/v1/messages"
HEADERS = {"x-api-key": os.environ["DS_API_KEY"], "anthropic-version": "2023-06-01"}
OWN = os.path.join(os.path.dirname(out_path), f"judge_cache_{MODEL}.jsonl")
client = httpx.Client(timeout=180)
lock = threading.Lock()
cache = {}
for path in (shared, OWN):
    if os.path.exists(path):
        for line in open(path, encoding="utf8"):
            r = json.loads(line)
            cache[r["key"]] = r


def call(body):
    for attempt in range(40):
        try:
            r = client.post(URL, headers=HEADERS, json=body)
        except httpx.TransportError:
            time.sleep(min(2 ** attempt, 60))
            continue
        # an account problem (balance, auth) must stop the run and reach the user; it is never retried or skipped
        assert "Insufficient Balance" not in r.text and r.status_code not in (401, 402, 403), r.text[:300]
        if r.status_code in (429, 500, 502, 503, 504, 529):
            time.sleep(min(2 ** attempt, 60))
            continue
        return r
    raise RuntimeError("api failed after 40 retries")


def judge(item):
    prompt = TEMPLATE.format(q=item["query"], p=item["passage"][:PASSAGE_CHARS])
    key = hashlib.sha1((MODEL + "|" + SYSTEM + "|" + prompt).encode()).hexdigest()
    if key in cache:
        rec = cache[key]
    else:
        body = {"model": MODEL, "max_tokens": 200, "temperature": 0, "system": SYSTEM, "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}]}
        t0 = time.time()
        r = call(body)
        if r.status_code == 400 and "Content Exists Risk" in r.text:
            rec = dict(key=key, status="refused", label=None, reason=r.text[:200], in_tok=0, out_tok=0, model=MODEL)
        else:
            assert r.status_code == 200, r.text[:300]
            j = r.json()
            text = "".join(b["text"] for b in j["content"] if b["type"] == "text")
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
            assert parsed["label"] in (0, 1, 2), text
            rec = dict(key=key, status="ok", label=parsed["label"], reason=parsed.get("reason", ""),
                       in_tok=j["usage"]["input_tokens"], out_tok=j["usage"]["output_tokens"],
                       sec=round(time.time() - t0, 2), model=MODEL)
        with lock:
            cache[key] = rec
            with open(OWN, "a", encoding="utf8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = dict(rec)
    out.setdefault("status", "ok")
    out["id"] = item["id"]
    return out


items = [json.loads(line) for line in open(items_path, encoding="utf8")]
done = refused = 0
with ThreadPoolExecutor(n_threads) as ex, open(out_path, "w", encoding="utf8") as out:
    for r in ex.map(judge, items):
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        done += 1
        refused += r["status"] == "refused"
        if done % 1000 == 0:
            print("judged", done, "/", len(items), "refused", refused)
own = [json.loads(line) for line in open(OWN, encoding="utf8")] if os.path.exists(OWN) else []
print("DONE", len(items), "refused", refused, "new_in_tok", sum(r["in_tok"] for r in own),
      "new_out_tok", sum(r["out_tok"] for r in own))
