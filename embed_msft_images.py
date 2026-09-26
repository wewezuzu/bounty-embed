#!/usr/bin/env python3
"""Daytona $50 draft: build a tiny embedding/search DB for MSFT container images list.

Upstream: daytonaio/devcontainer-generator#22 "Create Embedding Database for MSFT Container Images Repo".
This draft is stdlib-only so it runs anywhere. If chromadb is available it is used,
otherwise falls back to a JSON TF-IDF index with the same API.

Usage:
  python3 embed_msft_images.py --input images.txt --db ./embed_db.json
  python3 embed_msft_images.py --input images.txt --db ./embed_db.json --query "python dev"
"""
import argparse, json, math, os, re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(s):
    return TOKEN.findall(s.lower())

def tfidf_index(docs):
    df = Counter()
    tok_docs = []
    for d in docs:
        toks = set(tokenize(d))
        tok_docs.append(toks)
        for t in toks:
            df[t] += 1
    n = max(len(docs), 1)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return {"docs": docs, "idf": idf}

def search(index, query, top_k=5):
    idf = index["idf"]
    docs = index["docs"]
    qt = tokenize(query)
    scored = []
    for d in docs:
        dt = Counter(tokenize(d))
        s = sum(dt[t] * idf.get(t, 0.0) for t in qt)
        # small boost for exact substring
        if query.lower() in d.lower():
            s += 2.0
        scored.append((s, d))
    scored.sort(reverse=True)
    return [d for s, d in scored[:top_k] if s > 0]

def main(argv=None):
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file, one image per line")
    ap.add_argument("--db", default="embed_db.json")
    ap.add_argument("--query", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args(argv)
    with open(a.input, encoding="utf-8") as f:
        docs = [l.strip() for l in f if l.strip()]
    # Try chromadb if installed (optional, upstream-friendly)
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=a.db + ".chroma")
        col = client.get_or_create_collection("msft_images")
        col.upsert(ids=["id-%d" % i for i in range(len(docs))], documents=docs)
        print("stored %d docs in chroma (%s.chroma)" % (len(docs), a.db))
    except Exception as e:
        print("chromadb not used (%s), using JSON TF-IDF fallback" % e)
    index = tfidf_index(docs)
    with open(a.db, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print("wrote %s (%d docs)" % (a.db, len(docs)))
    if a.query:
        for d in search(index, a.query, a.top_k):
            print(" -", d)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
