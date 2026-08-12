#!/usr/bin/env python
"""Fetch public-domain esoteric/occult/masonic books from Project Gutenberg
via the Gutendex API, strip the boilerplate, and package as JSONL.

Only books with copyright == false (or null) are taken — genuinely public
domain. Uploaded to the HF dataset repo so build_corpus.py consumes them.

Usage:
    python scripts/gutenberg_to_jsonl.py                       # default esoteric topics
    python scripts/gutenberg_to_jsonl.py --topics magic alchemy
"""
import argparse, os, sys, json, re, time, io
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPO = "thefinalboss/fractus-datasets"

# Topics that surface the wanted corpus. Queried against Gutendex.
ESOTERIC_TOPICS = [
    "occultism", "magic", "alchemy", "mysticism", "rosicrucian",
    "hermeticism", "tarot", "theosophy", "kabbalah", "masonry",
    "freemasonry", "gnosticism", "witchcraft", "demonology", "astrology",
    "divination", "spiritualism", "esoteric", "ritual", "symbolism",
]

UA = {"User-Agent": "fractus-dataset/1.0"}


def http_get(url, timeout=120, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def query_topic(topic, max_pages=4):
    """Yield book dicts for a topic (paginated). Only public-domain."""
    url = f"https://gutendex.com/books/?topic={topic}"
    pages = 0
    seen = set()
    while url and pages < max_pages:
        try:
            data = json.loads(http_get(url, timeout=60).decode("utf-8"))
        except Exception as e:
            print(f"    gutendex page failed: {e}", flush=True)
            break
        for b in data.get("results", []):
            if b["id"] in seen:
                continue
            seen.add(b["id"])
            # copyright: False or None => public domain. True => skip.
            if b.get("copyright") is True:
                continue
            yield b
        url = data.get("next")
        pages += 1
        time.sleep(1)  # be polite to the API


def pick_text_url(formats):
    """Prefer utf-8 plain text; fall back to us-ascii."""
    for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii",
                "text/plain"):
        if key in formats:
            return formats[key]
    return None


# Project Gutenberg boilerplate markers (handle older + newer phrasings).
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS)?\s*PROJECT GUTENBERG.*?\*\*\*",
                      re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS)?\s*PROJECT GUTENBERG.*?\*\*\*",
                    re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text):
    """Cut to the text between the Gutenberg START/END markers."""
    s = START_RE.search(text)
    e = END_RE.search(text)
    if s and e and e.start() > s.end():
        return text[s.end():e.start()].strip()
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Fetch public-domain books from Gutenberg")
    ap.add_argument("--topics", nargs="*", default=ESOTERIC_TOPICS)
    ap.add_argument("--out", default="data/_gutenberg/esoteric.jsonl")
    ap.add_argument("--max-per-topic", type=int, default=60)
    ap.add_argument("--min-chars", type=int, default=5000,
                    help="skip tiny fragments (likely indexes)")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"=== Gutenberg fetch: topics={args.topics} ===", flush=True)

    # Collect unique books across all topics.
    books = {}
    for topic in args.topics:
        n = 0
        for b in query_topic(topic):
            if b["id"] not in books:
                books[b["id"]] = b
                n += 1
            if n >= args.max_per_topic:
                break
        print(f"  {topic:<14} -> {len(books)} unique total", flush=True)
    print(f"\n{len(books)} unique public-domain books to fetch.\n", flush=True)

    n_ok, n_skip, total_chars = 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        for bid, b in books.items():
            url = pick_text_url(b.get("formats", {}))
            if not url:
                n_skip += 1
                continue
            try:
                raw = http_get(url, timeout=180)
                text = raw.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"  skip #{bid}: {str(e)[:60]}", flush=True)
                n_skip += 1
                continue
            text = strip_boilerplate(text)
            if len(text) < args.min_chars:
                n_skip += 1
                continue
            authors = ", ".join(a.get("name", "?") for a in b.get("authors", []))
            out.write(json.dumps({
                "text": text,
                "title": b.get("title", "?"),
                "author": authors,
                "source": f"gutenberg/{bid}",
            }, ensure_ascii=False) + "\n")
            total_chars += len(text)
            n_ok += 1
            if n_ok % 5 == 0:
                print(f"  {n_ok}/{len(books)} fetched "
                      f"(~{total_chars//4:,} tokens so far)", flush=True)
            time.sleep(0.5)  # politeness

    print(f"\nFetched {n_ok} books ({n_skip} skipped), "
          f"{total_chars:,} chars (~{total_chars//4:,} tokens)", flush=True)

    if not args.no_upload and n_ok:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.out,
            path_in_repo="gutenberg/esoteric.jsonl",
            repo_id=HF_REPO, repo_type="dataset")
        print(f"  → uploaded to {HF_REPO}/gutenberg/esoteric.jsonl", flush=True)


if __name__ == "__main__":
    main()
