#!/usr/bin/env python
"""Convert Princeton WordNet (public domain) into a dictionary-style JSONL.

One natural-language entry per synset: "<lemma> (<pos>): <definition>.
Example: '<example>'. Synonyms: <lemmas>." This gives the LM real
lexical knowledge — definitions, word senses, examples, synonymy.

Usage:
    python scripts/wordnet_to_jsonl.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nltk

HF_REPO = "thefinalboss/fractus-datasets"
POS_NAME = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def main():
    out = "data/_wordnet/wordnet.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Ensure the corpus is present (small, one-time).
    try:
        from nltk.corpus import wordnet as wn
        list(wn.synsets("test"))
    except LookupError:
        print("downloading wordnet corpus…", flush=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

    n = 0
    total_chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for syn in wn.all_synsets():
            lemmas = sorted(set(l.name().replace("_", " ") for l in syn.lemmas()))
            if not lemmas:
                continue
            head = lemmas[0]
            pos = POS_NAME.get(syn.pos(), syn.pos())
            definition = (syn.definition() or "").strip()
            if not definition:
                continue
            examples = syn.examples()
            parts = [f"{head} ({pos}): {definition}."]
            if examples:
                parts.append("Example: " + " ".join(examples))
            if len(lemmas) > 1:
                parts.append("Synonyms: " + ", ".join(lemmas[1:8]))
            text = " ".join(parts)
            f.write(json.dumps({
                "text": text,
                "word": head,
                "pos": pos,
                "source": f"wordnet/{syn.name()}",
            }, ensure_ascii=False) + "\n")
            n += 1
            total_chars += len(text)

    print(f"{n} synset entries, {total_chars:,} chars (~{total_chars//4:,} tokens)",
          flush=True)

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=out,
                    path_in_repo="dictionaries/wordnet.jsonl",
                    repo_id=HF_REPO, repo_type="dataset")
    print(f"  → uploaded to {HF_REPO}/dictionaries/wordnet.jsonl", flush=True)


if __name__ == "__main__":
    main()
