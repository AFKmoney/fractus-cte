#!/usr/bin/env python
"""Convert GitHub repos to JSONL datasets for Fractus training.

Shallow-clones each repo, extracts every text-based file (code, markdown,
docs, config), and writes one JSONL per repo. Uploads each to the HF
dataset repo so build_corpus.py picks it up automatically.

Usage:
    python scripts/repo_to_jsonl.py AFKmoney/kortex AFKmoney/CogNet
    python scripts/repo_to_jsonl.py --batch all        # ~40 high-value repos
    python scripts/repo_to_jsonl.py --batch all --no-upload   # local only
"""
import argparse, os, sys, json, subprocess, tempfile, shutil, time, fnmatch
import urllib.request, io, tarfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPO = "thefinalboss/fractus-datasets"

# Text-bearing extensions (code + prose + config). All feed the LM.
TEXT_EXT = {
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".json5",
    ".md", ".mdx", ".markdown", ".rst", ".txt", ".org", ".tex", ".toml", ".yaml",
    ".yml", ".sh", ".bash", ".zsh", ".ps1", ".go", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cxx", ".java", ".rb", ".lua", ".sql", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".ipynb", ".csv", ".tsv", ".ini", ".cfg", ".conf", ".swift",
    ".kt", ".kts", ".php", ".r", ".jl", ".clj", ".cljs", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".fs", ".nim", ".v", ".sv", ".zig", ".odin", ".dart", ".scala",
    ".groovy", ".gradle", ".makefile", ".mk", ".dockerfile", ".env.example",
    ".gitignore", ".properties", ".tf", ".proto", ".thrift", ".graphql", ".vue",
    ".svelte", ".el", ".lisp", ".scm", ".rkt", ".f90", ".f95", ".asm", ".wat",
}
# Also include extensionless files with these names.
EXTENSIONLESS = {"Makefile", "Dockerfile", "LICENSE", "README", "Rakefile",
                 "CMakeLists", "Cargo", "Gemfile", "Procfile", "Vagrantfile",
                 "Brewfile", "Justfile", "BUILD", "WORKSPACE", "go.mod", "go.sum"}

# Skip these dirs entirely (junk / generated / deps).
SKIP_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv",
             "venv", "env", ".env", "target", ".next", ".cache", "vendor",
             "deps", "_build", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
             ".tox", ".eggs", "site-packages", "coverage", ".nuxt", "out",
             ".gradle", ".terraform", "bower_components", "Pods", "DerivedData",
             ".gitlab", ".circleci"}

MAX_FILE_BYTES = 2_000_000  # skip files bigger than 2MB (generated bloat)

# High-value AFKmoney repos with dense technical content (AI architectures,
# OS, papers, neural nets, tooling). Excludes repos already converted to .pt.
HIGH_VALUE_REPOS = [
    "radical-cognitive-architectures",
    "Fractal-Neural-Network",
    "kortex",
    "CogNet",
    "CogNet-MoE-1B",
    "oscillon-architecture",
    "Modele-Variance-Topologique",
    "kahnn",
    "prism",
    "prism-kb",
    "Alpha-N",
    "aether-ai",
    "aether-engine",
    "nexusOS",
    "nxs",
    "kuramoto-controller",
    "synergion",
    "omega-1",
    "omega2",
    "gguf-knowledge-extractor",
    "AICL",
    "NFRC",
    "lea2",
    "nova-spike-hybrid",
    "z-agent-desktop",
    "morphos",
    "datasetfoundry",
    "omega-hedge-fund",
    "ghost-packet-symbiotic",
    "ghost-packet-runtime",
    "mixstudio",
    "Crowd-Adaptive-Alpha-Swarm",
    "pumpfun-agent",
    "nexusOS-dev-",
    "omega-trader",
]


def should_include(path: str, name: str) -> bool:
    if name in EXTENSIONLESS:
        return True
    ext = os.path.splitext(name)[1].lower()
    return ext in TEXT_EXT


def clone(repo: str, dest: str) -> bool:
    """Download a repo tarball and extract into dest. Returns True on success.

    Uses codeload (HTTPS tarball) — far more reliable than git clone for
    batch ops (no git-protocol fetch-pack failures). Tries main then master,
    with one retry per branch.
    """
    owner = repo.split("/")[0] if "/" in repo else "AFKmoney"
    name = repo.split("/")[-1]
    last_err = ""
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{name}/tar.gz/refs/heads/{branch}"
        for attempt in (1, 2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "fractus-dataset"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    tar.extractall(dest)
                tops = [d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))]
                if tops:
                    stable = os.path.join(dest, "_repo")
                    if os.path.exists(stable):
                        shutil.rmtree(stable)
                    os.rename(os.path.join(dest, tops[0]), stable)
                return True
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:100]}"
                if attempt == 1:
                    time.sleep(3)
    print(f"  DOWNLOAD FAILED ({last_err})", flush=True)
    return False


def convert(repo: str, out_path: str) -> int:
    """Download repo, extract text to JSONL. Returns byte count written."""
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "_repo")
        if not clone(repo, td):
            return 0
        n_files, n_bytes = 0, 0
        with open(out_path, "w", encoding="utf-8") as out:
            for root, dirs, files in os.walk(local):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                           and not d.startswith(".")
                           and "egg-info" not in d]
                for fn in files:
                    if not should_include(root, fn):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        if os.path.getsize(fp) > MAX_FILE_BYTES:
                            continue
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    except Exception:
                        continue
                    if not text.strip():
                        continue
                    rel = os.path.relpath(fp, local)
                    out.write(json.dumps({
                        "text": text,
                        "source": f"{repo}/{rel}",
                        "file": fn,
                    }, ensure_ascii=False) + "\n")
                    n_files += 1
                    n_bytes += len(text)
        print(f"  {n_files} files, {n_bytes:,} chars (~{n_bytes//4:,} tokens)",
              flush=True)
        return n_bytes


def upload(local_path: str, repo_name: str):
    """Upload the JSONL to the HF dataset repo under <repo_name>/."""
    from huggingface_hub import HfApi
    api = HfApi()
    safe = repo_name.split("/")[-1]
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"repos/{safe}/data.jsonl",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"  → uploaded to {HF_REPO}/repos/{safe}/data.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Convert GitHub repos to JSONL")
    ap.add_argument("repos", nargs="*", help="repo names (e.g. AFKmoney/kortex)")
    ap.add_argument("--batch", choices=["all"], help="use the predefined high-value list")
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload")
    ap.add_argument("--out-dir", default="data/_repos_jsonl")
    args = ap.parse_args()

    repos = args.repos
    if args.batch == "all":
        repos = HIGH_VALUE_REPOS
    if not repos:
        ap.error("provide repos or --batch all")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Converting {len(repos)} repos ===", flush=True)
    t0 = time.time()
    grand_bytes = 0
    done = 0
    for i, repo in enumerate(repos, 1):
        name = repo.split("/")[-1]
        print(f"\n[{i}/{len(repos)}] {repo}", flush=True)
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        try:
            nb = convert(repo, out_path)
            if nb == 0:
                continue
            grand_bytes += nb
            done += 1
            if not args.no_upload:
                upload(out_path, repo)
            os.remove(out_path)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"\n=== DONE: {done}/{len(repos)} repos, {grand_bytes:,} chars "
          f"(~{grand_bytes//4:,} tokens) in {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
