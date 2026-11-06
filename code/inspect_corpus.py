"""
inspect_corpus.py - See what is inside a Hugging Face dataset repository BEFORE
downloading tens of GB of it.

Setup:   pip install huggingface_hub pyarrow
Run:     python inspect_corpus.py --repo OWNER/NAME --prefix "folder/"          # folder/size summary + dataset card
         python inspect_corpus.py --repo OWNER/NAME --sample                    # also download+preview a few small files
         python inspect_corpus.py --repo OWNER/NAME --download "folder/*.parquet" --out data/raw
                                                                                # download only files matching patterns
Examples used in this project are in readme.md.
"""
import argparse
import collections
import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEFAULT_REPO = "HuggingFaceFW/fineweb-edu"
TEXT_EXTS = {".txt", ".md", ".jsonl", ".parquet", ".json", ".arrow"}


def human(n):
    n = float(n or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024


def group_of(path, depth):
    parts = path.split("/")[:-1][:depth]
    return "/".join(parts) if parts else "(root)"


def preview_file(path):
    ext = os.path.splitext(path)[1].lower()
    print(f"\n--- preview: {path} ---")
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            print("schema:", pf.schema_arrow)
            batch = next(pf.iter_batches(batch_size=1))
            row = batch.to_pylist()[0]
            for k, v in row.items():
                s = str(v).replace("\n", "\\n")
                print(f"  {k}: {s[:300]}")
        except Exception as e:
            print(f"  (could not read parquet: {e})")
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        print(f.read(1200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--depth", type=int, default=2, help="folder depth used to group the summary")
    ap.add_argument("--prefix", default="", help="only look at files under this folder, e.g. 'PubMed Abstracts/'")
    ap.add_argument("--top", type=int, default=60, help="number of folders to list")
    ap.add_argument("--list", type=int, default=0, metavar="N",
                    help="also print the first N individual files (with sizes) under --prefix")
    ap.add_argument("--sample", action="store_true", help="download and preview a few small files")
    ap.add_argument("--download", nargs="+", metavar="PATTERN",
                    help="download only files matching these glob patterns")
    ap.add_argument("--out", default="data/raw", help="destination for --download")
    a = ap.parse_args()

    api = HfApi()
    info = api.dataset_info(a.repo, files_metadata=True)
    files = [s for s in (info.siblings or []) if s.rfilename.startswith(a.prefix)]
    print(f"Repo: {a.repo}  |  files{' under ' + repr(a.prefix) if a.prefix else ''}: {len(files):,}")

    # ---- dataset card -----------------------------------------------------
    try:
        readme = hf_hub_download(a.repo, "README.md", repo_type="dataset")
        with open(readme, encoding="utf-8", errors="replace") as f:
            print("\n=== README.md (first 4000 chars) ===")
            print(f.read(4000))
    except Exception as e:
        print(f"(no README.md readable: {e})")

    # ---- folder summary ---------------------------------------------------
    groups = collections.defaultdict(lambda: {"n": 0, "size": 0, "exts": collections.Counter()})
    total = 0
    for s in files:
        g = group_of(s.rfilename, a.depth)
        ext = os.path.splitext(s.rfilename)[1].lower() or "(none)"
        groups[g]["n"] += 1
        groups[g]["size"] += s.size or 0
        groups[g]["exts"][ext] += 1
        total += s.size or 0

    print(f"\n=== Folder summary (depth={a.depth}), total {human(total)} ===")
    print(f"{'folder':<55}{'files':>9}{'size':>12}  extensions")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["size"])
    for g, d in ranked[: a.top]:
        exts = ", ".join(f"{e}:{c}" for e, c in d["exts"].most_common(4))
        print(f"{g[:54]:<55}{d['n']:>9,}{human(d['size']):>12}  {exts}")
    if len(ranked) > a.top:
        print(f"... and {len(ranked) - a.top} more folders (raise --top to see them)")

    if a.list:
        print(f"\n=== first {a.list} files under {a.prefix!r} ===")
        for s_ in sorted(files, key=lambda x: x.rfilename)[: a.list]:
            print(f"{human(s_.size):>12}  {s_.rfilename}")

    # ---- sample previews --------------------------------------------------
    if a.sample:
        def previewable(s):
            ext = os.path.splitext(s.rfilename)[1].lower()
            cap = 60_000_000 if ext == ".parquet" else 5_000_000
            return ext in TEXT_EXTS and 2_000 < (s.size or 0) < cap
        cands = [s for s in files if previewable(s)]
        cands.sort(key=lambda s: s.rfilename)
        step = max(1, len(cands) // 4)
        for s in cands[::step][:4]:
            try:
                p = hf_hub_download(a.repo, s.rfilename, repo_type="dataset")
                preview_file(p)
            except Exception as e:
                print(f"(could not download {s.rfilename}: {e})")

    # ---- selective download ----------------------------------------------
    if a.download:
        print(f"\nDownloading files matching {a.download} into {a.out} ...")
        snapshot_download(a.repo, repo_type="dataset", allow_patterns=a.download, local_dir=a.out)
        print("done.")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
