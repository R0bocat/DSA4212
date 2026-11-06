"""
drop_flagged.py - remove the documents listed by overlap_check.py --flagged-out from their clean_*.jsonl files.

  python drop_flagged.py flagged.txt              # rewrites the files (a .bak copy is kept next to each)
  python drop_flagged.py flagged.txt --dry-run    # only report what would be removed

Afterwards re-encode just the affected splits (fast), e.g. for PubMed val/test:
  python prepare_data.py --src data/pubmed_raw --out data/pubmed_v2 --skip-clean \
         --tokenizer tokenizers/mix8192.json --encode-splits val test
"""
import argparse
import collections
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flagged", help="file with 'path<TAB>doc id' lines")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ids_by_file = collections.defaultdict(set)
    with open(a.flagged, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                path, doc_id = line.split("\t", 1)
                ids_by_file[path].add(doc_id)

    for path, ids in ids_by_file.items():
        kept, removed = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if json.loads(line)["id"] in ids:
                    removed += 1
                else:
                    kept.append(line)
        print(f"{path}: flagged {len(ids)}, found and {'would remove' if a.dry_run else 'removing'} {removed}")
        if a.dry_run or removed == 0:
            continue
        if not os.path.exists(path + ".bak"):
            shutil.copy(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
