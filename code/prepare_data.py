"""
prepare_data.py - Turn a folder of downloaded text into nanoGPT-ready .bin files.

Pipeline
  1. Walk --src for .txt/.md/.jsonl/.parquet files (optionally filtered by --include/--exclude)
  2. Clean text, split into paragraphs, and keep only paragraphs that look like clean English
     (stopword ratio, letter ratio, word length, non-ASCII ratio). This also drops other languages
     and most OCR garbage.
  3. Assign contiguous chunks (default), documents, or whole source folders to train / val / test
     by a deterministic hash.
  4. Remove duplicate paragraphs. Test is built first, then val, then train; train drops any
     paragraph that appears in test or val, so quoted passages cannot leak into evaluation.
  5. Train a byte-level BPE tokenizer on the TRAIN text only.
  6. Encode to uint16 .bin files (train documents are shuffled, so the first N tokens of
     train.bin are a random sample of documents - handy for "data size" experiments).
  7. Write meta.json with token/byte counts for converting loss to bits per byte.

Install:  pip install numpy tokenizers pyarrow     (pyarrow only needed for .parquet)

Example:
  python prepare_data.py --src data/pubmed_raw --out data/pubmed_v2 --filters minimal --vocab-size 8192
"""
import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

EOS = "<|endoftext|>"
DOC_EXTS = {".txt", ".md", ".jsonl", ".parquet", ".arrow"}  # .json is skipped: usually metadata sidecars

# Common English function words. 'his' is excluded on purpose (it also occurs in non-English text).
STOP = set("the of and to that is was he for it with as on be by this had not are but from or an "
           "they which you were her all she there would their we him been has when who will said".split())
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
PARA_SPLIT_RE = re.compile(r"\n\s*\n")
NORM_RE = re.compile(r"[^a-z0-9]+")
PAGE_NUM_RE = re.compile(r"^\d{1,4}$")                  # lines that are only a page number
HYPHEN_RE = re.compile(r"(?<=[a-z])- (?=[a-z])")          # 'jour- neys' -> 'journeys'
CAMEL_RE = re.compile(r"[a-z][A-Z][a-z]")                 # 'theirRansomertobe' (words fused by OCR)
MIX_RE = re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b")  # tokens mixing letters and digits ('1s', 'i14')
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.I)


# ------------------------------------------------------------------ text cleaning
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\u201c\u2018(\[]?[A-Z])")


def split_long(p, max_chars, target=1000):
    """Cut an oversized paragraph into ~target-char pieces at sentence boundaries."""
    if len(p) <= max_chars:
        return [p]
    pieces, cur = [], ""
    for sent in SENT_RE.split(p):
        while len(sent) > max_chars:  # no sentence punctuation: fall back to word windows
            cut = sent.rfind(" ", 0, target)
            cut = cut if cut > 0 else target
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:cut])
            sent = sent[cut:].lstrip()
        if cur and len(cur) + 1 + len(sent) > target:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        pieces.append(cur)
    return pieces


def split_paragraphs(text, unwrap, max_chars=2000, repair_hyphens=True):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CTRL_RE.sub("", text)
    out = []
    for p in PARA_SPLIT_RE.split(text):
        lines = [ln.strip() for ln in p.split("\n")]
        p = (" " if unwrap else "\n").join(ln for ln in lines if ln and not PAGE_NUM_RE.match(ln))
        p = SPACE_RE.sub(" ", p)
        p = (HYPHEN_RE.sub("", p) if repair_hyphens else p).strip()
        if p:
            out.extend(split_long(p, max_chars))
    return out


def check_paragraph(p, a):
    """Return None if the paragraph is kept, otherwise a short reason string."""
    minimal = a.filters == "minimal"   # born-digital text: only basic sanity checks
    words = p.split()
    n = len(words)
    if n < a.min_words:
        return "short"
    nonspace = max(1, len(p) - p.count(" ") - p.count("\n"))
    if sum(c.isalpha() for c in p) / nonspace < (0.6 if minimal else a.min_alpha):
        return "low_alpha"
    if sum(ord(c) > 127 for c in p) / nonspace > (0.2 if minimal else a.max_nonascii):
        return "non_ascii"
    stop = sum(1 for w in words if w.lower().strip(".,;:!?()[]\"'") in STOP)
    if stop / n < (min(a.min_stop, 0.10) if minimal else a.min_stop):
        return "not_english"
    avg = sum(len(w) for w in words) / n
    if not (3.0 <= avg <= (12.0 if minimal else 9.0)):
        return "word_length"
    if minimal:
        return None
    if max(len(w) for w in words) > a.max_word_len:
        return "fused_words"
    if len(CAMEL_RE.findall(p)) >= 2:
        return "fused_words"
    if sum(c.isdigit() for c in p) / nonspace > a.max_digit:
        return "digits"            # footnote lists, tables of references
    mixed = sum(1 for w in MIX_RE.findall(p) if not ORDINAL_RE.match(w))
    if mixed / n > a.max_mixed:
        return "mixed_alnum"       # OCR confusions like '1s', 'i14'
    return None


def para_hash(p):
    return int.from_bytes(hashlib.blake2b(NORM_RE.sub("", p.lower()).encode(), digest_size=8).digest(), "big")


# ------------------------------------------------------------------ source iteration
def group_of(rel, depth):
    parts = rel.split("/")[:-1][:depth]
    return "/".join(parts) if parts else "(root)"


def assign_split(key, a):
    h = hashlib.blake2b(f"{a.seed}:{key}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2**64
    if u < a.test_frac:
        return "test"
    if u < a.test_frac + a.val_frac:
        return "val"
    return "train"


def list_files(a):
    for dirpath, dirnames, filenames in os.walk(a.src):
        dirnames.sort()
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in DOC_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, a.src).replace(os.sep, "/")
            if a.include and not any(fnmatch.fnmatch(rel, p) for p in a.include):
                continue
            if a.exclude and any(fnmatch.fnmatch(rel, p) for p in a.exclude):
                continue
            yield rel, ext, path


def iter_docs(a, want_split):
    """Yield (doc_id, group, text) for documents assigned to `want_split`."""
    for rel, ext, path in list_files(a):
        group = group_of(rel, a.group_depth)

        def wanted(doc_id):
            if a.split_by == "chunk":
                return True  # split happens inside each document (see units_for_split)
            key = group if a.split_by == "group" else doc_id
            return assign_split(key, a) == want_split

        if ext in (".txt", ".md"):
            if not wanted(rel):
                continue
            if os.path.getsize(path) > a.max_file_mb * 2**20:
                print(f"  [skip] {rel}: larger than --max-file-mb")
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                yield rel, group, f.read()
        elif ext == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    doc_id = f"{rel}#{i}"
                    if not wanted(doc_id):
                        continue
                    try:
                        text = json.loads(line).get(a.text_field)
                    except Exception:
                        continue
                    if isinstance(text, str):
                        yield doc_id, group, text
        elif ext == ".arrow":  # Hugging Face datasets shard; needs: pip install datasets
            from datasets import Dataset
            ds = Dataset.from_file(path)
            idx = 0
            for batch in ds.select_columns([a.text_field]).iter(batch_size=256):
                for text in batch[a.text_field]:
                    doc_id = f"{rel}#{idx}"
                    idx += 1
                    if wanted(doc_id) and isinstance(text, str):
                        yield doc_id, group, text
        elif ext == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            idx = 0
            for batch in pf.iter_batches(columns=[a.text_field], batch_size=256):
                for text in batch.column(0).to_pylist():
                    doc_id = f"{rel}#{idx}"
                    idx += 1
                    if wanted(doc_id) and isinstance(text, str):
                        yield doc_id, group, text


# ------------------------------------------------------------------ cleaning pass
def units_for_split(a, split, doc_id, text):
    """Yield (unit_id, paragraphs) for the part of this document that belongs to `split`.

    doc/group mode: the whole document is one unit.
    chunk mode: the document is cut into contiguous blocks of ~a.chunk_chars characters and each
    block is assigned to a split by hash, so even a corpus of a few hundred huge files gives
    sensibly sized val/test sets.
    """
    paras = split_paragraphs(text, not a.keep_linebreaks, a.max_para_chars, a.filters != "minimal")
    if a.split_by != "chunk":
        yield doc_id, paras
        return
    unit, cur_len, cidx = [], 0, 0
    cur_split = assign_split(f"{doc_id}:0", a)
    for p in paras:
        if cur_len >= a.chunk_chars:
            if unit:
                yield f"{doc_id}:{cidx}", unit
            unit, cur_len, cidx = [], 0, cidx + 1
            cur_split = assign_split(f"{doc_id}:{cidx}", a)
        cur_len += len(p)
        if cur_split == split:
            unit.append(p)
    if unit:
        yield f"{doc_id}:{cidx}", unit


def clean_split(a, split, blocked, stats):
    path = os.path.join(a.out, f"clean_{split}.jsonl")
    seen = set()
    n_docs = n_chars = n_src = n_paras = n_bytes = 0
    limit = int(a.max_train_mb * 2**20) if (split == "train" and a.max_train_mb) else 0
    done = False
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as out:
        for src_id, group, text in iter_docs(a, split):
            n_src += 1
            for doc_id, paras in units_for_split(a, split, src_id, text):
                n_paras += len(paras)
                kept = []
                for p in paras:
                    reason = check_paragraph(p, a)
                    if reason is None:
                        h = para_hash(p)
                        if h in blocked or h in seen:
                            reason = "duplicate"
                        else:
                            seen.add(h)
                    if reason:
                        stats["dropped"][split][reason] += 1
                    else:
                        kept.append(p)
                doc = "\n\n".join(kept)
                if len(doc) < a.min_doc_chars:
                    stats["dropped"][split]["tiny_doc"] += 1
                    continue
                out.write(json.dumps({"id": doc_id, "group": group, "text": doc}, ensure_ascii=False) + "\n")
                n_docs += 1
                n_chars += len(doc)
                g = stats["groups"][split][group]
                g[0] += 1
                nb = len(doc.encode("utf-8"))
                g[1] += nb
                n_bytes += nb
                if limit and n_bytes >= limit:
                    done = True  # --max-train-mb reached
                    break
                if n_docs % 2000 == 0:
                    print(f"  [{split}] {n_docs:,} units, {n_chars / 1e6:,.1f}M chars kept ({time.time() - t0:.0f}s)")
            if done:
                break
    blocked |= seen
    if n_docs == 0:
        print(f"\n[{split}] DIAGNOSTICS: source files read={n_src:,}, paragraphs in this split={n_paras:,}, "
              f"dropped={dict(stats['dropped'][split])}")
        raise SystemExit(
            f"The {split} split is empty. If files read is ~0, check --src/--include. If paragraphs is 0, "
            f"the corpus is too small for --{split}-frac (raise it). If everything was dropped, the "
            f"filters are rejecting your text: read a file and loosen --min-stop / --min-alpha.")
    stats["splits"][split] = {"docs": n_docs, "chars": n_chars}
    print(f"[{split}] kept {n_docs:,} docs, {n_chars / 1e6:,.1f}M chars  ->  {path}")


# ------------------------------------------------------------------ tokenizer
def train_tokenizer(a):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    train_path = os.path.join(a.out, "clean_train.jsonl")
    size_mb = os.path.getsize(train_path) / 2**20
    p_keep = min(1.0, a.tok_train_mb / max(size_mb, 1e-9))
    rng = random.Random(a.seed)

    def texts():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if p_keep >= 1.0 or rng.random() < p_keep:
                    yield json.loads(line)["text"]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"Training BPE (vocab={a.vocab_size}) on ~{min(size_mb, a.tok_train_mb):,.0f} MB of TRAIN text ...")
    tok.train_from_iterator(texts(), trainer=trainer)
    path = os.path.join(a.out, f"tokenizer_{a.vocab_size}.json")
    tok.save(path)
    print(f"saved {path}")
    return tok


# ------------------------------------------------------------------ encoding
def encode_split(tok, a, split, shuffle):
    src = os.path.join(a.out, f"clean_{split}.jsonl")
    dst = os.path.join(a.out, f"{split}.bin")
    eos_id = tok.token_to_id(EOS)

    offsets, pos = [], 0
    with open(src, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    if shuffle:
        random.Random(a.seed).shuffle(offsets)

    n_tokens = n_bytes = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        for i in range(0, len(offsets), 512):
            texts = []
            for off in offsets[i:i + 512]:
                f.seek(off)
                texts.append(json.loads(f.readline())["text"])
            ids = []
            for t, e in zip(texts, tok.encode_batch(texts)):
                ids.extend(e.ids)
                ids.append(eos_id)
                n_bytes += len(t.encode("utf-8"))
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n_tokens += len(ids)
    print(f"[{split}] {len(offsets):,} docs -> {n_tokens / 1e6:,.1f}M tokens, "
          f"{n_bytes / 1e6:,.1f} MB text  ({n_bytes / max(n_tokens, 1):.2f} bytes/token)")
    return {"docs": len(offsets), "tokens": n_tokens, "bytes": n_bytes}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with downloaded text files")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--include", nargs="*", default=[], help="glob patterns (relative to --src) to keep")
    ap.add_argument("--exclude", nargs="*", default=[], help="glob patterns to drop")
    ap.add_argument("--text-field", default="text", help="field/column name for .jsonl/.parquet")
    ap.add_argument("--split-by", choices=["chunk", "doc", "group"], default="chunk",
                    help="chunk: hold out contiguous ~--chunk-chars blocks (best for few, large files); "
                         "doc: whole files; group: whole source folders (hardest, cleanest)")
    ap.add_argument("--chunk-chars", type=int, default=20000, help="block size for --split-by chunk")
    ap.add_argument("--group-depth", type=int, default=2, help="folder depth that defines a 'source'")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-para-chars", type=int, default=2000,
                    help="paragraphs longer than this are cut at sentence boundaries")
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--min-doc-chars", type=int, default=500)
    ap.add_argument("--min-stop", type=float, default=0.15, help="min fraction of English stopwords")
    ap.add_argument("--min-alpha", type=float, default=0.80)
    ap.add_argument("--filters", choices=["ocr", "minimal"], default="minimal",
                    help="ocr: strict, for scanned/OCR text (not used in the experiments). minimal: for clean born-digital "
                         "text such as PubMed - the OCR filters wrongly reject gene names, doses, chemicals")
    ap.add_argument("--max-nonascii", type=float, default=0.05)
    ap.add_argument("--max-word-len", type=int, default=24, help="drop paragraphs with longer 'words'")
    ap.add_argument("--max-digit", type=float, default=0.05, help="max fraction of digit characters")
    ap.add_argument("--max-mixed", type=float, default=0.02, help="max fraction of letter+digit tokens")
    ap.add_argument("--keep-linebreaks", action="store_true", help="do not unwrap hard-wrapped lines")
    ap.add_argument("--max-file-mb", type=int, default=1000)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tok-train-mb", type=float, default=500, help="max train text used to fit the tokenizer")
    ap.add_argument("--tokenizer", default=None, help="reuse an existing tokenizer.json instead of training")
    ap.add_argument("--skip-clean", action="store_true", help="reuse existing clean_*.jsonl files")
    ap.add_argument("--encode-splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"],
                    help="re-encode only these splits (e.g. 'val test' after removing duplicate documents)")
    ap.add_argument("--clean-only", action="store_true", help="stop after writing clean_*.jsonl (no tokenizer/encoding)")
    ap.add_argument("--max-train-mb", type=float, default=0,
                    help="stop collecting TRAIN text after this many MB (val/test still sample the whole source)")
    a = ap.parse_args()
    assert a.vocab_size < 65536, "uint16 storage needs vocab_size < 65536"
    os.makedirs(a.out, exist_ok=True)

    stats = {"dropped": defaultdict(Counter), "groups": defaultdict(lambda: defaultdict(lambda: [0, 0])),
             "splits": {}}

    # ---- 1. clean / filter / dedupe / split (test -> val -> train) --------
    if not a.skip_clean:
        blocked = set()
        for split in ["test", "val", "train"]:
            clean_split(a, split, blocked, stats)
        print("\nParagraphs dropped, by reason:")
        for split in ["train", "val", "test"]:
            print(f"  {split}: {dict(stats['dropped'][split])}")
        print("\nTop sources in train (docs, MB):")
        rows = sorted(stats["groups"]["train"].items(), key=lambda kv: -kv[1][1])[:25]
        for g, (d, b) in rows:
            print(f"  {g[:60]:<60}{d:>8,}{b / 1e6:>10,.1f}")
        with open(os.path.join(a.out, "cleaning_stats.json"), "w") as f:
            json.dump({"dropped": {s: dict(c) for s, c in stats["dropped"].items()},
                       "groups": {s: dict(g) for s, g in stats["groups"].items()},
                       "splits": stats["splits"], "args": vars(a)}, f, indent=1)

    if a.clean_only:
        print("--clean-only: stopping before tokenizer/encoding")
        return

    # ---- 2. tokenizer -----------------------------------------------------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer) if a.tokenizer else train_tokenizer(a)
    vocab = tok.get_vocab_size()
    assert vocab < 65536

    # ---- 3. encode --------------------------------------------------------
    meta_path = os.path.join(a.out, "meta.json")
    meta = {"vocab_size": vocab, "eos_id": tok.token_to_id(EOS), "dtype": "uint16",
            "train_order": "documents shuffled; any prefix of train.bin is a random sample of documents"}
    if set(a.encode_splits) != {"train", "val", "test"} and os.path.exists(meta_path):
        with open(meta_path) as f:  # partial re-encode: keep the other splits' entries
            old = json.load(f)
        if old.get("vocab_size") != vocab:
            raise SystemExit("existing meta.json was made with a different tokenizer; re-encode ALL splits")
        meta = {**old, **meta}
    for split in a.encode_splits:
        meta[split] = encode_split(tok, a, split, shuffle=(split == "train"))
    for split in ["val", "test"]:
        if split in a.encode_splits:
            meta[split]["tokens_per_byte"] = meta[split]["tokens"] / meta[split]["bytes"]
    meta["bits_per_byte_formula"] = "bpb = mean_loss_nats_per_token / ln(2) * tokens_per_byte"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {meta_path}")
    if "tokens_per_byte" in meta.get("val", {}):
        print(f"example: val loss 3.0 nats/token -> {3.0 / math.log(2) * meta['val']['tokens_per_byte']:.3f} bits/byte")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
