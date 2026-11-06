"""
shared_tokenizer.py - train ONE byte-level BPE tokenizer on an even mix (by bytes) of several corpora.

Using the same tokenizer for the domain and the general corpus removes tokenizer efficiency as a confound
when you mix them. Run it on the clean_train.jsonl files produced by prepare_data.py.

  python shared_tokenizer.py --inputs data/pubmed_v2/clean_train.jsonl data/fineweb_v1/clean_train.jsonl \
         --mb-each 400 --vocab-size 8192 --out tokenizers/mix8192.json

Then re-encode each corpus with it (the cleaning is NOT repeated):
  python prepare_data.py --src data/pubmed_raw   --out data/pubmed_v2   --skip-clean --tokenizer tokenizers/mix8192.json
  python prepare_data.py --src data/fineweb_raw  --out data/fineweb_v1  --skip-clean --tokenizer tokenizers/mix8192.json
"""
import argparse
import json
import os
import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


def sampled_texts(path, target_bytes, rng):
    """Yield documents from a jsonl file, sampled uniformly so that ~target_bytes of text come out."""
    p = min(1.0, target_bytes / max(os.path.getsize(path), 1))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if p >= 1.0 or rng.random() < p:
                yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="clean_train.jsonl files, one per corpus")
    ap.add_argument("--mb-each", type=float, default=400, help="MB of text taken from EACH corpus")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    assert a.vocab_size < 65536
    rng = random.Random(a.seed)
    target = a.mb_each * 2**20

    counts = {p: 0 for p in a.inputs}

    def stream():
        for path in a.inputs:  # BPE merge statistics do not depend on document order
            for t in sampled_texts(path, target, rng):
                counts[path] += len(t.encode("utf-8"))
                yield t

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(stream(), trainer=trainer)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tok.save(a.out)

    total = sum(counts.values())
    print(f"\nsaved {a.out}  (vocab {tok.get_vocab_size()}, eos id {tok.token_to_id(EOS)})")
    print("training text by corpus:")
    for p, b in counts.items():
        print(f"  {b / 2**20:8.1f} MB  ({100 * b / total:4.1f}%)  {p}")

    # efficiency check: bytes per token on a held-out-ish sample of each corpus (higher = more efficient)
    print("tokenizer efficiency (bytes per token, 300 sampled docs each):")
    for p in a.inputs:
        docs = list(sampled_texts(p, 3e6, random.Random(a.seed + 1)))[:300]
        n_tok = sum(len(e.ids) for e in tok.encode_batch(docs))
        n_byte = sum(len(d.encode("utf-8")) for d in docs)
        print(f"  {n_byte / max(n_tok, 1):5.2f}  {p}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # every path is relative to the repository root
    main()
