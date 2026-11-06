"""
train_utils.py - numpy-only helpers used by train.py (kept separate so they can be tested without a GPU).
"""
import json
import math
import os

import numpy as np


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def lr_at(step, max_steps, lr, warmup_steps, min_ratio=0.1):
    """Linear warmup, then cosine decay from `lr` down to `lr * min_ratio` at max_steps."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return lr * min_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def bpb(loss_nats_per_token, tokens_per_byte):
    """Convert mean loss in nats/token to bits per byte of the underlying text."""
    return loss_nats_per_token / math.log(2.0) * tokens_per_byte


class TokenSource:
    """Random-window sampler over a uint16 token file, optionally restricted to its first `limit` tokens.

    prepare_data.py shuffles training documents, so the first `limit` tokens of train.bin are a random
    sample of documents: `limit` is how we create a domain corpus with U unique tokens.
    """

    def __init__(self, path, block_size, limit=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = min(limit, len(self.data)) if limit else len(self.data)
        self.block = block_size
        if self.n <= block_size + 1:
            raise ValueError(f"{path}: only {self.n} usable tokens, need more than block_size+1")

    def sample(self, k, rng):
        """k random windows -> (x, y) int64 arrays of shape (k, block); y is x shifted by one token."""
        ix = rng.integers(0, self.n - self.block - 1, size=k)
        x = np.stack([self.data[i:i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64)
        return x, y

    def eval_batches(self, batch_size, max_blocks=0):
        """Deterministic, non-overlapping windows from the start of the (limited) data."""
        n_blocks = (self.n - 1) // self.block
        if max_blocks:
            n_blocks = min(n_blocks, max_blocks)
        for i in range(0, n_blocks, batch_size):
            js = range(i, min(i + batch_size, n_blocks))
            x = np.stack([self.data[j * self.block:(j + 1) * self.block] for j in js]).astype(np.int64)
            y = np.stack([self.data[j * self.block + 1:(j + 1) * self.block + 1] for j in js]).astype(np.int64)
            yield x, y


def mix_counts(batch_size, frac, rng):
    """How many of the batch's sequences come from the domain corpus.
    Stochastic rounding keeps the long-run share exactly `frac` even when frac*batch_size is fractional."""
    n = int(math.floor(frac * batch_size + rng.random()))
    return max(0, min(batch_size, n))


def get_batch(domain, general, frac, batch_size, rng):
    """Mixed batch: `frac` of sequences from `domain`, the rest from `general` (if there is one)."""
    n_dom = batch_size if general is None else mix_counts(batch_size, frac, rng)
    parts = []
    if n_dom > 0:
        parts.append(domain.sample(n_dom, rng))
    if batch_size - n_dom > 0:
        parts.append(general.sample(batch_size - n_dom, rng))
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return x, y, n_dom
