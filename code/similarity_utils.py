"""
similarity_utils.py - numpy-only helpers for the "does similarity of the general data matter?" extension.

Everything works directly on the uint16 token files written by prepare_data.py (train.bin etc.), whose layout is
    doc0 tokens, EOS, doc1 tokens, EOS, ...
so no text is re-read and nothing is re-tokenised. Designed for ~1 GB token files: all passes are chunked and
vectorised, and only the first `window` tokens of each document are ever sent to a model.
"""
import numpy as np

EOS = 0
_MULT = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
                  0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB, 0xBF58476D1CE4E5B9,
                  0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3, 0x589965CC75374CC3,
                  0x1D8E4E27C47D124F, 0xEB44ACCAB455D165, 0x2545F4914F6CDD1D, 0x9FB21C651E98DF25],
                 dtype=np.uint64)


# ----------------------------------------------------------------- documents inside a token file
def eos_positions(tokens, eos=EOS, chunk=1 << 26):
    out = [np.flatnonzero(np.asarray(tokens[s:s + chunk]) == eos) + s for s in range(0, len(tokens), chunk)]
    return np.concatenate(out).astype(np.int64) if out else np.empty(0, np.int64)


def doc_table(tokens, eos=EOS):
    """(starts, lengths) of every complete document; a document ends at its EOS token (EOS not counted in length)."""
    e = eos_positions(tokens, eos)
    starts = np.concatenate([[0], e[:-1] + 1]).astype(np.int64)
    return starts, (e - starts).astype(np.int64)


def gather_windows(tokens, starts, length):
    """(N, length+1) int64: for each doc the window that begins at the EOS just BEFORE it, i.e. [EOS, doc tokens, ...].
    Needs starts >= 1 (a preceding EOS must exist). Rows are contiguous reads, so this is fast on a memmap."""
    idx = (starts - 1)[:, None] + np.arange(length + 1)[None, :]
    np.minimum(idx, len(tokens) - 1, out=idx)
    return np.asarray(tokens[idx]).astype(np.int64)


def score_docs(tokens, starts, lengths, nll_fns, window=256, batch=192, min_len=32, log=None):
    """Mean next-token NLL (nats) of the first `window` tokens of each eligible document under each model.

    nll_fns: callables f(x, y) with x, y int64 arrays (B, window); y is x shifted by one token; returns (B, window)
             NLL of each target. Docs shorter than `min_len` (and the first doc, which has no preceding EOS) are skipped.
    Returns (doc_indices, nll) with nll of shape (len(doc_indices), len(nll_fns)).
    """
    elig = np.flatnonzero((lengths >= min_len) & (starts >= 1))
    out = np.zeros((len(elig), len(nll_fns)), dtype=np.float64)
    pos = np.arange(window)[None, :]
    for i in range(0, len(elig), batch):
        ids = elig[i:i + batch]
        w = gather_windows(tokens, starts[ids], window)
        x, y = w[:, :-1], w[:, 1:]
        valid = pos < np.minimum(window, lengths[ids])[:, None]   # targets that are real tokens of this doc
        denom = valid.sum(1)
        for m, fn in enumerate(nll_fns):
            out[i:i + len(ids), m] = (fn(x, y) * valid).sum(1) / denom
        if log and (i // batch) % 100 == 0:
            log(i + len(ids), len(elig))
    return elig, out


# ----------------------------------------------------------------- decontamination
def scan_shingles(tokens, n=12, mod=16, chunk=1 << 24):
    """Yield (positions, hashes) of the sampled n-token shingles (1 in `mod`) of a token file, chunk by chunk."""
    total = len(tokens)
    for a in range(0, max(total - n + 1, 0), chunk):
        t = np.asarray(tokens[a:a + chunk + n - 1], dtype=np.uint64)
        m = len(t) - n + 1
        if m <= 0:
            continue
        with np.errstate(over="ignore"):
            s = np.zeros(m, dtype=np.uint64)
            for k in range(n):
                s += t[k:k + m] * _MULT[k % len(_MULT)]
            s ^= s >> np.uint64(29)
            s *= np.uint64(0xBF58476D1CE4E5B9)
            s ^= s >> np.uint64(32)
        keep = np.flatnonzero((s % np.uint64(mod)) == 0)
        yield keep + a, s[keep]


def reference_hashes(token_arrays, n=12, mod=16):
    """Sorted unique sampled shingle hashes of several token arrays (the evaluation sets)."""
    parts = [h for tk in token_arrays for _, h in scan_shingles(tk, n, mod)]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)


def contaminated_hits(tokens, starts, ref_sorted, n=12, mod=16):
    """Number of sampled shingles of each document that also occur in `ref_sorted`."""
    counts = np.zeros(len(starts), dtype=np.int64)
    if len(ref_sorted) == 0:
        return counts
    for pos, h in scan_shingles(tokens, n, mod):
        j = np.minimum(np.searchsorted(ref_sorted, h), len(ref_sorted) - 1)
        hit = pos[ref_sorted[j] == h]
        if len(hit):
            counts += np.bincount(np.searchsorted(starts, hit, side="right") - 1, minlength=len(starts))
    return counts


# ----------------------------------------------------------------- pools
def equal_mass_pools(order, lengths, k):
    """Split documents (already sorted, most similar first) into k pools holding ~equal numbers of TOKENS."""
    mass = np.cumsum(lengths[order])
    cuts = np.searchsorted(mass, mass[-1] * np.arange(1, k) / k)
    return np.split(order, cuts)


def random_pool(candidates, lengths, token_mass, rng):
    perm = rng.permutation(candidates)
    return perm[: int(np.searchsorted(np.cumsum(lengths[perm]), token_mass)) + 1]


def write_pool(path, bins, bin_idx, starts, lengths, doc_ids, rng, chunk_docs=20000):
    """Write the chosen documents (shuffled, each with its trailing EOS) as one uint16 token file. Returns #tokens."""
    order = rng.permutation(doc_ids)
    n = 0
    with open(path, "wb") as f:
        for i in range(0, len(order), chunk_docs):
            part = np.concatenate([bins[bin_idx[j]][starts[j]: starts[j] + lengths[j] + 1] for j in order[i:i + chunk_docs]])
            part.astype(np.uint16, copy=False).tofile(f)
            n += len(part)
    return n
