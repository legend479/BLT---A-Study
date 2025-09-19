# src/patcher.py
"""
Entropy-based byte patcher and deterministic n-gram hashing utilities.

This version follows the assignment spec:
 - sliding window W (default 10)
 - Shannon entropy threshold (default 2.0)
 - max_patch_len default 15
 - n-grams up to 3 (1,2,3) and 4096 hash buckets per n-gram size
 - patch_to_ngram_buckets returns per-n lists so you can use separate
   embedding tables for each n-gram size.

Run as a script for a small sanity demo.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import List, Tuple, Dict
import hashlib
import sys


def shannon_entropy_window(b: bytes) -> float:
    """
    Compute Shannon entropy (in bits) for the given bytes window.

    Returns 0.0 for empty input.
    """
    if not b:
        return 0.0
    counts = Counter(b)
    total = len(b)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def patch_bytes(
    data: bytes,
    W: int = 10,
    entropy_threshold: float = 2.0,
    max_patch_len: int = 15
) -> List[bytes]:
    """
    Entropy-based byte patching with sliding window of size W.

    Cuts patch when:
      - Shannon entropy(window) > entropy_threshold, OR
      - current patch length >= max_patch_len

    Returns list of patches (bytes). Concatenation of patches equals input.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")

    n = len(data)
    if n == 0:
        return []

    patches: List[bytes] = []
    start = 0
    i = 0
    while i < n:
        window = data[i: i + W]
        ent = shannon_entropy_window(window)
        if ent > entropy_threshold or (i - start) >= max_patch_len:
            if i == start:
                i = start + 1
            patches.append(bytes(data[start:i]))
            start = i
        else:
            i += 1

    if start < n:
        patches.append(bytes(data[start:n]))

    # Sanity re-concatenation check (assertions can be disabled with -O)
    assert b"".join(patches) == data, "Patches do not re-concatenate to original data"

    return patches


def ngram_hash(ngram: bytes, buckets: int = 4096, seed: int = 1337) -> int:
    """
    Deterministic hash of ngram bytes into [0, buckets).
    Uses sha256 with seed mixed in.
    """
    if not isinstance(ngram, (bytes, bytearray)):
        raise TypeError("ngram must be bytes")
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    seed_bytes = int(seed & 0xFFFFFFFF).to_bytes(4, "little")
    h = hashlib.sha256()
    h.update(seed_bytes)
    h.update(ngram)
    digest = h.digest()
    num = int.from_bytes(digest[:8], "little")
    return int(num % buckets)


def patch_to_ngram_buckets(
    patch: bytes,
    buckets: int = 4096,
    seed: int = 1337,
    ngrams: Tuple[int, ...] = (1, 2, 3),
    unique_only: bool = False,
    flatten: bool = False
) -> Dict[int, List[int]] or List[int]:
    """
    Convert a patch into n-gram bucket ids for each n in `ngrams`.

    Returns:
      - if flatten=False (default): a dict {n: [bucket_ids_for_n], ...}
      - if flatten=True: a single list concatenating buckets in ascending n order
        (same order as dict would produce).

    Use unique_only=True to deduplicate bucket ids per-patch (keeps first-seen).
    """
    if not isinstance(patch, (bytes, bytearray)):
        raise TypeError("patch must be bytes or bytearray")

    result: Dict[int, List[int]] = {}
    mv = memoryview(patch)
    for n in ngrams:
        if n <= 0:
            result[n] = []
            continue
        if len(patch) < n:
            result[n] = []
            continue
        seen = set()
        out_n: List[int] = []
        for i in range(0, len(patch) - n + 1):
            ng = mv[i:i + n].tobytes()
            bkt = ngram_hash(ng, buckets=buckets, seed=seed)
            if unique_only:
                if bkt in seen:
                    continue
                seen.add(bkt)
            out_n.append(bkt)
        result[n] = out_n

    if flatten:
        flat: List[int] = []
        for n in sorted(result.keys()):
            flat.extend(result[n])
        return flat

    return result


# ---------------------------
# Small sanity/demo when run
# ---------------------------
def _demo_examples():
    examples = [
        b"aaaaaaaabbbbbbbb",               # low entropy: large patches
        b"Hello, world!",                 # mixed
        b"The quick brown fox jumps over the lazy dog.",    # natural text
        b"0123456789abcdef" * 3,         # higher entropy
        b"",                             # empty
    ]
    return examples


def _print_patch_info(data: bytes, W: int, threshold: float, max_patch_len: int, buckets: int, seed: int):
    print("INPUT:", repr(data))
    patches = patch_bytes(data, W=W, entropy_threshold=threshold, max_patch_len=max_patch_len)
    print(f"  {len(patches)} patches (W={W}, thr={threshold}, max_patch_len={max_patch_len})")
    for idx, p in enumerate(patches):
        ent = shannon_entropy_window(p)
        ngram_buckets = patch_to_ngram_buckets(p, buckets=buckets, seed=seed, ngrams=(1,2,3))
        counts = {n: len(ngram_buckets[n]) for n in sorted(ngram_buckets.keys())}
        sample = {n: ngram_buckets[n][:8] for n in sorted(ngram_buckets.keys())}
        print(f"   - P{idx}: len={len(p)} ent={ent:.4f} ngram_counts={counts} buckets_sample={sample}")
    recon = b"".join(patches)
    print("  Reconstructed equals original:", recon == data)
    print("-" * 60)


if __name__ == "__main__":
    print("Running src/patcher.py sanity demo\n")

    W = 10
    entropy_threshold = 2.0
    max_patch_len = 15
    buckets = 4096
    seed = 1337

    examples = _demo_examples()
    for ex in examples:
        _print_patch_info(ex, W, entropy_threshold, max_patch_len, buckets, seed)

    # Sanity checks
    for ex in examples:
        patches = patch_bytes(ex, W=W, entropy_threshold=entropy_threshold, max_patch_len=max_patch_len)
        assert b"".join(patches) == ex, "Concatenation invariant failed"

    # hashing determinism
    sample_ng = b"ab"
    b1 = ngram_hash(sample_ng, buckets=buckets, seed=seed)
    b2 = ngram_hash(sample_ng, buckets=buckets, seed=seed)
    assert b1 == b2, "ngram_hash not deterministic for same seed"

    # check per-n counts
    p = b"ABCDE"
    per_n = patch_to_ngram_buckets(p, buckets=buckets, seed=seed, ngrams=(1,2))
    # len for 1-grams = 5, 2-grams =4 => total 9
    assert len(per_n[1]) == 5 and len(per_n[2]) == 4, f"unexpected ngram counts per n: { {n: len(per_n[n]) for n in per_n} }"

    print("\nAll sanity checks passed.")
    sys.exit(0)
