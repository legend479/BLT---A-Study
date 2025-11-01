# src/dataset.py
"""
Preprocessing utilities and Dataset classes for BLT PoC.

Main function:
    preprocess_csv_to_pt(csv_path, out_pt, mode='blt', tokenizer=None, W=10, entropy_threshold=2.0,
                         max_patch_len=15, buckets=4096, seed=1337, ngrams=(1,2,3),
                         force_rebuild=False, manifest_out=None)

- Produces a processed torch file (list of sample dicts) suitable for training/eval.
- Produces manifest.json containing preprocessing hyperparameters and tokenizer path.

Dataset classes:
 - BLTProcessedDataset(processed_pt_path)
 - CharProcessedDataset(processed_pt_path)

Collate:
 - collate_fn(batch, mode='blt')  # returns dict consumed by training loop / models

Helper:
 - make_dataloaders(train_pt, val_pt=None, mode='blt', batch_size=64, num_workers=4)
"""

from __future__ import annotations
import os
import sys
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from patcher import patch_bytes, patch_to_ngram_buckets
from tokenizer import Tokenizer

# -----------------------------
# Preprocessing: main function
# -----------------------------
def preprocess_csv_to_pt(
    csv_path: str,
    out_pt: str,
    mode: str = "blt",   # "blt" or "char"
    tokenizer: Optional[Tokenizer] = None,
    W: int = 10,
    entropy_threshold: float = 2.0,
    max_patch_len: int = 15,
    buckets: int = 4096,
    seed: int = 1337,
    ngrams: Tuple[int, ...] = (1, 2, 3),
    force_rebuild: bool = False,
    manifest_out: Optional[str] = None,
    tokenizer_out: Optional[str] = None,
    strict_ascii: bool = True,
    sample_limit: Optional[int] = None
) -> Tuple[str, str]:
    """
    Read CSV with columns ['input','target'] and produce a processed .pt file
    and a manifest.json describing preprocessing parameters.

    Args:
        csv_path: path to input CSV (expects columns input,target)
        out_pt: path to write processed torch file (e.g., data/processed/train_blt.pt)
        mode: 'blt' or 'char' -> controls whether to precompute patches or just chars
        tokenizer: Tokenizer instance (if None, a new Tokenizer() will be created and saved)
        W, entropy_threshold, max_patch_len: patcher params
        buckets, seed, ngrams: hashing params
        force_rebuild: if True overwrite existing out_pt
        manifest_out: path to manifest.json (if None, will be sibling to out_pt)
        tokenizer_out: path to save tokenizer (if None, uses "<out_dir>/tokenizer.json")
        strict_ascii: if True raise on non-printable characters, else map to <UNK>
        sample_limit: if set, only preprocess first sample_limit rows (handy for quick tests)

    Returns:
        (out_pt, manifest_out) paths written
    """
    if mode not in ("blt", "char"):
        raise ValueError("mode must be 'blt' or 'char'")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = os.path.dirname(out_pt) or "."
    os.makedirs(out_dir, exist_ok=True)

    if manifest_out is None:
        manifest_out = os.path.join(out_dir, "manifest.json")
    if tokenizer_out is None:
        tokenizer_out = os.path.join(out_dir, "tokenizer.json")

    if os.path.exists(out_pt) and os.path.exists(manifest_out) and not force_rebuild:
        print(f"[preprocess] Found existing processed file at {out_pt} and manifest; skipping (use force_rebuild=True to overwrite).")
        return out_pt, manifest_out

    # load CSV
    df = pd.read_csv(csv_path)
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must contain columns 'input' and 'target'")

    if sample_limit is not None:
        df = df.iloc[:sample_limit].copy()

    # tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer()
    # persist tokenizer
    tokenizer.save(tokenizer_out)

    processed_samples: List[Dict[str, Any]] = []
    stats = {
        "num_rows": int(len(df)),
        "num_processed": 0,
        "num_skipped": 0,
        "patch_counts": [],
        "char_counts": []
    }

    # iterate rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing rows"):
        inp = row["input"]
        tgt = row["target"]

        # basic validation
        if not isinstance(inp, str) or not isinstance(tgt, str):
            stats["num_skipped"] += 1
            continue

        # ascii / strict handling
        try:
            inp_bytes = inp.encode("ascii")
        except UnicodeEncodeError:
            if strict_ascii:
                print(f"[preprocess] Skipping row due to non-ascii characters: {inp!r}")
                stats["num_skipped"] += 1
                continue
            else:
                # Replace non-ASCII with '?' and continue
                inp_bytes = inp.encode("ascii", errors="replace")
        except Exception as e:
            print(f"[preprocess] Unexpected encoding error for input {inp!r}: {e}")
            stats["num_skipped"] += 1
            continue

        # tokenize target with tokenizer (add SOS/EOS)
        try:
            target_ids = tokenizer.encode(tgt, add_sos=True, add_eos=True, strict=strict_ascii)
        except ValueError as e:
            # tokenization failed
            if strict_ascii:
                print(f"[preprocess] Skipping row due to tokenization error: {e}")
                stats["num_skipped"] += 1
                continue
            else:
                # fallback: map unknowns to UNK and continue
                target_ids = tokenizer.encode(tgt, add_sos=True, add_eos=True, strict=False)

        sample_record: Dict[str, Any] = {
            "input": inp,
            "target": tgt,
            "target_ids": np.array(target_ids, dtype=np.int16),
            "num_chars": int(len(inp)),
        }

        if mode == "char":
            # For baseline char model we also precompute input_ids if desired
            try:
                input_ids = tokenizer.encode(inp, add_sos=False, add_eos=False, strict=strict_ascii)
            except ValueError:
                if strict_ascii:
                    stats["num_skipped"] += 1
                    continue
                else:
                    input_ids = tokenizer.encode(inp, add_sos=False, add_eos=False, strict=False)
            sample_record["input_ids"] = np.array(input_ids, dtype=np.int16)
            sample_record["num_tokens"] = int(len(input_ids))
            stats["char_counts"].append(int(len(inp)))

        else:  # mode == "blt"
            # compute patches and per-n bucket lists for each patch
            patches = patch_bytes(inp_bytes, W=W, entropy_threshold=entropy_threshold, max_patch_len=max_patch_len)
            # patches is list of bytes. For each patch produce dict n-> list of bucket ids
            patch_dicts: List[Dict[int, np.ndarray]] = []
            for patch in patches:
                per_n = patch_to_ngram_buckets(patch, buckets=buckets, seed=seed, ngrams=ngrams, unique_only=False, flatten=False)
                # convert each list to numpy array (int32)
                per_n_arr = {int(n): np.array(per_n[n], dtype=np.int32) for n in per_n}
                patch_dicts.append(per_n_arr)

            sample_record["patches"] = patch_dicts  # list of dicts {1: np.array, 2: np.array, 3: np.array}
            sample_record["num_patches"] = int(len(patch_dicts))
            stats["patch_counts"].append(int(len(patch_dicts)))
            stats["char_counts"].append(int(len(inp)))

        processed_samples.append(sample_record)
        stats["num_processed"] += 1

    # save processed samples as a torch file
    # Torch can serialize python lists/dicts/numpy arrays conveniently
    torch.save(processed_samples, out_pt)

    # manifest
    manifest = {
        "mode": mode,
        "processed_file": out_pt,
        "tokenizer_path": tokenizer_out,
        "W": int(W),
        "entropy_threshold": float(entropy_threshold),
        "max_patch_len": int(max_patch_len),
        "buckets": int(buckets),
        "seed": int(seed),
        "ngrams": list(int(n) for n in ngrams),
        "num_rows": int(len(df)),
        "num_processed": int(stats["num_processed"]),
        "strict_ascii": bool(strict_ascii)
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # also write stats to a small json next to out_pt
    stats_out = os.path.splitext(out_pt)[0] + "_stats.json"
    try:
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass

    print(f"[preprocess] Wrote processed dataset to {out_pt}")
    print(f"[preprocess] Wrote manifest to {manifest_out}")
    print(f"[preprocess] Stats written to {stats_out}")

    return out_pt, manifest_out


# -----------------------------------
# Dataset classes and collate helper
# -----------------------------------
class BLTProcessedDataset(Dataset):
    """Loads processed .pt produced by preprocess_csv_to_pt in mode='blt'"""
    def __init__(self, processed_pt_path: str):
        if not os.path.exists(processed_pt_path):
            raise FileNotFoundError(f"Processed file not found: {processed_pt_path}")
        try:
            self.samples: List[Dict[str, Any]] = torch.load(processed_pt_path, weights_only=False)
        except (RuntimeError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Error loading processed dataset: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class CharProcessedDataset(Dataset):
    """Loads processed .pt produced by preprocess_csv_to_pt in mode='char'"""
    def __init__(self, processed_pt_path: str):
        if not os.path.exists(processed_pt_path):
            raise FileNotFoundError(f"Processed file not found: {processed_pt_path}")
        try:
            self.samples: List[Dict[str, Any]] = torch.load(processed_pt_path, weights_only=False)
        except (RuntimeError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"Error loading processed dataset: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, Any]], mode: str = "blt", pad_id: int = 0) -> Dict[str, Any]:
    """
    Collate function to be passed to DataLoader.

    For BLT mode:
      - batch: list of sample dicts where each sample has keys:
          'input', 'target', 'target_ids' (np.array), 'patches' (list of dicts {n: np.array})
      - returns a dict with:
          'patches': nested list [B][P] where each element is dict {n: np.array(bucket_ids)}
          'patch_mask': torch.BoolTensor[B, P] (True if patch exists)
          'target_ids': torch.LongTensor[B, T] (padded with pad_id)
          'target_mask': torch.BoolTensor[B, T] (True for non-pad tokens)
          'raw_inputs': list of input strings
    For char mode:
      - expects sample['input_ids'] and 'target_ids'
      - returns 'input_ids', 'input_mask', 'target_ids', 'target_mask', 'raw_inputs'
    """
    if mode not in ("blt", "char"):
        raise ValueError("mode must be 'blt' or 'char'")

    batch_size = len(batch)
    raw_inputs = [s.get("input", "") for s in batch]

    # target padding
    target_lengths = [len(s["target_ids"]) for s in batch]
    max_tlen = max(target_lengths) if target_lengths else 0
    target_padded = torch.full((batch_size, max_tlen), pad_id, dtype=torch.long)
    target_mask = torch.zeros((batch_size, max_tlen), dtype=torch.bool)
    for i, s in enumerate(batch):
        arr = s["target_ids"]
        target_padded[i, : len(arr)] = torch.from_numpy(np.array(arr, dtype=np.int64))
        target_mask[i, : len(arr)] = True

    if mode == "char":
        # input_ids expected present
        input_lengths = [len(s.get("input_ids", [])) for s in batch]
        max_il = max(input_lengths) if input_lengths else 0
        input_padded = torch.full((batch_size, max_il), pad_id, dtype=torch.long)
        input_mask = torch.zeros((batch_size, max_il), dtype=torch.bool)
        for i, s in enumerate(batch):
            inp = s.get("input_ids", np.array([], dtype=np.int16))
            if len(inp) > 0:
                input_padded[i, : len(inp)] = torch.from_numpy(np.array(inp, dtype=np.int64))
                input_mask[i, : len(inp)] = True

        return {
            "input_ids": input_padded,
            "input_mask": input_mask,
            "target_ids": target_padded,
            "target_mask": target_mask,
            "raw_inputs": raw_inputs
        }

    # mode == 'blt'
    # compute max number of patches in this batch
    num_patches_list = [s.get("num_patches", 0) for s in batch]
    max_patches = max(num_patches_list) if num_patches_list else 0

    # build nested list [B][P] where element is patch dict {n: np.array}
    patches_nested: List[List[Dict[int, np.ndarray]]] = []
    patch_mask = torch.zeros((batch_size, max_patches), dtype=torch.bool)

    for i, s in enumerate(batch):
        patches = s.get("patches", [])  # list of dicts
        row_patches: List[Dict[int, np.ndarray]] = []
        for p_idx in range(max_patches):
            if p_idx < len(patches):
                # Optimized per-patch tensor conversion
                raw_patch = patches[p_idx]  # dict {n: np.array}
                patch_t: Dict[int, np.ndarray] = {}  # Keep as numpy for now, convert in model
                for n_key, arr in raw_patch.items():
                    # Efficient numpy array handling
                    if isinstance(arr, np.ndarray):
                        if arr.dtype != np.int32:
                            arr = arr.astype(np.int32)
                        patch_t[int(n_key)] = arr
                    else:
                        patch_t[int(n_key)] = np.array(arr, dtype=np.int32)
                row_patches.append(patch_t)
                patch_mask[i, p_idx] = True
            else:
                # placeholder for missing patch (empty dict)
                row_patches.append({})  # model's PatchEmbedder should handle empty dict as empty patch
        patches_nested.append(row_patches)

    return {
        "patches": patches_nested,   # nested python list, leave numpy arrays inside
        "patch_mask": patch_mask,
        "target_ids": target_padded,
        "target_mask": target_mask,
        "raw_inputs": raw_inputs
    }


def make_dataloaders(
    train_pt: str,
    val_pt: Optional[str] = None,
    mode: str = "blt",
    batch_size: int = 64,
    num_workers: int = 4,
    pad_id: int = 0,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Construct DataLoaders for train and optional validation.

    Returns (train_loader, val_loader_or_None)
    """
    if mode == "blt":
        train_ds = BLTProcessedDataset(train_pt)
    else:
        train_ds = CharProcessedDataset(train_pt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=lambda b: collate_fn(b, mode=mode, pad_id=pad_id),
                              pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=2)

    val_loader = None
    if val_pt:
        if mode == "blt":
            val_ds = BLTProcessedDataset(val_pt)
        else:
            val_ds = CharProcessedDataset(val_pt)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=lambda b: collate_fn(b, mode=mode, pad_id=pad_id),
                                pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=2)

    return train_loader, val_loader


# -----------------
# small demo / test
# -----------------
if __name__ == "__main__":
    # create a tiny CSV for demonstration
    demo_csv = "tmp_demo.csv"
    df = pd.DataFrame({
        "input": ["hello", "aaaaabbbbcccc", "The quick brown fox", ""],
        "target": ["olleh", "cccbbbaaaaa", "xof nworb kciuq ehT", ""]
    })
    df.to_csv(demo_csv, index=False)

    out_pt = "tmp_processed_demo.pt"
    manifest_path = "tmp_manifest.json"
    out_pt, manifest = preprocess_csv_to_pt(
        demo_csv,
        out_pt,
        mode="blt",
        W=10,
        entropy_threshold=2.0,
        max_patch_len=15,
        buckets=4096,
        seed=1337,
        ngrams=(1, 2, 3),
        force_rebuild=True,
        manifest_out=manifest_path,
        tokenizer_out="tmp_tokenizer.json",
        strict_ascii=False
    )

    print("Wrote:", out_pt)
    print("Manifest:", manifest)
    samples = torch.load(out_pt, weights_only=False)
    print("Sample count:", len(samples))
    for i, s in enumerate(samples):
        print(f"Sample {i}: input={s['input']!r} num_patches={s.get('num_patches', 'N/A')} num_chars={s.get('num_chars', 'N/A')}")
        if "patches" in s:
            for p_i, p in enumerate(s["patches"]):
                print(f"  patch {p_i} per-n counts: {{n: len}} ->", {n: len(p.get(n, [])) for n in p.keys()})
    # cleanup demo files
    try:
        os.remove(demo_csv)
        os.remove(out_pt)
        os.remove(manifest_path)
        os.remove("tmp_tokenizer.json")
    except Exception:
        pass
