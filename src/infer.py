# src/infer.py
"""Interactive inference for BLT and Baseline models."""

import argparse
import torch
from tokenizer import Tokenizer
from baseline_model import BaselineModel
from blt_model import BLTModel
from patcher import patch_bytes, patch_to_ngram_buckets
import numpy as np


def prepare_blt_input(text: str, device):
    text_bytes = text.encode("ascii", errors="replace")
    patches = patch_bytes(text_bytes, W=10, entropy_threshold=2.0, max_patch_len=15)
    patch_dicts = []
    for patch in patches:
        per_n = patch_to_ngram_buckets(patch, buckets=4096, seed=1337, ngrams=(1, 2, 3), unique_only=False, flatten=False)
        per_n_arr = {int(n): np.array(per_n[n], dtype=np.int32) for n in per_n}
        patch_dicts.append(per_n_arr)
    patches_nested = [patch_dicts]
    patch_mask = torch.ones((1, len(patch_dicts)), dtype=torch.bool, device=device)
    return patches_nested, patch_mask


def prepare_char_input(text: str, tokenizer: Tokenizer, device):
    input_ids = tokenizer.encode(text, add_sos=False, add_eos=False, strict=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    input_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, input_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["blt", "char"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})
    
    tokenizer_path = args.tokenizer or f"data/processed/tokenizer_{args.mode}.json"
    tokenizer = Tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    
    if args.mode == "char":
        model = BaselineModel(vocab_size=vocab_size, d_model=model_args.get("d_model", 128),
                             nhead=model_args.get("nhead", 4), num_encoder_layers=model_args.get("num_encoder_layers", 2),
                             num_decoder_layers=model_args.get("num_decoder_layers", 2),
                             dim_feedforward=model_args.get("dim_feedforward", 256), dropout=0.0)
    else:
        model = BLTModel(vocab_size=vocab_size, d_model=model_args.get("d_model", 128),
                        nhead=model_args.get("nhead", 4), num_encoder_layers=model_args.get("num_encoder_layers", 2),
                        num_decoder_layers=model_args.get("num_decoder_layers", 2),
                        dim_feedforward=model_args.get("dim_feedforward", 256), dropout=0.0)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Beam width: {args.beam_width}")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Input: ").strip()
        if text.lower() in ["quit", "exit"]:
            break
        if not text:
            continue
        
        try:
            if args.mode == "char":
                input_ids, input_mask = prepare_char_input(text, tokenizer, device)
                if args.beam_width > 1:
                    decoded_ids = model.beam_search_decode(input_ids, input_mask, tokenizer.sos_id,
                                                          tokenizer.eos_id, args.max_len, args.beam_width, args.length_penalty)
                else:
                    decoded_ids = model.greedy_decode(input_ids, input_mask, tokenizer.sos_id, tokenizer.eos_id, args.max_len)
            else:
                patches_nested, patch_mask = prepare_blt_input(text, device)
                if args.beam_width > 1:
                    decoded_ids = model.beam_search_decode(patches_nested, patch_mask, tokenizer.sos_id,
                                                          tokenizer.eos_id, args.max_len, args.beam_width, args.length_penalty)
                else:
                    decoded_ids = model.greedy_decode(patches_nested, patch_mask, tokenizer.sos_id, tokenizer.eos_id, args.max_len)
            
            output_text = tokenizer.decode(decoded_ids[0], strip_specials=True)
            print(f"Output: {output_text}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
