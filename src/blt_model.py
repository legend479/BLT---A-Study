# src/blt_model.py
"""
BLT PoC model implementation.

- PatchEmbedder: builds patch embeddings from per-n bucket IDs
- BLTModel: encoder-decoder transformer using patch embeddings as input memory
"""

from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbedder(nn.Module):
    """
    Converts patch bucket dicts {n: np.array(bucket_ids)} into patch embeddings.

    For each n in (1,2,3) it has an nn.Embedding(buckets, d_model).
    For each patch: embedding = sum_n ( sum_{id in bucket_ids} Emb_n[id] ).
    """

    def __init__(self, d_model: int = 64, buckets: int = 4096, ngrams=(1, 2, 3)):
        super().__init__()
        self.d_model = d_model
        self.buckets = buckets
        self.ngrams = ngrams
        self.emb_tables = nn.ModuleDict({
            str(n): nn.Embedding(buckets, d_model) for n in ngrams
        })

    def forward(self, patches_nested: List[List[Dict[int, np.ndarray]]], patch_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches_nested: list of length B; each element is a list of length P;
                            each [p] is dict {n: np.array(bucket_ids)}.
            patch_mask: BoolTensor [B, P], True if patch exists.

        Returns:
            patch_embs: Tensor [B, P, d_model] with embeddings for each patch.
        """
        B, P = patch_mask.shape
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        out = torch.zeros((B, P, self.d_model), device=device, dtype=dtype)

        for b in range(B):
            for p in range(P):
                if not patch_mask[b, p]:
                    continue
                patch = patches_nested[b][p]
                emb_sum = torch.zeros((self.d_model,), device=device, dtype=dtype)
                for n in self.ngrams:
                    ids = patch.get(n, None)
                    if ids is None or len(ids) == 0:
                        continue
                    ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
                    emb_sum = emb_sum + self.emb_tables[str(n)](ids_t).sum(dim=0)
                out[b, p] = emb_sum
        return out


class BLTModel(nn.Module):
    """
    BLT encoder-decoder transformer model.

    - Encoder: patch embeddings -> TransformerEncoder
    - Decoder: char tokens -> TransformerDecoder
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_patches: int = 512,
        buckets: int = 4096,
        ngrams=(1, 2, 3),
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_patches = max_patches

        # Patch embedder
        self.patch_embedder = PatchEmbedder(d_model=d_model, buckets=buckets, ngrams=ngrams)

        # Positional embeddings for patches
        self.patch_pos_emb = nn.Embedding(max_patches, d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Embedding(1024, d_model)  # max target length

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output head
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        patches_nested: List[List[Dict[int, np.ndarray]]],
        patch_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward training pass.

        Args:
            patches_nested: nested patch structure (list of list of dicts)
            patch_mask: BoolTensor [B, P]
            target_ids: LongTensor [B, T]
            target_mask: BoolTensor [B, T]

        Returns:
            logits: FloatTensor [B, T, vocab_size]
        """
        B, T = target_ids.shape
        device = next(self.parameters()).device

        # Encoder: patch embeddings
        patch_embs = self.patch_embedder(patches_nested, patch_mask)  # [B,P,d]
        P = patch_embs.size(1)
        pos_ids = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        patch_embs = patch_embs + self.patch_pos_emb(pos_ids)
        memory = self.encoder(patch_embs, src_key_padding_mask=~patch_mask)

        # Decoder: token embeddings
        pos_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tgt_embs = self.token_emb(target_ids) + self.token_pos_emb(pos_t)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        logits = self.decoder(
            tgt_embs,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~target_mask,
            memory_key_padding_mask=~patch_mask
        )
        logits = self.output_proj(logits)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        patches_nested: List[List[Dict[int, np.ndarray]]],
        patch_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> List[List[int]]:
        """
        Greedy autoregressive decoding.

        Returns: list of decoded ID lists (without SOS, stops at EOS if encountered).
        """
        B = len(patches_nested)
        device = next(self.parameters()).device

        # Encoder memory
        patch_embs = self.patch_embedder(patches_nested, patch_mask)
        P = patch_embs.size(1)
        pos_ids = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        patch_embs = patch_embs + self.patch_pos_emb(pos_ids)
        memory = self.encoder(patch_embs, src_key_padding_mask=~patch_mask)

        ys = torch.full((B, 1), sos_id, device=device, dtype=torch.long)
        finished = [False] * B
        outputs: List[List[int]] = [[] for _ in range(B)]

        for t in range(max_len):
            pos_t = torch.arange(ys.size(1), device=device).unsqueeze(0).expand(B, ys.size(1))
            tgt_embs = self.token_emb(ys) + self.token_pos_emb(pos_t)
            causal_mask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=device, dtype=torch.bool), diagonal=1)
            dec_out = self.decoder(
                tgt_embs,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=~patch_mask
            )
            logits = self.output_proj(dec_out[:, -1])  # [B,vocab]
            next_ids = torch.argmax(logits, dim=-1)

            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)

            for b in range(B):
                if not finished[b]:
                    nid = int(next_ids[b].item())
                    if nid == eos_id:
                        finished[b] = True
                    else:
                        outputs[b].append(nid)

            if all(finished):
                break

        return outputs

    @torch.no_grad()
    def beam_search_decode(
        self,
        patches_nested: List[List[Dict[int, np.ndarray]]],
        patch_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 128,
        beam_width: int = 4,
        length_penalty: float = 1.0
    ) -> List[List[int]]:
        """
        Beam search decoding for BLTModel.
        Returns list of decoded ID lists (without SOS, stops at EOS if encountered).
        """
        B = len(patches_nested)
        device = next(self.parameters()).device

        # Encoder memory
        patch_embs = self.patch_embedder(patches_nested, patch_mask)
        P = patch_embs.size(1)
        pos_ids = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        patch_embs = patch_embs + self.patch_pos_emb(pos_ids)
        memory = self.encoder(patch_embs, src_key_padding_mask=~patch_mask)

        # Initialize beams: each beam is (tokens, logprob)
        beams = [[([sos_id], 0.0)] for _ in range(B)]

        def length_norm(score: float, length: int) -> float:
            # GNMT length penalty (Wu et al. 2016)
            lp = ((5 + length) / 6) ** length_penalty
            return score / lp

        for t in range(max_len):
            new_beams = []
            for b in range(B):
                candidates = []
                for seq, score in beams[b]:
                    if seq[-1] == eos_id:
                        candidates.append((seq, score))  # already finished
                        continue

                    ys = torch.as_tensor(seq, device=device).unsqueeze(0)  # [1, len]
                    pos_t = torch.arange(ys.size(1), device=device).unsqueeze(0)
                    tgt_embs = self.token_emb(ys) + self.token_pos_emb(pos_t)
                    causal_mask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=device, dtype=torch.bool), diagonal=1)

                    dec_out = self.decoder(
                        tgt_embs,
                        memory[b:b+1],
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=~patch_mask[b:b+1]
                    )
                    logits = self.output_proj(dec_out[:, -1])  # [1,vocab]
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab]

                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                    for k in range(beam_width):
                        nid = int(topk_ids[k].item())
                        nscore = score + float(topk_log_probs[k].item())
                        candidates.append((seq + [nid], nscore))

                # Select top beam_width candidates after length-normalization
                candidates = sorted(candidates, key=lambda x: length_norm(x[1], len(x[0])), reverse=True)[:beam_width]
                new_beams.append(candidates)
            beams = new_beams

            if all(all(seq[-1] == eos_id for seq, _ in beam) for beam in beams):
                break

        outputs = []
        for b in range(B):
            best_seq, best_score = max(beams[b], key=lambda x: length_norm(x[1], len(x[0])))
            # strip SOS and cut at EOS
            if eos_id in best_seq:
                eos_index = best_seq.index(eos_id)
                best_seq = best_seq[1:eos_index]
            else:
                best_seq = best_seq[1:]
            outputs.append(best_seq)

        return outputs


# -----------------
# Simple sanity test
# -----------------
if __name__ == "__main__":
    vocab_size = 100
    model = BLTModel(vocab_size=vocab_size, d_model=32, nhead=2)

    # Fake data: batch of 2 samples, 3 patches each
    patches_nested = [
        [{1: np.array([1,2]), 2: np.array([3])}, {1: np.array([4])}, {}],
        [{1: np.array([5])}, {}, {}]
    ]
    patch_mask = torch.tensor([[1,1,0],[1,0,0]], dtype=torch.bool)
    target_ids = torch.randint(0, vocab_size, (2, 5))
    target_mask = torch.ones_like(target_ids, dtype=torch.bool)

    logits = model(patches_nested, patch_mask, target_ids, target_mask)
    print("logits shape:", logits.shape)

    greedy_decoded = model.greedy_decode(patches_nested, patch_mask, sos_id=1, eos_id=2, max_len=10)
    print("greedy decoded IDs:", greedy_decoded)

    beam_decoded = model.beam_search_decode(patches_nested, patch_mask, sos_id=1, eos_id=2, max_len=10, beam_width=3)
    print("beam search decoded IDs:", beam_decoded)
