# src/baseline_model.py
"""
Baseline character-level encoder-decoder model (no patching).

- Encoder: character embeddings -> TransformerEncoder
- Decoder: autoregressive TransformerDecoder
- Supports forward() for training and greedy_decode() for inference
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Projection to vocab
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src_ids: [B, S] input character IDs
            src_mask: [B, S] bool mask (True = valid)
            tgt_ids: [B, T] target character IDs
            tgt_mask: [B, T] bool mask (True = valid)

        Returns:
            logits: [B, T, vocab_size]
        """
        B, S = src_ids.shape
        B, T = tgt_ids.shape
        device = src_ids.device

        # Source embeddings + pos
        pos_s = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        src_embs = self.token_emb(src_ids) + self.pos_emb(pos_s)
        memory = self.encoder(src_embs, src_key_padding_mask=~src_mask)

        # Target embeddings + pos
        pos_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tgt_embs = self.token_emb(tgt_ids) + self.pos_emb(pos_t)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        out = self.decoder(
            tgt_embs,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~tgt_mask,
            memory_key_padding_mask=~src_mask,
        )
        logits = self.output_proj(out)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 128
    ) -> List[List[int]]:
        """
        Greedy autoregressive decoding.

        Args:
            src_ids: [B, S] source IDs
            src_mask: [B, S] source mask
            sos_id: start-of-sequence token ID
            eos_id: end-of-sequence token ID
            max_len: maximum length to decode

        Returns:
            List of token ID sequences (w/o SOS, cut at EOS if found)
        """
        B, S = src_ids.shape
        device = src_ids.device

        # Encode source
        pos_s = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        src_embs = self.token_emb(src_ids) + self.pos_emb(pos_s)
        memory = self.encoder(src_embs, src_key_padding_mask=~src_mask)

        # Init targets with SOS
        ys = torch.full((B, 1), sos_id, device=device, dtype=torch.long)
        outputs = [[] for _ in range(B)]
        finished = [False] * B

        for t in range(max_len):
            pos_t = torch.arange(ys.size(1), device=device).unsqueeze(0).expand(B, ys.size(1))
            tgt_embs = self.token_emb(ys) + self.pos_emb(pos_t)

            causal_mask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=device, dtype=torch.bool), diagonal=1)

            dec_out = self.decoder(
                tgt_embs,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=~src_mask
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
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 128,
        beam_width: int = 5,
        length_penalty: float = 1.0
    ) -> List[List[int]]:
        """
        Beam search decoding.

        Args:
            src_ids: [B, S] source IDs
            src_mask: [B, S] source mask
            sos_id: start-of-sequence token ID
            eos_id: end-of-sequence token ID
            max_len: maximum length to decode
            beam_width: beam width
            length_penalty: alpha for GNMT length penalty (1.0 = mild, 0.0 = off)

        Returns:
            List of token ID sequences (w/o SOS, cut at EOS if found)
        """
        B, S = src_ids.shape
        device = src_ids.device

        # Encode source
        pos_s = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        src_embs = self.token_emb(src_ids) + self.pos_emb(pos_s)
        memory = self.encoder(src_embs, src_key_padding_mask=~src_mask)

        def length_norm(score, length):
            lp = ((5 + length) / 6) ** length_penalty
            return score / lp

        outputs = []
        for b in range(B):
            mem_b = memory[b:b+1]
            src_mask_b = src_mask[b:b+1]

            beams = [(0.0, [sos_id])]  # (logprob score, sequence)
            completed_beams = []

            for t in range(max_len):
                new_beams = []
                for score, seq in beams:
                    if seq[-1] == eos_id:
                        completed_beams.append((score, seq))
                        continue

                    ys = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)  # [1, len]
                    pos_t = torch.arange(ys.size(1), device=device).unsqueeze(0)
                    tgt_embs = self.token_emb(ys) + self.pos_emb(pos_t)

                    causal_mask = torch.triu(
                        torch.ones(ys.size(1), ys.size(1), device=device, dtype=torch.bool), diagonal=1
                    )

                    dec_out = self.decoder(
                        tgt_embs,
                        mem_b,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=~src_mask_b
                    )
                    logits = self.output_proj(dec_out[:, -1])  # [1,vocab]
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab]

                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                    for k in range(beam_width):
                        new_score = score + topk_log_probs[k].item()
                        new_seq = seq + [int(topk_ids[k].item())]
                        new_beams.append((new_score, new_seq))

                # prune
                beams = sorted(new_beams, key=lambda x: length_norm(x[0], len(x[1])), reverse=True)[:beam_width]

                if all(seq[-1] == eos_id for _, seq in beams):
                    break

            completed_beams.extend(beams)
            best_seq = max(completed_beams, key=lambda x: length_norm(x[0], len(x[1])))[1]

            # strip SOS + EOS
            if eos_id in best_seq:
                eos_idx = best_seq.index(eos_id)
                best_seq = best_seq[1:eos_idx]
            else:
                best_seq = best_seq[1:]
            outputs.append(best_seq)

        return outputs


# ----------------
# Quick sanity test
# ----------------
if __name__ == "__main__":
    vocab_size = 50
    model = BaselineModel(vocab_size=vocab_size, d_model=32, nhead=2)

    src_ids = torch.randint(0, vocab_size, (2, 7))
    src_mask = torch.ones_like(src_ids, dtype=torch.bool)
    tgt_ids = torch.randint(0, vocab_size, (2, 5))
    tgt_mask = torch.ones_like(tgt_ids, dtype=torch.bool)

    logits = model(src_ids, src_mask, tgt_ids, tgt_mask)
    print("logits shape:", logits.shape)

    greedy_decoded = model.greedy_decode(src_ids, src_mask, sos_id=1, eos_id=2, max_len=10)
    print("decoded IDs:", greedy_decoded)

    beam_decoded = model.beam_search_decode(src_ids, src_mask, sos_id=1, eos_id=2, max_len=10, beam_width=3)
    print("beam search decoded IDs:", beam_decoded)
