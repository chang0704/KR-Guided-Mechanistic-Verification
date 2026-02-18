"""
Decoder-only Transformer for binary arithmetic.

Architecture (§4.2):
  ℓ=2, H=2, d=128, d_mlp=4d=512, GELU
  Pre-norm LayerNorm (ε=1e-5), learned pos/tok embeddings (T_max=49)
  No dropout, no output bias, no weight tying
  Kaiming uniform init → 403,840 parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, H: int, T_max: int):
        super().__init__()
        self.H, self.dk = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer("mask",
            torch.triu(torch.ones(T_max, T_max), diagonal=1).bool())

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.H, self.dk).transpose(1, 2)
        k = k.view(B, T, self.H, self.dk).transpose(1, 2)
        v = v.view(B, T, self.H, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.dk ** -0.5)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_mlp, bias=True)
        self.fc2 = nn.Linear(d_mlp, d, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d, H, d_mlp, T_max, eps=1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.attn = CausalSelfAttention(d, H, T_max)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.mlp = MLP(d, d_mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BinaryArithmeticTransformer(nn.Module):
    def __init__(self, vocab=3, d=128, layers=2, heads=2, d_mlp=512,
                 T_max=49, eps=1e-5):
        super().__init__()
        self.d = d
        self.n_layers = layers
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(T_max, d)
        self.blocks = nn.ModuleList(
            [Block(d, heads, d_mlp, T_max, eps) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d, eps=eps)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def get_activations(self, idx, layer: int, component: str):
        """Extract activations at layer (1-indexed), component ∈ {attn, mlp}."""
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for i, blk in enumerate(self.blocks):
            h = blk.ln1(x)
            a = x + blk.attn(h)
            if i + 1 == layer and component == "attn":
                return a
            m = a + blk.mlp(blk.ln2(a))
            if i + 1 == layer and component == "mlp":
                return m
            x = m
        raise ValueError(f"layer={layer}, component={component}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
