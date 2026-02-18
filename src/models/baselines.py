"""
Parameter-matched recurrent / SSM baselines.

Hidden dims chosen so total params ≤ Transformer's 403,840:
  LSTM  d_h=223  →  400,957
  GRU   d_h=258  →  402,483
  Mamba d_h=239  →  403,674
"""

import torch
import torch.nn as nn


class RecurrentBaseline(nn.Module):
    def __init__(self, vocab=3, d_h=223, rnn="lstm"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_h)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(d_h, d_h, batch_first=True)
        self.head = nn.Linear(d_h, vocab, bias=False)

    def forward(self, idx):
        out, _ = self.rnn(self.emb(idx))
        return self.head(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class MambaBaseline(nn.Module):
    """Simplified selective-scan SSM baseline."""
    def __init__(self, vocab=3, d_h=239, d_state=16, d_conv=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_h)
        self.in_proj = nn.Linear(d_h, d_h * 2, bias=False)
        self.conv = nn.Conv1d(d_h, d_h, d_conv, padding=d_conv - 1, groups=d_h)
        self.x_proj = nn.Linear(d_h, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_h, bias=True)
        self.out_proj = nn.Linear(d_h, d_h, bias=False)
        self.head = nn.Linear(d_h, vocab, bias=False)
        self.D = nn.Parameter(torch.ones(d_h))

    def forward(self, idx):
        x = self.emb(idx)
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_c = self.conv(x_ssm.transpose(1, 2))[:, :, :x_ssm.shape[1]].transpose(1, 2)
        x_c = torch.silu(x_c)
        bp = self.x_proj(x_c)
        dt = torch.softplus(self.dt_proj(bp[:, :, :bp.shape[-1] // 2]))
        out = x_c * dt + x_c * self.D
        out = self.out_proj(out * torch.silu(z))
        return self.head(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


BASELINE_CONFIGS = {
    "lstm":  {"rnn": "lstm", "d_h": 223},
    "gru":   {"rnn": "gru",  "d_h": 258},
    "mamba": {"d_h": 239},
}

def build_baseline(name, vocab=3):
    if name in ("lstm", "gru"):
        c = BASELINE_CONFIGS[name]
        return RecurrentBaseline(vocab, c["d_h"], c["rnn"])
    elif name == "mamba":
        return MambaBaseline(vocab, BASELINE_CONFIGS["mamba"]["d_h"])
    raise ValueError(name)
