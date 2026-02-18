"""
TopK Sparse Autoencoder for unsupervised feature discovery.

Config (§4.5): D=8d=1024, K=8, Adam lr=3e-4, batch 256, 50 epochs,
  20,000 forward passes, decoder columns renormalized to unit norm.
Monosemantic: |ρ| > 0.9 and accuracy > 0.95.
DAS–SAE convergence: cos > 0.8.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder: h_hat = W_dec @ TopK(W_enc @ h + b, K) + b_dec."""

    def __init__(self, d_in, D, K):
        super().__init__()
        self.D, self.K = D, K
        self.encoder = nn.Linear(d_in, D, bias=True)
        self.decoder = nn.Linear(D, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))

    def encode(self, h):
        z = self.encoder(h)
        topk_val, topk_idx = z.topk(self.K, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, topk_val)
        return sparse

    def forward(self, h):
        z = self.encode(h)
        return self.decoder(z), z

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(
            self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))


class SAEAnalyzer:
    """Train SAE and compute feature–variable alignment."""

    def __init__(self, d=128, mult=8, K=8, lr=3e-4, bs=256, epochs=50,
                 n_passes=20000, device="cuda"):
        self.d, self.D = d, d * mult
        self.K, self.lr, self.bs = K, lr, bs
        self.epochs, self.n_passes = epochs, n_passes
        self.device = device

    def collect(self, model, dataset, layer, ctype, positions=None):
        """Collect activations from converged model."""
        model.eval()
        n = dataset.n
        acts = []
        with torch.no_grad():
            for i in range(min(self.n_passes, len(dataset))):
                tok, _ = dataset[i]
                tok = tok.unsqueeze(0).to(self.device)
                h = model.get_activations(tok, layer, ctype)
                if positions is not None:
                    for p in positions:
                        acts.append(h[0, n + 1 + p])
                else:
                    for k in range(n):
                        acts.append(h[0, n + 1 + k])
        return torch.stack(acts)

    def train(self, activations):
        """Train TopK SAE on activations tensor."""
        sae = TopKSAE(self.d, self.D, self.K).to(self.device)
        opt = optim.Adam(sae.parameters(), lr=self.lr)
        N = activations.shape[0]

        for ep in range(self.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, self.bs):
                batch = activations[perm[i:i + self.bs]]
                h_hat, z = sae(batch)
                loss = (batch - h_hat).pow(2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                sae.normalize_decoder()
        return sae

    def r_squared(self, sae, activations):
        """R² reconstruction quality."""
        sae.eval()
        with torch.no_grad():
            h_hat, _ = sae(activations)
            mu = activations.mean(0, keepdim=True)
            return 1.0 - (activations - h_hat).pow(2).sum() / \
                   (activations - mu).pow(2).sum()

    def best_feature(self, sae, activations, var_vals):
        """Find SAE feature with highest |ρ| to a causal variable."""
        sae.eval()
        with torch.no_grad():
            _, z = sae(activations)
        z_np, v_np = z.cpu().numpy(), var_vals.cpu().numpy()

        best_j, best_rho, best_acc = -1, 0.0, 0.0
        for j in range(self.D):
            rho, _ = pearsonr(z_np[:, j], v_np)
            acc = np.mean((z_np[:, j] > 0).astype(float) == v_np)
            if abs(rho) > abs(best_rho):
                best_j, best_rho, best_acc = j, rho, acc

        return {
            "feature_idx": best_j,
            "rho": abs(best_rho),
            "accuracy": best_acc,
            "monosemantic": abs(best_rho) > 0.9 and best_acc > 0.95,
        }

    def second_best_rho(self, sae, activations, var_vals, exclude_j):
        """Second-best |ρ| → compute uniqueness gap Δ|ρ|."""
        sae.eval()
        with torch.no_grad():
            _, z = sae(activations)
        z_np, v_np = z.cpu().numpy(), var_vals.cpu().numpy()

        best2 = 0.0
        for j in range(self.D):
            if j == exclude_j:
                continue
            rho, _ = pearsonr(z_np[:, j], v_np)
            best2 = max(best2, abs(rho))
        return best2

    @staticmethod
    def das_sae_cosine(sae, feature_j, das_direction):
        """Cosine similarity between DAS direction and SAE decoder column."""
        col = sae.decoder.weight[:, feature_j]
        return abs(torch.nn.functional.cosine_similarity(
            das_direction.unsqueeze(0), col.unsqueeze(0)).item())
