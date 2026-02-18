"""
Distributed Alignment Search (DAS) with IIA trajectory tracking.

DAS learns R ∈ O(d) identifying an r-dim subspace encoding causal variable V_i.
Interchange Intervention Accuracy (IIA) measures counterfactual faithfulness.

Config (§4.4): r=1, Adam (0.9, 0.999, lr=1e-3), 10 epochs,
  2,000 train / 1,000 eval base–source pairs per (variable, component, position).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional


class OrthogonalRotation(nn.Module):
    """Learnable rotation R ∈ O(d) via Cayley parameterization."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.A = nn.Parameter(torch.zeros(d, d))

    def matrix(self):
        S = self.A - self.A.T
        I = torch.eye(self.d, device=self.A.device)
        return torch.linalg.solve(I + S, I - S)


class DASVerifier:
    """
    Runs DAS for each (variable, component, position) triple.

    Args:
        model: Trained Transformer with get_activations() method.
        d: Hidden dimension (128).
        r: Subspace dimension (1).
        lr, betas, epochs: DAS optimizer settings.
    """

    def __init__(self, model, d=128, r=1, lr=1e-3, betas=(0.9, 0.999),
                 epochs=10, device="cuda"):
        self.model = model
        self.d, self.r = d, r
        self.lr, self.betas, self.epochs = lr, betas, epochs
        self.device = device

    def _intervene(self, base_act, source_act, R):
        """Swap first r dims in rotated space."""
        b_rot = base_act @ R
        s_rot = source_act @ R
        out = b_rot.clone()
        out[:, :self.r] = s_rot[:, :self.r]
        return out @ R.T

    def train_rotation(self, base_act, source_act, target, n_epochs=None):
        """Train R on one (variable, component, position) triple.

        Returns: (R matrix, final IIA)
        """
        rot = OrthogonalRotation(self.d).to(self.device)
        opt = optim.Adam(rot.parameters(), lr=self.lr, betas=self.betas)
        ce = nn.CrossEntropyLoss()

        for _ in range(n_epochs or self.epochs):
            opt.zero_grad()
            R = rot.matrix()
            h_int = self._intervene(base_act, source_act, R)
            logits = self.model.head(self.model.ln_f(h_int))
            loss = ce(logits, target)
            loss.backward()
            opt.step()

        with torch.no_grad():
            R = rot.matrix()
            h_int = self._intervene(base_act, source_act, R)
            preds = self.model.head(self.model.ln_f(h_int)).argmax(-1)
            iia = (preds == target).float().mean().item()
        return R.detach(), iia

    def verify_all(self, dataset, scm, n, n_train=2000, n_eval=1000):
        """
        Full DAS verification: all variables × all components × all positions.

        Returns: dict[var_name][comp_name] → mean IIA
        """
        self.model.eval()
        components = [(1, "attn"), (1, "mlp"), (2, "attn"), (2, "mlp")]
        results = {}

        for var in scm.variables:
            var_results = {}
            for layer, ctype in components:
                cname = f"L{layer}-{'at' if ctype == 'attn' else 'M'}"
                pos_iias = []

                for k in range(n):
                    # Sample base-source pairs where V_i(k) differs
                    base_tok, source_tok, targets = self._sample_pairs(
                        dataset, var, k, n_train + n_eval)

                    if base_tok is None:
                        pos_iias.append(0.5)
                        continue

                    out_pos = n + 1 + k
                    with torch.no_grad():
                        b_act = self.model.get_activations(
                            base_tok[:n_train].to(self.device),
                            layer, ctype)[:, out_pos]
                        s_act = self.model.get_activations(
                            source_tok[:n_train].to(self.device),
                            layer, ctype)[:, out_pos]

                    _, iia = self.train_rotation(
                        b_act, s_act,
                        targets[:n_train].to(self.device))
                    pos_iias.append(iia)

                var_results[cname] = np.mean(pos_iias)
            results[var.name] = var_results

        return results

    def _sample_pairs(self, dataset, var, k, n_pairs):
        """Sample base-source pairs where var(k) differs."""
        group0, group1 = [], []
        for i in range(len(dataset)):
            tok, cv = dataset[i]
            bits = tok[:dataset.n].tolist()
            val = var.compute(bits, k)
            (group0 if val == 0 else group1).append(i)

        if not group0 or not group1:
            return None, None, None

        rng = np.random.RandomState(k)
        m = min(n_pairs, len(group0), len(group1))
        idx0 = rng.choice(group0, m, replace=len(group0) < m)
        idx1 = rng.choice(group1, m, replace=len(group1) < m)

        base_tok = torch.stack([dataset[i][0] for i in idx0])
        source_tok = torch.stack([dataset[i][0] for i in idx1])

        # Target: what base output should become after intervention
        n = dataset.n
        targets = source_tok[:, n + 1 + k]
        return base_tok, source_tok, targets

    def iia_trajectory(self, ckpt_paths, scm, n, best_components, dataset):
        """
        Compute IIA_min trajectory across training checkpoints.

        Returns: list of (step, iia_min) tuples.
        """
        trajectory = []
        for path in ckpt_paths:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            step = ckpt["step"]

            var_iias = []
            for var, (layer, ctype) in zip(scm.variables, best_components):
                pos_iias = []
                for k in range(n):
                    base_tok, source_tok, targets = self._sample_pairs(
                        dataset, var, k, 1000)
                    if base_tok is None:
                        pos_iias.append(0.5)
                        continue
                    out_pos = n + 1 + k
                    with torch.no_grad():
                        b_act = self.model.get_activations(
                            base_tok.to(self.device), layer, ctype)[:, out_pos]
                        s_act = self.model.get_activations(
                            source_tok.to(self.device), layer, ctype)[:, out_pos]
                    _, iia = self.train_rotation(b_act, s_act,
                                                 targets.to(self.device))
                    pos_iias.append(iia)
                var_iias.append(np.mean(pos_iias))
            trajectory.append((step, min(var_iias)))

        return trajectory
