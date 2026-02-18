#!/usr/bin/env python3
"""
SAE cross-validation pipeline.

Usage:
  python scripts/verify_sae.py --task increment --n 16 --seeds 5
"""

import argparse, glob, sys
from pathlib import Path
import torch, yaml, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.binary_arithmetic import BinaryArithmeticDataset
from src.models.transformer import BinaryArithmeticTransformer
from src.verification.scm import compile_increment, compile_decrement
from src.verification.sae import SAEAnalyzer


def main(args):
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scm = compile_increment() if args.task == "increment" else compile_decrement()

    # ── Find last checkpoint per seed ──
    pattern = f"checkpoints/{args.task}_transformer_n{args.n}_s*_step*.pt"
    all_p = sorted(glob.glob(pattern))
    by_seed = {}
    for p in all_p:
        s = int(p.split("_s")[1].split("_")[0])
        by_seed[s] = p
    paths = [by_seed[s] for s in sorted(by_seed)][:args.seeds]

    print(f"SAE analysis on {len(paths)} seed(s), {args.task} n={args.n}")

    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BinaryArithmeticTransformer(
            d=cfg["model"]["d_model"], layers=cfg["model"]["layers"],
            heads=cfg["model"]["heads"], d_mlp=cfg["model"]["d_mlp"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        ds = BinaryArithmeticDataset(args.n, args.task, "test")
        analyzer = SAEAnalyzer(
            d=cfg["model"]["d_model"],
            mult=cfg["sae"]["dict_multiplier"],
            K=cfg["sae"]["topk"],
            lr=cfg["sae"]["lr"],
            bs=cfg["sae"]["batch_size"],
            epochs=cfg["sae"]["epochs"],
            n_passes=cfg["sae"]["num_forward_passes"],
            device=device,
        )

        for var in scm.variables:
            layer, ctype = var.predicted_component
            cname = f"L{layer}-{'attn' if ctype == 'attn' else 'MLP'}"

            # Collect activations
            acts = analyzer.collect(model, ds, layer, ctype)

            # Collect variable values
            var_vals = []
            for i in range(min(analyzer.n_passes, len(ds))):
                tok, _ = ds[i]
                bits = tok[:args.n].tolist()
                for k in range(args.n):
                    var_vals.append(var.compute(bits, k))
            var_vals = torch.tensor(var_vals[:acts.shape[0]], dtype=torch.float)

            # Train SAE
            sae = analyzer.train(acts)
            r2 = analyzer.r_squared(sae, acts)

            # Best feature
            res = analyzer.best_feature(sae, acts, var_vals)
            rho2 = analyzer.second_best_rho(sae, acts, var_vals, res["feature_idx"])

            print(f"  {var.name:12s} @ {cname}:  z_{res['feature_idx']:>4d}  "
                  f"|ρ|={res['rho']:.2f}  acc={res['accuracy']:.2f}  "
                  f"R²={r2:.3f}  2nd={rho2:.2f}  Δ={res['rho']-rho2:.2f}  "
                  f"{'✓ mono' if res['monosemantic'] else '✗'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="increment")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    main(p.parse_args())
