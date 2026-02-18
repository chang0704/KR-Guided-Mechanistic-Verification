#!/usr/bin/env python3
"""
DAS verification pipeline.

Usage:
  python scripts/verify_das.py --task increment --n 16 --seeds 5
  python scripts/verify_das.py --checkpoint checkpoints/increment_transformer_n16_s0_step99800.pt
"""

import argparse, glob, sys
from pathlib import Path
import torch, yaml, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.binary_arithmetic import BinaryArithmeticDataset
from src.models.transformer import BinaryArithmeticTransformer
from src.verification.scm import compile_increment, compile_decrement
from src.verification.das import DASVerifier


def main(args):
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scm = compile_increment() if args.task == "increment" else compile_decrement()

    # ── Find checkpoints ──
    if args.checkpoint:
        paths = [args.checkpoint]
    else:
        pattern = f"checkpoints/{args.task}_transformer_n{args.n}_s*_step*.pt"
        all_p = sorted(glob.glob(pattern))
        by_seed = {}
        for p in all_p:
            s = int(p.split("_s")[1].split("_")[0])
            by_seed[s] = p  # last step per seed
        paths = [by_seed[s] for s in sorted(by_seed)][:args.seeds]

    print(f"Verifying {len(paths)} checkpoint(s) on {args.task} n={args.n}")

    all_results = []
    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BinaryArithmeticTransformer(
            d=cfg["model"]["d_model"], layers=cfg["model"]["layers"],
            heads=cfg["model"]["heads"], d_mlp=cfg["model"]["d_mlp"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        ds = BinaryArithmeticDataset(args.n, args.task, "test")
        verifier = DASVerifier(model, d=cfg["model"]["d_model"],
                               r=cfg["das"]["subspace_dim"],
                               lr=cfg["das"]["lr"], epochs=cfg["das"]["epochs"],
                               device=device)
        results = verifier.verify_all(ds, scm, args.n,
                                      cfg["das"]["train_pairs"],
                                      cfg["das"]["eval_pairs"])
        for var_name, comp_dict in results.items():
            best = max(comp_dict, key=comp_dict.get)
            print(f"  {var_name:12s}: best={best} IIA={comp_dict[best]:.3f}")
        all_results.append(results)

    if len(all_results) > 1:
        print("\n══ Aggregated (mean ± std) ══")
        for vn in all_results[0]:
            for cn in all_results[0][vn]:
                vals = [r[vn][cn] for r in all_results]
                print(f"  {vn:12s} @ {cn}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--task", default="increment")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    main(p.parse_args())
