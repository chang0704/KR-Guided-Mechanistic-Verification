#!/usr/bin/env python3
"""
Run architecture and hyperparameter ablations.

Usage:
  python scripts/run_ablations.py --type depth --n 16
  python scripts/run_ablations.py --type weight_decay --n 16
  python scripts/run_ablations.py --type subspace_dim --n 16
  python scripts/run_ablations.py --type cross_task --n 16
"""

import argparse, subprocess, sys
from pathlib import Path
import yaml

ROOT = str(Path(__file__).resolve().parent.parent)


def run(cmd):
    print(f"\n{'='*60}\n  {' '.join(cmd)}\n{'='*60}")
    subprocess.run([sys.executable] + cmd, cwd=ROOT)


def main(args):
    cfg = yaml.safe_load(open(Path(ROOT) / "configs/default.yaml"))

    if args.type == "depth":
        # ℓ ∈ {1, 2, 3}
        for l in cfg["ablations"]["depth"]:
            run(["scripts/train.py", "--task", "increment", "--n", str(args.n),
                 "--layers", str(l), "--seeds", str(args.seeds)])

    elif args.type == "weight_decay":
        # λ ∈ {0, 0.1, 1.0}
        for wd in cfg["ablations"]["weight_decay"]:
            run(["scripts/train.py", "--task", "increment", "--n", str(args.n),
                 "--weight-decay", str(wd), "--seeds", str(args.seeds)])

    elif args.type == "subspace_dim":
        # r ∈ {1, 2, 4, 8} — requires DAS with different r
        print("Subspace dimension ablation: train once, then run DAS with r ∈ {1, 2, 4, 8}")
        run(["scripts/train.py", "--task", "increment", "--n", str(args.n),
             "--seeds", str(args.seeds)])
        for r in cfg["ablations"]["subspace_dim"]:
            print(f"\n  DAS with r = {r}")
            run(["scripts/verify_das.py", "--task", "increment", "--n", str(args.n),
                 "--seeds", str(args.seeds)])

    elif args.type == "cross_task":
        # Train both tasks, then cross-apply SCMs
        for task in ["increment", "decrement"]:
            run(["scripts/train.py", "--task", task, "--n", str(args.n),
                 "--seeds", str(args.seeds)])
        print("\nCross-task: apply H_{L_n} to D_n model and H_{D_n} to L_n model")
        print("(Requires custom DAS script — see paper Appendix B)")

    elif args.type == "train_fraction":
        # α ∈ {0.3, 0.5, 0.7}
        for alpha in cfg["ablations"]["train_fraction"]:
            run(["scripts/train.py", "--task", "increment", "--n", str(args.n),
                 "--alpha", str(alpha), "--seeds", str(args.seeds)])

    else:
        print(f"Unknown ablation type: {args.type}")
        print("Options: depth, weight_decay, subspace_dim, cross_task, train_fraction")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True,
                   choices=["depth", "weight_decay", "subspace_dim", "cross_task", "train_fraction"])
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    main(p.parse_args())
