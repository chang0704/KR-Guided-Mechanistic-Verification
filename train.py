#!/usr/bin/env python3
"""
Train Transformer / baselines on n-bit binary arithmetic with grokking.

Usage:
  python scripts/train.py --task increment --n 16 --seeds 5
  python scripts/train.py --task decrement --n 16 --model lstm
"""

import argparse, math, os, sys
from pathlib import Path
import torch, torch.nn as nn, yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.binary_arithmetic import build_dataloaders
from src.models.transformer import BinaryArithmeticTransformer
from src.models.baselines import build_baseline
from src.utils.metrics import per_sequence_accuracy, weight_norm, grokking_step, memorization_step


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def main(args):
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        # ── Build model ──
        if args.model == "transformer":
            model = BinaryArithmeticTransformer(
                d=cfg["model"]["d_model"], layers=args.layers or cfg["model"]["layers"],
                heads=cfg["model"]["heads"], d_mlp=cfg["model"]["d_mlp"],
                T_max=cfg["model"]["max_seq_len"],
            ).to(device)
        else:
            model = build_baseline(args.model).to(device)
        print(f"[seed={seed}] {args.model} → {model.count_parameters():,} params")

        # ── Build data ──
        train_dl, test_dl = build_dataloaders(
            args.n, args.task, args.alpha, cfg["training"]["batch_size"], seed=seed)

        # ── Optimizer ──
        wd = args.weight_decay if args.weight_decay is not None else cfg["training"]["weight_decay"]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                betas=tuple(cfg["training"]["betas"]), weight_decay=wd)
        ce = nn.CrossEntropyLoss()
        total = cfg["training"]["total_steps"]
        warmup = cfg["training"]["warmup_steps"]
        ckpt_freq = cfg["training"]["checkpoint_every"]
        n = args.n

        # ── Training loop ──
        step, history = 0, []
        train_iter = iter(train_dl)
        pbar = tqdm(total=total, desc=f"seed={seed}")
        while step < total:
            try:
                tokens, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                tokens, _ = next(train_iter)
            tokens = tokens.to(device)

            for pg in opt.param_groups:
                pg["lr"] = cosine_lr(step, warmup, total, cfg["training"]["lr"])

            logits = model(tokens)
            loss = ce(logits[:, n:-1].reshape(-1, 3), tokens[:, n+1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

            if step % ckpt_freq == 0 or step == total - 1:
                model.eval()
                with torch.no_grad():
                    tr_acc = per_sequence_accuracy(logits, tokens, n)
                    te_accs = []
                    for tb, _ in test_dl:
                        tb = tb.to(device)
                        te_accs.append(per_sequence_accuracy(model(tb), tb, n))
                    te_acc = sum(te_accs) / max(len(te_accs), 1)
                    wn = weight_norm(model)
                history.append({"step": step, "train": tr_acc, "test": te_acc, "wn": wn})
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "step": step,
                    "config": cfg,
                    "metrics": {"train_seq_acc": tr_acc, "test_seq_acc": te_acc,
                                "weight_norm": wn},
                }, f"{args.output_dir}/{args.task}_{args.model}_n{n}_s{seed}_step{step}.pt")
                model.train()

            step += 1; pbar.update(1)
            if step % 2000 == 0 and history:
                h = history[-1]
                pbar.set_postfix(tr=f"{h['train']:.3f}", te=f"{h['test']:.3f}", wn=f"{h['wn']:.0f}")
        pbar.close()

        mem = memorization_step([(h["step"], h["train"]) for h in history])
        grok = grokking_step([(h["step"], h["test"]) for h in history])
        gap = (grok - mem) if mem and grok else "N/A"
        print(f"  → mem={mem}  grok={grok}  gap={gap}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["increment", "decrement"], default="increment")
    p.add_argument("--model", choices=["transformer", "lstm", "gru", "mamba"], default="transformer")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--output-dir", default="checkpoints")
    main(p.parse_args())
