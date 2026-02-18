"""
Evaluation metrics for binary arithmetic and mechanistic verification.
"""

import torch
import numpy as np


def per_sequence_accuracy(logits, tokens, n):
    """Fraction of test sequences where all n output bits are correct."""
    preds = logits[:, n:-1].argmax(-1)          # predict y_1...y_n
    targets = tokens[:, n + 1:]                  # ground truth y_1...y_n
    return (preds == targets).all(-1).float().mean().item()


def per_position_accuracy(logits, tokens, n):
    """Fraction of correctly predicted output bits."""
    preds = logits[:, n:-1].argmax(-1)
    targets = tokens[:, n + 1:]
    return (preds == targets).float().mean().item()


def weight_norm(model):
    """L₂ weight norm ‖θ‖₂."""
    return np.sqrt(sum(p.data.pow(2).sum().item() for p in model.parameters()))


def grokking_step(accs, threshold=0.99):
    """First step where test per-seq acc ≥ threshold."""
    for step, acc in accs:
        if acc >= threshold:
            return step
    return None


def memorization_step(accs, threshold=1.0):
    """First step where train per-seq acc ≥ threshold."""
    for step, acc in accs:
        if acc >= threshold:
            return step
    return None
