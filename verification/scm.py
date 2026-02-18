"""
B-RASP → SCM compilation for increment and decrement.

Increment L_n:                          Decrement D_n:
  B_prev(k) := b_k                       B^D_prev(k) := b_k
  C_in(k)   := ∧_{j=k+1}^n  b_j         W_in(k)     := ∧_{j=k+1}^n ¬b_j
  Y_out(k)  := B_prev(k) ⊕ C_in(k)      Y^D_out(k)  := B^D_prev(k) ⊕ W_in(k)

DAG: B_prev → Y_out ← C_in/W_in   (same topology, different structural equations)

Component predictions:
  B_prev  → L1-attn  (attention copy)
  C_in/W_in → L1-MLP (chain computation)
  Y_out   → L2-MLP   (XOR output)
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CausalVariable:
    name: str
    compute: Callable[[list[int], int], int]
    predicted_component: tuple[int, str]   # (layer, 'attn'|'mlp')


@dataclass
class CompiledSCM:
    task: str
    variables: list[CausalVariable]

    def evaluate(self, bits: list[int], k: int) -> dict[str, int]:
        return {v.name: v.compute(bits, k) for v in self.variables}


# ── Structural equation helpers ──

def _carry(bits, k):
    n = len(bits)
    return 1 if k == n - 1 else int(all(bits[j] for j in range(k + 1, n)))

def _borrow(bits, k):
    n = len(bits)
    return 1 if k == n - 1 else int(all(bits[j] == 0 for j in range(k + 1, n)))


# ── Compiled SCMs ──

def compile_increment() -> CompiledSCM:
    return CompiledSCM("increment", [
        CausalVariable("B_prev", lambda b, k: b[k],                (1, "attn")),
        CausalVariable("C_in",   _carry,                           (1, "mlp")),
        CausalVariable("Y_out",  lambda b, k: b[k] ^ _carry(b, k),(2, "mlp")),
    ])

def compile_decrement() -> CompiledSCM:
    return CompiledSCM("decrement", [
        CausalVariable("B_prev_D", lambda b, k: b[k],                  (1, "attn")),
        CausalVariable("W_in",     _borrow,                            (1, "mlp")),
        CausalVariable("Y_out_D",  lambda b, k: b[k] ^ _borrow(b, k), (2, "mlp")),
    ])

def compile_wrong_scm(op="and") -> CompiledSCM:
    """Wrong-SCM negative control: replace XOR with AND or OR."""
    fn = (lambda b, k: b[k] & _carry(b, k)) if op == "and" \
         else (lambda b, k: b[k] | _carry(b, k))
    return CompiledSCM(f"wrong_{op}", [
        CausalVariable("B_prev", lambda b, k: b[k], (1, "attn")),
        CausalVariable("C_in",   _carry,             (1, "mlp")),
        CausalVariable("Y_out",  fn,                  (2, "mlp")),
    ])
