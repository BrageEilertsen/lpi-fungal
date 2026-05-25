"""Delta-formula analyzer: which tailoring-edit multisets exactly supply Delta.

Delta = formula(y) - formula(core) is the exact net atoms tailoring must add. Given the
admissible edit types (gene budget) we enumerate non-negative edit-type multisets whose
summed formula deltas equal Delta, subject to a parsimony cap and per-family gene-count
limits (+ slack). A non-empty result is a hard, threshold-free feasibility certificate
for the formula budget; emptiness prunes the (z_pks, z_tail) candidate outright.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from lpi.executor.tailoring import EditType, admissible_edits

_ELEMENTS = ("C", "H", "O", "Cl", "Br", "N", "F", "S")


def formula_counter(mol) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for a in mol.GetAtoms():
        c[a.GetSymbol()] += 1
        c["H"] += a.GetTotalNumHs()
    return c


def delta(target, core) -> dict[str, int]:
    ty, co = formula_counter(target), formula_counter(core)
    keys = set(ty) | set(co)
    return {k: ty.get(k, 0) - co.get(k, 0) for k in keys if ty.get(k, 0) - co.get(k, 0)}


@dataclass
class EditMultiset:
    counts: dict[str, int]  # edit-type name -> count

    @property
    def n_edits(self) -> int:
        return sum(self.counts.values())


def _vec(d: dict[str, int]) -> tuple[int, ...]:
    return tuple(d.get(e, 0) for e in _ELEMENTS)


def explain_delta(target_delta: dict[str, int], gene_budget: dict[str, int],
                  max_edits: int = 6, family_slack: int = 1,
                  max_solutions: int = 64) -> list[EditMultiset]:
    """Edit-type multisets (admissible under gene_budget) whose deltas sum to target_delta.

    Parsimony: total edits <= max_edits. Gene budget: per-family edit count <=
    gene_budget[family] + family_slack. N in the delta is NOT supplied by any tailoring
    edit (NRPS-appended amino acids carry N) -> such targets return [] here and are bucketed
    separately by the caller.
    """
    if target_delta.get("N", 0) != 0:
        return []  # nitrogen => NRPS fusion, not a tailoring edit in this vocabulary
    # Collapse onto distinct (family, delta) classes: hydroxylation/epoxidation (both
    # oxygenase +O) and o-/c-methylation (both methyltransferase +CH2) are chemically
    # interchangeable at the formula/budget level, so counting them separately would
    # inflate |Z*(y)| with relabellings of the same budget.
    seen_sig: set[tuple] = set()
    edits = []
    for e in admissible_edits(gene_budget):
        sig = (e.family, tuple(sorted(e.delta.items())))
        if sig not in seen_sig:
            seen_sig.add(sig)
            edits.append(e)
    if not edits and any(target_delta.values()):
        return []
    target = _vec(target_delta)
    fam_cap = {fam: gene_budget.get(fam, 0) + family_slack
               for fam in {e.family for e in edits}}

    solutions: list[EditMultiset] = []

    def recurse(i: int, remaining: tuple[int, ...], used: dict[str, int],
                fam_used: dict[str, int], n: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if all(v == 0 for v in remaining):
            solutions.append(EditMultiset(dict(used)))
            return
        if i >= len(edits) or n >= max_edits:
            return
        e = edits[i]
        ev = _vec(e.delta)
        cap = fam_cap.get(e.family, 0) - fam_used.get(e.family, 0)
        max_k = min(max_edits - n, cap)
        for k in range(0, max_k + 1):
            new_rem = tuple(r - k * d for r, d in zip(remaining, ev))
            new_used = dict(used)
            new_fam = dict(fam_used)
            if k:
                new_used[e.name] = k
                new_fam[e.family] = new_fam.get(e.family, 0) + k
            recurse(i + 1, new_rem, new_used, new_fam, n + k)

    recurse(0, target, {}, {}, 0)
    return solutions


def is_explainable(target, core, gene_budget: dict[str, int], **kw) -> bool:
    return bool(explain_delta(delta(target, core), gene_budget, **kw))
