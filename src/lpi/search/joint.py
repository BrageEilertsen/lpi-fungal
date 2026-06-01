"""Joint (z_pks, z_tail) verification: explain a tailored product exactly-by-budget.

For a target product y and the cluster's gene budget, find PKS cores (z_pks) and a
tailoring-edit multiset (z_tail) such that:
  * the core's carbon skeleton SKELETON-COMPLETELY embeds in y (additive tailoring keeps
    the backbone; unmatched y-carbons are single-bond-separable decorations), and
  * Delta = formula(y) - formula(core) is exactly supplied by a gene+formula-budgeted,
    parsimonious edit multiset (delta_formula.explain_delta).

A cluster is *budget-verified* if such (core, edit-multiset) exists. This is threshold-
free: the backbone is in y and every net atom is accounted for by an enzyme the cluster
actually has. It is a strong proxy for full atom-level exact reconstruction (the remaining
gap is edit POSITIONS, which affect |Z*| multiplicity, not existence) -- positional
application is the next refinement.

The PKS-core enumeration is bounded (carbon band + per-depth beam + program cap), so the
verified count is a LOWER BOUND under the current executor + bound settings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rdkit import Chem

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Extender, Program, ReductionState, Release
from lpi.eval.corematch import heavy_skeleton_complete_extra_bonds
from lpi.executor import core as _core
from lpi.search.delta_formula import delta, explain_delta
from lpi.search.beam import _carbon_count

_REDUCTIONS = (ReductionState.KETO, ReductionState.KR, ReductionState.DH, ReductionState.ER)
_RELEASES = (Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
             Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE, Release.MACROLACTONIZATION)


@dataclass
class Explanation:
    program: Program
    core_smiles: str
    edit_counts: dict[str, int]
    n_edits: int


@dataclass
class JointResult:
    target: str
    explanations: list[Explanation] = field(default_factory=list)
    n_cores_tried: int = 0
    runtime_s: float = 0.0

    @property
    def verified(self) -> bool:
        return bool(self.explanations)

    @property
    def min_edits(self) -> int | None:
        return min((e.n_edits for e in self.explanations), default=None)

    @property
    def z_star_size(self) -> int:
        """Parsimonious |Z*|: distinct explanations at the minimal edit count."""
        m = self.min_edits
        return sum(1 for e in self.explanations if e.n_edits == m) if m is not None else 0


def _enumerate_cores(max_carbons: int, min_carbons: int, starters: tuple[str, ...],
                     beam_width: int, program_cap: int):
    """Yield (Program, core Mol), LARGEST core first (parsimony: a larger core needs fewer
    tailoring edits). For each chain length (n_cycles, descending) we beam over reduction/
    methyl patterns x release modes, exec, and yield cores in the carbon band."""
    cycle_opts = [Cycle(reduction=r, c_methyl=me)
                  for r in _REDUCTIONS for me in (False, True)]
    yielded = 0
    starter_c = {"acetyl": 2, "propionyl": 3, "butyryl": 4}
    for starter in starters:
        sc = starter_c.get(starter, 2)
        max_cycles = max(1, (max_carbons - sc) // 2 + 1)
        for n_cycles in range(max_cycles, 0, -1):  # longest chains first
            # beam over the n_cycles-long reduction/methyl patterns
            beam: list[tuple[Cycle, ...]] = [()]
            for _ in range(n_cycles):
                nxt: list[tuple[Cycle, ...]] = []
                for cycles in beam:
                    for cyc in cycle_opts:
                        nc = cycles + (cyc,)
                        cc = sc + sum(3 if c.c_methyl else 2 for c in nc)
                        if cc <= max_carbons:
                            nxt.append(nc)
                beam = nxt[:beam_width]
            for nc in beam:
                if len(nc) != n_cycles:
                    continue
                for rel in _RELEASES:
                    prog = Program(starter, nc, rel)
                    try:
                        mol = _core.exec(prog)
                    except Exception:  # noqa: BLE001
                        continue
                    if min_carbons <= _carbon_count(mol) <= max_carbons:
                        yield prog, mol
                        yielded += 1
                        if yielded >= program_cap:
                            return


def explain_cluster(target_smiles: str, gene_budget: dict[str, int],
                    starters: tuple[str, ...] = ("acetyl", "propionyl"),
                    max_edits: int = 6, max_decoration_carbons: int = 12,
                    max_extra_ring_bonds: int = 2, beam_width: int = 400,
                    program_cap: int = 6000, collect: int = 8) -> JointResult:
    t0 = time.perf_counter()
    y = M.mol_from_smiles(target_smiles)
    y_c = _carbon_count(y)
    res = JointResult(target=target_smiles)
    seen: set[str] = set()
    min_c = max(2, y_c - max_decoration_carbons)
    best = max_edits + 1  # current minimal edit count among verified explanations
    for prog, core in _enumerate_cores(y_c, min_c, starters, beam_width, program_cap):
        res.n_cores_tried += 1
        ms = explain_delta(delta(y, core), gene_budget, max_edits=max_edits)
        if not ms:
            continue
        best_ms = min(ms, key=lambda m: m.n_edits)
        if best_ms.n_edits > best:
            continue  # parsimony prune: cannot be a minimal-edit explanation
        core_smiles = M.canonical_smiles(core)
        # |Z*(y)| counts distinct chemical explanations (core structure + canonical edit
        # multiset), not distinct PKS programs that happen to yield the same core.
        key = (core_smiles, tuple(sorted(best_ms.counts.items())))
        if key in seen:
            continue
        # heavy-atom skeleton embedding (keeps O-/N-linked decorations connected); allow
        # as many extra ring-closure bonds as there are oxidative_cyclization edits.
        ring_edits = best_ms.counts.get("oxidative_cyclization", 0)
        if heavy_skeleton_complete_extra_bonds(
                core, y, max_extra_ring_bonds=max(max_extra_ring_bonds, ring_edits)) is None:
            continue
        seen.add(key)
        e = best_ms.n_edits
        expl = Explanation(prog, core_smiles, best_ms.counts, e)
        if e < best:
            best = e
            res.explanations = [x for x in res.explanations if x.n_edits <= best]
        res.explanations.append(expl)
        if best == 0 and res.z_star_size >= collect:
            break  # enough zero-edit (exact PKS) explanations gathered
    res.explanations.sort(key=lambda x: x.n_edits)
    res.runtime_s = time.perf_counter() - t0
    return res
