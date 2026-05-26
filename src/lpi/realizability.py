"""Realizability: design-by-grammar-inversion over the natural-cluster manifold.

The third direction. The executor runs forward (program -> structure) and the verifier runs inverse
(structure -> programs). This module adds *design*: given a target program (e.g. one producing a
desired structure), how far is it from something nature already builds? A program ``z`` is realizable
to degree ``r`` if there is a known natural program ``z0`` reachable from ``z`` by ``r`` typed edits
-- the operations combinatorial biosynthesis actually performs (reductive-domain editing, AT
starter/extender swaps, module deletion, C-methyltransferase insertion, cyclization/release
reprogramming). The output is a *distance-to-realizable* coordinate for every target, plus the edit
path and a feasibility verdict.

Soundness boundary, stated plainly: the per-edit *tier* below is a CLASS-level feasibility judgment
(these edit classes are established in combinatorial biosynthesis). It does NOT attach specific paper
citations -- curating which exact edits are documented in the directed-evolution literature
(Khosla / Cane / Leadlay / ...) is a separate, web-verifiable data step, and is deliberately not
fabricated here. The metric's machinery is exact; the literature weighting is pluggable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from lpi.chem.program import Cycle, Program, ReductionState as R, Release


class Tier(IntEnum):
    """Class-level feasibility of an edit operation (cost weight); 1 = most established."""

    DOCUMENTED = 1     # an established combinatorial-biosynthesis edit class
    PLAUSIBLE = 2      # grammatically legal, harder / less commonly engineered
    SPECULATIVE = 3    # in-grammar but ~undocumented in vivo


# edit kind -> class-level tier (see module docstring: NOT specific-paper-cited)
_TIER: dict[str, Tier] = {
    "reduction": Tier.DOCUMENTED,    # KR/DH/ER activity edit / reductive-domain swap
    "extender": Tier.DOCUMENTED,     # AT extender-unit swap (e.g. malonyl<->methylmalonyl)
    "starter": Tier.DOCUMENTED,      # loading-module / starter-unit swap
    "cycle_remove": Tier.DOCUMENTED, # module deletion / reduced iteration count
    "c_methyl": Tier.PLAUSIBLE,      # C-methyltransferase insertion / removal
    "cycle_add": Tier.PLAUSIBLE,     # module insertion / extra iteration
    "release": Tier.SPECULATIVE,     # cyclase/thioesterase reprogramming (release-mode change)
}


@dataclass(frozen=True)
class Edit:
    kind: str
    detail: str
    tier: Tier


@dataclass
class Realizability:
    target: Program
    nearest_id: str
    nearest: Program
    edits: list[Edit]
    cost: int
    verdict: str        # 'natural' | 'engineerable' | 'speculative'

    @property
    def n_edits(self) -> int:
        return len(self.edits)


def edits_from(target: Program, template: Program) -> list[Edit]:
    """The typed edit operations that transform ``template`` into ``target`` (positional v1:
    cycle-by-cycle up to the shorter program, plus chain-length and starter/release deltas)."""
    out: list[Edit] = []
    if target.starter != template.starter:
        out.append(Edit("starter", f"{template.starter}->{target.starter}", _TIER["starter"]))
    if target.release != template.release:
        out.append(Edit("release", f"{template.release.value}->{target.release.value}",
                        _TIER["release"]))
    nt, nm = len(target.cycles), len(template.cycles)
    for i in range(min(nt, nm)):
        ct, cm = target.cycles[i], template.cycles[i]
        if ct.reduction != cm.reduction:
            out.append(Edit("reduction", f"cyc{i + 1} {cm.reduction.value}->{ct.reduction.value}",
                            _TIER["reduction"]))
        if ct.c_methyl != cm.c_methyl:
            out.append(Edit("c_methyl", f"cyc{i + 1} {'+' if ct.c_methyl else '-'}C-MeT",
                            _TIER["c_methyl"]))
        if ct.extender != cm.extender:
            out.append(Edit("extender", f"cyc{i + 1} {cm.extender.value}->{ct.extender.value}",
                            _TIER["extender"]))
    for i in range(min(nt, nm), max(nt, nm)):
        if nt > nm:
            out.append(Edit("cycle_add", f"+cyc{i + 1}", _TIER["cycle_add"]))
        else:
            out.append(Edit("cycle_remove", f"-cyc{i + 1}", _TIER["cycle_remove"]))
    return out


def cost(edits: list[Edit]) -> int:
    """Tier-weighted edit cost (DOCUMENTED edits are cheap; SPECULATIVE ones expensive)."""
    return sum(int(e.tier) for e in edits)


#: at most this many documented/plausible edits still counts as "engineerable".
ENGINEERABLE_MAX_EDITS = 3


def _verdict(edits: list[Edit]) -> str:
    """natural (already exists) | engineerable (a few documented/plausible edits from a natural
    cluster) | speculative (a tier-3/undocumented edit, or too many edits to be a near neighbour)."""
    if not edits:
        return "natural"
    if any(e.tier == Tier.SPECULATIVE for e in edits):
        return "speculative"
    if len(edits) <= ENGINEERABLE_MAX_EDITS:
        return "engineerable"
    return "speculative"


def realize(target: Program, manifold: dict[str, Program]) -> Realizability:
    """Distance-to-realizable: the nearest natural program, the edit path, cost, and verdict."""
    best: tuple[str, Program, list[Edit], int] | None = None
    for cid, z0 in manifold.items():
        es = edits_from(target, z0)
        c = cost(es)
        if best is None or c < best[3]:
            best = (cid, z0, es, c)
    cid, z0, es, c = best  # type: ignore[misc]
    return Realizability(target, cid, z0, es, c, _verdict(es))


# ---- the natural-cluster manifold (real reachable fungal PKS programs) -------------
# Programs are the exact reachable solutions from the MIBiG reachability scan (results/reachability.csv).
def natural_manifold() -> dict[str, Program]:
    K, KR, DH = R.KETO, R.KR, R.DH
    return {
        "BGC0001121 orsellinic": Program("acetyl", (Cycle(K), Cycle(K), Cycle(K)),
                                         Release.ALDOL_AROMATIC),
        "BGC0001275 6-MSA": Program("acetyl", (Cycle(K), Cycle(KR), Cycle(K)),
                                    Release.ALDOL_AROMATIC),
        "BGC0001244 mellein": Program("acetyl", (Cycle(KR), Cycle(K), Cycle(KR), Cycle(K)),
                                      Release.DIHYDROISOCOUMARIN),
        "BGC0001489 6-OH-mellein": Program("acetyl", (Cycle(KR), Cycle(K), Cycle(K), Cycle(K)),
                                           Release.DIHYDROISOCOUMARIN),
        "BGC0002240 BAB": Program("acetyl", (Cycle(DH), Cycle(KR), Cycle(DH), Cycle(DH), Cycle(DH)),
                                  Release.HYDROLYSIS),
    }
