"""Verifier / program search: target molecule y -> verified consistent set Z*(y).

Implements Section 4.3: enumerate programs by beam search, execute each (partial)
program with the deterministic executor, prune any partial whose partial product cannot
reach y, and collect the verified set

    Z*(y) = { z : exec(z, O) ~= y }                                            (Eq. 4)

The executor is the oracle: a program is in Z*(y) iff its product matches y on the
constitutional graph (the Phase-0 executor is achiral; stereo is Phase 2).

Pruning. The released/cyclized product's carbon count equals the chain's carbon count
(cyclization removes only O/H). A program's chain carbons are
``2 (acetyl starter) + 2*n_cycles + n_cmet`` (malonyl extender). So:
  * a partial whose chain already has MORE carbons than the target is pruned (overshoot);
  * only programs whose chain carbons EQUAL the target's are executed at termination.
This, plus a small beam, keeps the search far below the naive 8^N (paper: the program
space is small enough to enumerate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rdkit import Chem

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Extender, Program, ReductionState, Release
from lpi.executor import core, operators as op

_ALL_REDUCTIONS = (
    ReductionState.KETO,
    ReductionState.KR,
    ReductionState.DH,
    ReductionState.ER,
)
_ALL_RELEASES = (
    Release.HYDROLYSIS,
    Release.LACTONIZATION,
    Release.ALDOL_AROMATIC,
    Release.DIHYDROISOCOUMARIN,
)

# Max whole waters a single-mode cyclization can plausibly expel (lactone + aromatic
# aromatization). Used only as a loose partial-pruning bound, not for matching.
_MAX_WATERS = 4


@dataclass
class OperatorSpec:
    """The operator set available to a BGC (Section 4.1 'admissible action set').

    For Phase-1 curated search this is set permissively so the search must *recover* the
    known program from many alternatives. In Phase 2 it is narrowed by the BGC's domain
    content (e.g. no C-MeT if no MeT domain; reductive states bounded by KR/DH/ER
    presence and trans-acting rescue).
    """

    starters: tuple[str, ...] = ("acetyl",)
    reductions: tuple[ReductionState, ...] = _ALL_REDUCTIONS
    allow_c_methyl: bool = True
    releases: tuple[Release, ...] = _ALL_RELEASES
    extenders: tuple[Extender, ...] = (Extender.MALONYL,)
    max_cycles: int | None = None  # default derived from the target carbon count


@dataclass
class SearchStats:
    target_smiles: str
    target_carbons: int
    n_programs_executed: int = 0
    n_pruned_overshoot: int = 0
    runtime_s: float = 0.0
    budget_exhausted: bool = False  # hit max_executions before finishing (scan only)


@dataclass
class SearchResult:
    target: str
    z_star: list[Program] = field(default_factory=list)
    stats: SearchStats | None = None

    @property
    def size(self) -> int:
        return len(self.z_star)


def _carbon_count(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


def _formula_cho(mol: Chem.Mol) -> tuple[int, int, int]:
    """(#C, #H including implicit, #O) of a molecule."""
    c = o = 0
    h = 0
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        h += a.GetTotalNumHs()
        if z == 6:
            c += 1
        elif z == 8:
            o += 1
        elif z == 1:
            h += 1
    return c, h, o


# Per-elongation-cycle contribution to the RELEASED LINEAR ACID formula (C, H, O),
# from the validated operator mass deltas (extension +C2H2O, KR +H2, DH -H2O, ER +H2,
# C-MeT +CH2). Acetyl starter released as acid = acetic acid C2H4O2 (the base).
_CYCLE_DELTA = {
    ReductionState.KETO: (2, 2, 1),
    ReductionState.KR: (2, 4, 1),
    ReductionState.DH: (2, 2, 0),
    ReductionState.ER: (2, 4, 0),
}


def linear_acid_formula(program: Program) -> tuple[int, int, int]:
    """Analytic (C, H, O) of the TE-hydrolysis (linear acid) product, no RDKit call.

    Lets the search prune executor calls: a cyclized product is the linear acid minus
    k whole waters (k>=0), so a program can match target y only if there is an integer
    k>=0 with C_lin==C_y, O_lin-k==O_y, H_lin-2k==H_y.
    """
    # base = acetic acid C2H4O2 for an acetyl starter; other starters offset by their
    # extra carbons/hydrogens (all saturated acyl starters: propionyl +CH2, butyryl +2CH2).
    base = {"acetyl": (2, 4, 2), "propionyl": (3, 6, 2), "butyryl": (4, 8, 2),
            "benzoyl": (7, 6, 2)}.get(program.starter, (2, 4, 2))
    c, h, o = base
    for cyc in program.cycles:
        dc, dh, do = _CYCLE_DELTA[cyc.reduction]
        c += dc
        h += dh
        o += do
        if cyc.c_methyl:
            c += 1
            h += 2
    return c, h, o


def formula_feasible(target: tuple[int, int, int], spec: OperatorSpec,
                     max_cycles: int, max_waters: int = 4) -> bool:
    """Necessary condition: does ANY program's product formula equal `target`?

    A program's released linear acid has, for a starter (cb,hb,ob) and n cycles with
    n_oxy KETO/KR cycles, n_highH KR/ER cycles, n_me C-MeT methyls:
        C = cb + 2n + n_me
        H = hb + 2n + 2*n_highH + 2*n_me
        O = ob + n_oxy
    Cyclization loses k whole waters (k=0 for hydrolysis). We check for integer solutions
    with 0<=n_me,n_oxy,n_highH<=n and 0<=k<=max_waters. Infeasible => provably unreachable
    (skip the structural search); feasible => still must be confirmed structurally.
    """
    ct, ht, ot = target
    _BASE = {"acetyl": (2, 4, 2), "propionyl": (3, 6, 2), "butyryl": (4, 8, 2),
             "benzoyl": (7, 6, 2)}
    allow_me = spec.allow_c_methyl
    for starter in spec.starters:
        cb, hb, ob = _BASE.get(starter, (2, 4, 2))
        for n in range(1, max_cycles + 1):
            n_me = ct - cb - 2 * n
            if n_me < 0 or n_me > n or (n_me > 0 and not allow_me):
                continue
            for k in range(0, max_waters + 1):
                n_oxy = ot - ob + k
                if n_oxy < 0 or n_oxy > n:
                    continue
                two_high = ht - hb - 2 * n - 2 * n_me + 2 * k
                if two_high < 0 or two_high % 2:
                    continue
                n_high = two_high // 2
                if 0 <= n_high <= n:
                    return True
    return False


def _formula_consistent(linear: tuple[int, int, int], target: tuple[int, int, int],
                        release: Release) -> bool:
    """Could `program` (with this release) yield `target`? Cheap necessary condition."""
    cl, hl, ol = linear
    ct, ht, ot = target
    if cl != ct:
        return False
    if release in (Release.HYDROLYSIS, Release.NONE):
        return hl == ht and ol == ot  # no water lost
    k = ol - ot  # waters lost on cyclization
    return k >= 0 and (hl - 2 * k) == ht


def _chain_carbons(program: Program) -> int:
    # acetyl starter = 2 C; each malonyl cycle +2 C; each C-MeT +1 C.
    n = 2 if program.starter == "acetyl" else _carbon_count(op.load_starter(program.starter)) - 0
    c = n
    for cyc in program.cycles:
        c += 3 if cyc.extender is Extender.METHYLMALONYL else 2
        if cyc.c_methyl:
            c += 1
    return c


def _cycle_options(spec: OperatorSpec) -> list[Cycle]:
    opts: list[Cycle] = []
    methyls = (False, True) if spec.allow_c_methyl else (False,)
    for ext in spec.extenders:
        for red in spec.reductions:
            for me in methyls:
                opts.append(Cycle(reduction=red, c_methyl=me, extender=ext))
    return opts


def search(target_smiles: str, spec: OperatorSpec | None = None,
           beam_width: int = 2000, max_executions: int | None = None) -> SearchResult:
    """Beam-search programs and return the verified consistent set Z*(y).

    ``max_executions`` (optional) caps the number of executor calls so a reachability
    scan over many targets cannot hang; if hit, ``stats.budget_exhausted`` is set and the
    returned Z*(y) is a lower bound (search was incomplete).
    """
    spec = spec or OperatorSpec()
    t0 = time.perf_counter()
    target = M.mol_from_smiles(target_smiles)
    target_c = _carbon_count(target)
    target_formula = _formula_cho(target)
    target_flat = M.canonical_smiles(M.strip_stereo(target))
    max_cycles = spec.max_cycles if spec.max_cycles is not None else max(1, target_c // 2)

    stats = SearchStats(target_smiles=target_smiles, target_carbons=target_c)
    z_star: list[Program] = []
    cycle_opts = _cycle_options(spec)

    def try_release(partial_cycles: tuple[Cycle, ...], starter: str) -> None:
        base_prog = Program(starter=starter, cycles=partial_cycles)
        if _chain_carbons(base_prog) != target_c:
            return  # final carbon count must equal target
        lin_formula = linear_acid_formula(base_prog)
        for rel in spec.releases:
            if not _formula_consistent(lin_formula, target_formula, rel):
                continue  # formula prefilter: skip executor call when y is unreachable
            prog = Program(starter=starter, cycles=partial_cycles, release=rel)
            if max_executions is not None and stats.n_programs_executed >= max_executions:
                stats.budget_exhausted = True
                return
            try:
                res = core.run(prog)
            except Exception:  # noqa: BLE001
                continue
            stats.n_programs_executed += 1
            # Use the product of the DECLARED release only. For non-hydrolysis releases,
            # res.final is None when the cyclization did not fire -- such a program did
            # not actually produce that structure, so it must NOT fall back to the linear
            # acid (that would let every failed cyclization spuriously match and inflate
            # |Z*(y)|). HYDROLYSIS sets res.final == res.linear already.
            product = res.final
            if product is None:
                continue
            if M.canonical_smiles(M.strip_stereo(product)) == target_flat:
                z_star.append(prog)

    # Beam over cycle depth. A "beam state" is (starter, cycles-so-far).
    for starter in spec.starters:
        beam: list[tuple[str, tuple[Cycle, ...]]] = [(starter, ())]
        # depth 0 release (e.g. just the starter) is possible but rarely matches; include.
        try_release((), starter)
        for _depth in range(max_cycles):
            if stats.budget_exhausted:
                break
            next_beam: list[tuple[str, tuple[Cycle, ...]]] = []
            for st, cycles in beam:
                for cyc in cycle_opts:
                    new_cycles = cycles + (cyc,)
                    prog_partial = Program(starter=st, cycles=new_cycles)
                    cc = _chain_carbons(prog_partial)
                    if cc > target_c:
                        stats.n_pruned_overshoot += 1
                        continue  # carbon overshoot -> cannot reach target
                    # Oxygen prune: the linear-acid O of a partial only grows with future
                    # cycles, and release removes at most MAX_WATERS; so if even the best
                    # case (no further O added, max water lost) overshoots target O, prune.
                    if linear_acid_formula(prog_partial)[2] - _MAX_WATERS > target_formula[2]:
                        stats.n_pruned_overshoot += 1
                        continue
                    try_release(new_cycles, st)
                    next_beam.append((st, new_cycles))
            # Beam pruning: prefer partials closest to the target carbon count (those that
            # can still terminate soon). Deterministic tie-break by program repr.
            next_beam.sort(key=lambda s: (target_c - _chain_carbons(Program(s[0], s[1])),
                                          repr(s[1])))
            beam = next_beam[:beam_width]
            if not beam:
                break

    stats.runtime_s = time.perf_counter() - t0
    # Deterministic, de-duplicated ordering of Z*(y).
    uniq = {repr(p): p for p in z_star}
    result = SearchResult(target=target_smiles,
                          z_star=[uniq[k] for k in sorted(uniq)], stats=stats)
    return result
