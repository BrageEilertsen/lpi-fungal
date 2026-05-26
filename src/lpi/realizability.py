"""Realizability: design-by-grammar-inversion with a structural/control axis decomposition.

The executor runs forward (program -> structure) and the verifier runs inverse (structure ->
programs). This module adds *design*: given a target program, how far is it from something nature
already builds, and -- crucially -- *along which axis*?

Two edit axes, because they map onto fundamentally different wet-lab capabilities:

* STRUCTURAL edits change the **domain content** -- adding a reductive domain, swapping the
  loading/AT starter or extender, reprogramming the release cyclase. These transfer from bacterial
  modular-PKS engineering precedent (domain swaps, module deletion) and are comparatively documented.
* CONTROL edits change the **iteration program** with the domain set fixed -- which cycle a present
  domain fires on (the per-cycle reduction pattern) or the iteration count. This is the
  iteration-grammar frontier: exactly the 100%->57% wall of the differential probe, now as a design
  dimension. No other tool can even formulate it, because none has the iteration-grammar machinery.

A design at "(2 structural, 1 control)" is a different proposition than "(3 structural, 0 control)":
the first needs iteration-program editing (frontier protein engineering), the second is standard.

Cost basis (curated 2026-05-26; see data/curation/edit_tiers.csv + edit_precedent.md): per-edit tiers
are grounded in real PKS-engineering precedent. Structural edits with bacterial modular-PKS precedent
-- AT/extender swap, loading/starter swap, reductive-loop add -- are DOCUMENTED; de-novo C-MeT
insertion (add_cmt) and cyclization-mode reprogramming (release) are harder. CONTROL edits are the
iteration-grammar frontier (Cox 2023, Nat. Prod. Rep. 40:9-27): a single edit near a real template is
achievable (Fisch & Cox 2011, JACS 133:16635) but stacking them is "extremely difficult" because the
programme is emergent and non-separable -- so control cost is SUPER-ADDITIVE (quadratic in the number
of control edits; see ``cost``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from lpi.chem.program import Cycle, Extender, Program, ReductionState as R, Release


class Axis(Enum):
    STRUCTURAL = "structural"   # domain content -- bacterial-precedent-transferable
    CONTROL = "control"         # iteration program -- the iteration-grammar frontier


class Tier(IntEnum):
    DOCUMENTED = 1
    PLAUSIBLE = 2
    SPECULATIVE = 3


# reductive domains required to reach each beta-state.
_RED_DOMAINS: dict[R, frozenset[str]] = {
    R.KETO: frozenset(),
    R.KR: frozenset({"KR"}),
    R.DH: frozenset({"KR", "DH"}),
    R.ER: frozenset({"KR", "DH", "ER"}),
}

ENGINEERABLE_MAX_EDITS = 3


@dataclass(frozen=True)
class Edit:
    kind: str
    detail: str
    tier: Tier
    axis: Axis


@dataclass(frozen=True)
class Template:
    """A natural cluster: its program plus the catalytic domain set it carries (sets the axis of an
    edit -- a reduction within the domain capability is control, one needing a new domain is structural)."""

    id: str
    program: Program
    domains: frozenset[str]


@dataclass
class Realizability:
    target: Program
    nearest_id: str
    nearest: Template
    edits: list[Edit]
    verdict: str        # natural | engineerable | frontier | speculative

    @property
    def structural_edits(self) -> list[Edit]:
        return [e for e in self.edits if e.axis is Axis.STRUCTURAL]

    @property
    def control_edits(self) -> list[Edit]:
        return [e for e in self.edits if e.axis is Axis.CONTROL]

    @property
    def structural_cost(self) -> int:
        return len(self.structural_edits)

    @property
    def control_cost(self) -> int:
        return len(self.control_edits)


def _norm(domains) -> frozenset[str]:
    return frozenset(str(d).upper() for d in domains)


def _target_domains(prog: Program) -> frozenset[str]:
    """The catalytic (reductive + C-MeT) domains a program requires."""
    d: set[str] = set()
    for c in prog.cycles:
        d |= _RED_DOMAINS[c.reduction]
        if c.c_methyl:
            d.add("CMT")
    return frozenset(d)


def cost(edits: list[Edit]) -> int:
    """Total realizability cost. Structural edits sum linearly by tier (independent domain-content
    swaps with bacterial precedent). CONTROL edits are SUPER-ADDITIVE -- quadratic in their count --
    because the iteration programme is an emergent, non-separable function of all per-cycle decisions
    (Cox 2023): one control edit near a real template is achievable (Fisch & Cox 2011) but stacked
    control edits scale toward the iteration-program wall. The form ``(sum of control tiers) * (number
    of control edits)`` reduces to the linear tier cost at k=1 and grows as ~k^2 for k>1."""
    structural = sum(int(e.tier) for e in edits if e.axis is Axis.STRUCTURAL)
    control_edits = [e for e in edits if e.axis is Axis.CONTROL]
    control = sum(int(e.tier) for e in control_edits) * len(control_edits)
    return structural + control


def edits_from(target: Program, template: Template) -> list[Edit]:
    """Typed edits transforming ``template`` into ``target``, each tagged with axis + tier
    (positional v1: cycle-by-cycle to the shorter program, plus length and starter/release deltas)."""
    dom = _norm(template.domains)
    tp = template.program
    out: list[Edit] = []
    # STRUCTURAL: domain-content delta -- catalytic domains the target needs that the template lacks.
    # (Adding a domain is the structural edit; programming when it fires is a control edit, below.)
    for d in sorted(_target_domains(target) - dom):
        if d == "CMT":   # de-novo C-MeT insertion into a methylation-free synthase: no precedent
            out.append(Edit("add_cmt", "+CMT", Tier.SPECULATIVE, Axis.STRUCTURAL))
        else:            # reductive-loop add (KR/DH/ER): mature modular-PKS precedent
            out.append(Edit("add_reductive", f"+{d}", Tier.DOCUMENTED, Axis.STRUCTURAL))
    if target.starter != tp.starter:
        out.append(Edit("starter", f"{tp.starter}->{target.starter}", Tier.DOCUMENTED, Axis.STRUCTURAL))
    if {c.extender for c in target.cycles} != {c.extender for c in tp.cycles}:
        out.append(Edit("extender", "AT extender swap", Tier.DOCUMENTED, Axis.STRUCTURAL))
    if target.release != tp.release:   # TE/cyclase swap -- structurally a domain swap (bacterial canon)
        out.append(Edit("release", f"{tp.release.value}->{target.release.value}",
                        Tier.PLAUSIBLE, Axis.STRUCTURAL))
    # CONTROL: the iteration program given the available domains -- which cycle each domain fires on
    # (per-cycle reduction / C-MeT) and the iteration count. This is the iteration-grammar frontier.
    nt, nm = len(target.cycles), len(tp.cycles)
    for i in range(min(nt, nm)):
        ct, cm = target.cycles[i], tp.cycles[i]
        if ct.reduction != cm.reduction:
            out.append(Edit("reduction", f"cyc{i + 1} {cm.reduction.value}->{ct.reduction.value}",
                            Tier.PLAUSIBLE, Axis.CONTROL))
        if ct.c_methyl != cm.c_methyl:
            out.append(Edit("c_methyl", f"cyc{i + 1} {'+' if ct.c_methyl else '-'}C-MeT",
                            Tier.PLAUSIBLE, Axis.CONTROL))
    for i in range(min(nt, nm), max(nt, nm)):
        if nt > nm:      # adding an iteration: per-cycle reprogramming, plausible near a template
            out.append(Edit("cycle_add", f"+cyc{i + 1}", Tier.PLAUSIBLE, Axis.CONTROL))
        else:            # removing an iteration changes the program's termination count -- speculative
            out.append(Edit("cycle_remove", f"-cyc{i + 1}", Tier.SPECULATIVE, Axis.CONTROL))
    return out


def _verdict(edits: list[Edit]) -> str:
    if not edits:
        return "natural"
    if any(e.tier is Tier.SPECULATIVE for e in edits):   # release reprogramming
        return "speculative"
    if any(e.axis is Axis.CONTROL for e in edits):        # needs iteration-program editing
        return "frontier"
    if len(edits) <= ENGINEERABLE_MAX_EDITS:              # documented domain-content edits only
        return "engineerable"
    return "speculative"


def realize(target: Program, manifold: list[Template]) -> Realizability:
    """Nearest natural cluster + edit path, ranked by total cost -- the most realizable route
    (control edits stack super-additively; see ``cost``)."""
    best: tuple[int, Template, list[Edit]] | None = None
    for tmpl in manifold:
        es = edits_from(target, tmpl)
        c = cost(es)
        if best is None or c < best[0]:
            best = (c, tmpl, es)
    _, tmpl, es = best  # type: ignore[misc]
    return Realizability(target, tmpl.id, tmpl, es, _verdict(es))


# ---- the natural-cluster manifold (real reachable fungal PKS programs + domain architectures) ----
def natural_manifold() -> list[Template]:
    K, KR, DH = R.KETO, R.KR, R.DH
    nr = frozenset({"KS", "AT", "ACP"})                       # non-reducing
    pr = frozenset({"KS", "AT", "DH", "KR", "ACP"})           # partially reducing
    hr = frozenset({"KS", "AT", "DH", "KR", "ER", "ACP"})     # highly reducing
    return [
        Template("BGC0001121 orsellinic", Program("acetyl", (Cycle(K), Cycle(K), Cycle(K)),
                                                  Release.ALDOL_AROMATIC), nr),
        Template("BGC0001275 6-MSA", Program("acetyl", (Cycle(K), Cycle(KR), Cycle(K)),
                                             Release.ALDOL_AROMATIC), pr),
        Template("BGC0001244 mellein", Program("acetyl", (Cycle(KR), Cycle(K), Cycle(KR), Cycle(K)),
                                               Release.DIHYDROISOCOUMARIN), pr),
        Template("BGC0001489 6-OH-mellein", Program("acetyl", (Cycle(KR), Cycle(K), Cycle(K), Cycle(K)),
                                                    Release.DIHYDROISOCOUMARIN), pr),
        Template("BGC0002240 BAB", Program("acetyl", (Cycle(DH), Cycle(KR), Cycle(DH), Cycle(DH),
                                                      Cycle(DH)), Release.HYDROLYSIS), hr),
    ]


# ---- the design neighbourhood: 1-edit neighbours of a template, by axis -----------
_STARTERS = ("acetyl", "propionyl", "butyryl", "hexanoyl")
_RELEASES = (Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
             Release.DIHYDROISOCOUMARIN)


def one_edit_neighbours(template: Template) -> set[Program]:
    """Every program one typed edit from the template (both axes), for design-space enumeration."""
    p, dom = template.program, _norm(template.domains)
    cyc = list(p.cycles)
    out: set[Program] = set()
    states = [s for s in (R.KETO, R.KR, R.DH, R.ER)]
    for i in range(len(cyc)):
        for s in states:                                   # reduction edits (control or structural)
            if s != cyc[i].reduction:
                nc = cyc.copy(); nc[i] = Cycle(s, cyc[i].c_methyl, cyc[i].extender)
                out.add(Program(p.starter, tuple(nc), p.release))
        nc = cyc.copy(); nc[i] = Cycle(cyc[i].reduction, not cyc[i].c_methyl, cyc[i].extender)
        out.add(Program(p.starter, tuple(nc), p.release))   # C-MeT toggle
        other_ext = Extender.METHYLMALONYL if cyc[i].extender is Extender.MALONYL else Extender.MALONYL
        nc = cyc.copy(); nc[i] = Cycle(cyc[i].reduction, cyc[i].c_methyl, other_ext)
        out.add(Program(p.starter, tuple(nc), p.release))   # extender swap (structural)
    for s in states:                                        # iteration-count change (control)
        out.add(Program(p.starter, tuple(cyc + [Cycle(s)]), p.release))
    if len(cyc) > 1:
        out.add(Program(p.starter, tuple(cyc[:-1]), p.release))
    for st in _STARTERS:                                    # starter swap (structural)
        if st != p.starter:
            out.add(Program(st, p.cycles, p.release))
    for rel in _RELEASES:                                   # release reprogram (structural, speculative)
        if rel != p.release:
            out.add(Program(p.starter, p.cycles, rel))
    out.discard(p)
    return out


def design_space(manifold: list[Template]) -> dict:
    """Enumerate the 1-edit design neighbourhood of the whole manifold; classify each distinct design
    by its most-realizable route's axis breakdown (the decisive structural-vs-control count)."""
    designs: dict[str, Realizability] = {}
    natural = {repr(t.program) for t in manifold}
    for t in manifold:
        for prog in one_edit_neighbours(t):
            if repr(prog) in natural:
                continue
            r = realize(prog, manifold)
            designs[repr(prog)] = r
    engineerable = [r for r in designs.values() if r.verdict == "engineerable"]
    frontier = [r for r in designs.values() if r.verdict == "frontier"]
    speculative = [r for r in designs.values() if r.verdict == "speculative"]
    return {
        "n_designs": len(designs),
        "engineerable": engineerable,      # structural-only, documented PKS engineering
        "frontier": frontier,              # involve a control edit -- iteration-program editing
        "speculative": speculative,        # release reprogram / too far
    }


def distinct_products(realizations: list[Realizability]) -> int:
    """Distinct constitutional product structures among a design set -- the honest diversity count,
    since different edit-specs can collapse to the same molecule (esp. release reprogramming)."""
    from lpi.chem import mol as M
    from lpi.executor import core
    prods: set[str] = set()
    for r in realizations:
        try:
            prods.add(M.canonical_smiles(M.strip_stereo(core.exec(r.target))))
        except Exception:  # noqa: BLE001 - an edited program whose cyclization cannot fire
            pass
    return len(prods)
