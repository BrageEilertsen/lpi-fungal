"""Realizability metric: design-by-grammar-inversion edit distance over the program manifold."""
from __future__ import annotations

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.realizability import Tier, cost, edits_from, natural_manifold, realize

_MAN = natural_manifold()
_6MSA = _MAN["BGC0001275 6-MSA"]          # acetyl + K,KR,K -> aldol aromatic
_ORS = _MAN["BGC0001121 orsellinic"]      # acetyl + K,K,K -> aldol aromatic


def test_natural_program_is_distance_zero():
    r = realize(_6MSA, _MAN)
    assert r.cost == 0 and r.n_edits == 0 and r.verdict == "natural"
    assert r.nearest_id == "BGC0001275 6-MSA"


def test_single_reduction_flip_is_one_documented_edit():
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert r.nearest_id == "BGC0001275 6-MSA"     # nearest is 6-MSA (one cycle differs)
    assert r.cost == 1 and r.n_edits == 1
    assert r.edits[0].kind == "reduction" and r.edits[0].tier is Tier.DOCUMENTED
    assert r.verdict == "engineerable"


def test_release_reprogramming_is_speculative():
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), Release.LACTONIZATION)
    r = realize(target, _MAN)
    assert any(e.kind == "release" and e.tier is Tier.SPECULATIVE for e in r.edits)
    assert r.verdict == "speculative"


def test_many_edits_even_if_documented_is_not_engineerable():
    # 5 reduction flips from 6-MSA's nearest neighbour -> too far to call engineerable
    target = Program("acetyl", (Cycle(R.ER), Cycle(R.ER), Cycle(R.ER), Cycle(R.ER), Cycle(R.ER)),
                     Release.HYDROLYSIS)
    r = realize(target, _MAN)
    assert r.verdict == "speculative"             # documented edit classes, but too many


def test_edit_tiers_and_cost():
    # orsellinic + one C-MeT (tier 2) -> cost 2, engineerable
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KETO, c_methyl=True), Cycle(R.KETO)),
                     Release.ALDOL_AROMATIC)
    es = edits_from(target, _ORS)
    assert len(es) == 1 and es[0].kind == "c_methyl" and es[0].tier is Tier.PLAUSIBLE
    assert cost(es) == 2
    assert realize(target, _MAN).verdict == "engineerable"


def test_nearest_template_is_chosen():
    # a 4-cycle reduced program is closest to mellein/6-OH-mellein (4-cycle), not 6-MSA (3-cycle)
    target = Program("acetyl", (Cycle(R.KR), Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)),
                     Release.DIHYDROISOCOUMARIN)
    r = realize(target, _MAN)
    assert r.nearest_id == "BGC0001244 mellein" and r.cost == 0   # this *is* mellein's program
