"""Executor-level tests: cascade ordering, linear checkpoint, purity/determinism."""

from __future__ import annotations

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.executor import core


def test_linear_checkpoint_always_present_even_with_cyclization():
    prog = Program("acetyl", (Cycle(R.KETO),) * 3, Release.ALDOL_AROMATIC)
    res = core.run(prog)
    assert res.linear is not None
    # linear is the released keto-acid; final is the aromatic
    assert M.canonical_smiles(res.linear) == "CC(=O)CC(=O)CC(=O)CC(=O)O"
    assert M.canonical_smiles(res.final) == "Cc1cc(O)cc(O)c1C(=O)O"


def test_reduction_cascade_is_ordered():
    # ER state must pass through KR and DH: a fully reduced diketide is butyric acid.
    res = core.run(Program("acetyl", (Cycle(R.ER),), Release.HYDROLYSIS))
    assert M.canonical_smiles(res.linear) == "CCCC(=O)O"


def test_chain_length_bookkeeping():
    # N elongation cycles -> N+1 ketide units; fully reduced gives a 2(N+1)-carbon acid
    res = core.run(Program("acetyl", (Cycle(R.ER),) * 3, Release.HYDROLYSIS))
    assert M.canonical_smiles(res.linear) == "CCCCCCCC(=O)O"  # octanoic acid (C8)


def test_exec_is_pure_and_deterministic():
    prog = Program("acetyl", (Cycle(R.KETO), Cycle(R.DH), Cycle(R.KETO)), Release.ALDOL_AROMATIC)
    a = M.canonical_smiles(core.exec(prog))
    b = M.canonical_smiles(core.exec(prog))
    assert a == b == "Cc1cccc(O)c1C(=O)O"  # 6-MSA, stable across runs


def test_hydrolysis_release_final_equals_linear():
    res = core.run(Program("acetyl", (Cycle(R.KR),), Release.HYDROLYSIS))
    assert M.canonical_smiles(res.final) == M.canonical_smiles(res.linear)
