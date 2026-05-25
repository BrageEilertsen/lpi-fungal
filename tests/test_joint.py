"""Joint (z_pks, z_tail) verification tests + gene/Delta budget."""

from __future__ import annotations

from lpi.data.genebudget import family_of
from lpi.search.delta_formula import delta, explain_delta
from lpi.search.joint import explain_cluster
from lpi.chem import mol as M


def test_gene_family_mapping():
    assert family_of("cytochrome P450 monooxygenase") == "oxygenase"
    assert family_of("O-methyltransferase") == "methyltransferase"
    assert family_of("flavin-dependent halogenase") == "halogenase"
    assert family_of("polyketide synthase") is None  # the synthase, not tailoring


def test_delta_needs_enzyme_in_budget():
    # +O requires an oxygenase; absent -> no explanation
    assert explain_delta({"O": 1}, {"methyltransferase": 1}) == []
    assert explain_delta({"O": 1}, {"oxygenase": 1})


def test_nitrogen_delta_rejected():
    # N in the delta = NRPS fusion, not a tailoring edit
    assert explain_delta({"N": 1, "C": 5, "H": 9, "O": 1}, {"oxygenase": 3}) == []


def test_untailored_verifies_zero_edits():
    r = explain_cluster("Cc1cc(O)cc(O)c1C(=O)O", {"oxygenase": 2})  # orsellinic
    assert r.verified and r.min_edits == 0


def test_halogenation_one_edit():
    r = explain_cluster("Cc1ccc(Cl)c(O)c1C(=O)O", {"halogenase": 1})  # chloro-6-MSA
    assert r.verified and r.min_edits == 1
    assert "halogenation_cl" in r.explanations[0].edit_counts


def test_methylation_one_edit():
    r = explain_cluster("COc1cc(C)c(C(=O)O)c(O)c1", {"methyltransferase": 1})  # 4-OMe orsellinic
    assert r.verified and r.min_edits == 1


def test_nrps_hybrid_not_verified():
    # an N-containing (amino-acid-appended) product cannot be a pure PKS+tailoring product
    r = explain_cluster("CC(C)CC(NC(=O)c1ccccc1)C(=O)O", {"oxygenase": 2})
    assert not r.verified
