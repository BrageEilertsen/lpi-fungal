"""BGC<->peak matching (the metabologenomics layer): structure recovery + honest attribution.

Locks the core machine behavior: over many clusters x many peaks, the matcher (a) recovers the
correct structure per real peak, (b) leaves a decoy peak unexplained, and (c) flags cluster
attribution as AMBIGUOUS when grammar-degenerate clusters tie -- mass gives the molecule, not the
producing cluster.
"""
from __future__ import annotations

from lpi.chem import mol as M
from lpi.engine import State
from lpi.grammars import PKS
from lpi.matching import Attribution, Cluster, Peak, match
from lpi.observe import ADDUCTS, exact_mass, ion_mz

_ORS = "Cc1cc(O)cc(O)c1C(=O)O"          # orsellinic acid, C8H8O4 (non-reducing)
_MSA = "Cc1cccc(O)c1C(=O)O"             # 6-methylsalicylic acid, C8H8O3 (partially reducing)
_MEL = "CC1Cc2cccc(O)c2C(=O)O1"         # mellein, C10H10O3 (partially reducing)


def _flat(smiles: str) -> str:
    return M.canonical_smiles(M.strip_stereo(M.mol_from_smiles(smiles)))


def _peak(smiles: str, pid: str) -> Peak:
    return Peak(pid, ion_mz(exact_mass(smiles), ADDUCTS["[M-H]-"]), ("[M-H]-",), 5.0)


def _setup():
    clusters = [
        Cluster("NR", {"KS", "AT", "ACP"}, PKS, 2, 5, "orsellinic (NR)"),
        Cluster("PR1", {"KS", "AT", "DH", "KR", "ACP"}, PKS, 2, 6, "6-MSA (PR)"),
        Cluster("PR2", {"KS", "AT", "DH", "KR", "ACP"}, PKS, 2, 6, "mellein (PR)"),
    ]
    peaks = [
        _peak(_ORS, "ors"), _peak(_MSA, "msa"), _peak(_MEL, "mel"),
        Peak("decoy", ion_mz(323.1234, ADDUCTS["[M-H]-"]), ("[M-H]-",), 5.0),
    ]
    return clusters, peaks


def test_decoy_peak_is_unexplained():
    clusters, peaks = _setup()
    res = match(clusters, peaks)
    assert "decoy" in res.unexplained_peaks


def test_real_peaks_recover_the_correct_structure():
    clusters, peaks = _setup()
    calls = match(clusters, peaks).peak_calls()
    for pid, smi in [("ors", _ORS), ("msa", _MSA), ("mel", _MEL)]:
        assert calls[pid].structure_verdict is State.VERIFIED
        assert _flat(smi) in {_flat(s) for s in calls[pid].structures}


def test_structure_and_attribution_are_separate_axes():
    clusters, peaks = _setup()
    calls = match(clusters, peaks).peak_calls()
    msa = calls["msa"]
    # 6-MSA: the molecule is pinned (structure VERIFIED) but the cluster is not (both PR clusters
    # share {KS,AT,DH,KR,ACP}) -- the two axes diverge, which is the whole point of separating them
    assert msa.structure_verdict is State.VERIFIED
    assert msa.attribution is Attribution.AMBIGUOUS
    assert {"PR1", "PR2"} <= set(msa.clusters)
    # the non-reducing cluster cannot make 6-MSA (needs a ketoreduction) -> not a candidate producer
    assert "NR" not in msa.clusters


def test_match_graph_has_no_edges_to_unreachable_peak():
    clusters, peaks = _setup()
    res = match(clusters, peaks)
    assert all(e.peak_id != "decoy" for e in res.edges)
