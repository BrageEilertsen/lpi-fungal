"""BGC<->peak matching demo: a genome's clusters x a metabolome's peaks -> ranked discoveries.

The metabologenomics step that makes the engine a discovery tool rather than a one-cluster verifier.
We take four real reachable fungal PKS clusters (literature domain architectures; MIBiG annotates
them only as class "PKS") and a "metabolome" of their real monoisotopic masses (as [M-H]- ions) plus
a decoy peak, then run the shared engine over every cluster x peak pair to build a scored assignment
graph. For each peak we report the honest discovery call: the verified structure(s) and the
cluster(s) tied at the top -- separating *what was made* (structure) from *which cluster made it*
(attribution, which is degenerate when clusters share a domain alphabet).
"""
from __future__ import annotations

import json
import os

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.grammars import PKS
from lpi.matching import Cluster, Peak, match
from lpi.observe import ADDUCTS, exact_mass, ion_mz

RDLogger.DisableLog("rdApp.*")

_MIBIG = "data/raw/mibig_json_4.0"

# (accession, name, literature domain architecture, chain band)
CLUSTERS = [
    ("BGC0001121", "orsellinic acid synthase (NR)", {"KS", "AT", "ACP"}, (2, 5)),
    ("BGC0001275", "6-MSA synthase (PR)", {"KS", "AT", "DH", "KR", "ACP"}, (2, 6)),
    ("BGC0001244", "mellein synthase (PR)", {"KS", "AT", "DH", "KR", "ACP"}, (2, 6)),
    ("BGC0001489", "6-hydroxymellein synthase (PR)", {"KS", "AT", "DH", "KR", "ACP"}, (2, 6)),
]


def _product_smiles(bgc: str) -> str | None:
    path = os.path.join(_MIBIG, f"{bgc}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    cps = d.get("compounds", [])
    return cps[0].get("structure") if cps else None


def _flat(smiles: str) -> str:
    return M.canonical_smiles(M.strip_stereo(M.mol_from_smiles(smiles)))


def main() -> None:
    print("BGC<->peak matching: 4 real fungal clusters x a metabolome (their masses + 1 decoy)\n")
    clusters, truth = [], {}
    peaks = []
    for bgc, name, domains, (lo, hi) in CLUSTERS:
        clusters.append(Cluster(bgc, domains, PKS, lo, hi, name))
        smi = _product_smiles(bgc)
        if smi is None:
            print(f"  (skip {bgc}: no local MIBiG record)")
            continue
        mz = ion_mz(exact_mass(smi), ADDUCTS["[M-H]-"])
        pid = f"peak@{mz:.4f}"
        peaks.append(Peak(pid, mz, ("[M-H]-",), 5.0, name=name.split(" synthase")[0]))
        truth[pid] = (bgc, _flat(smi))
    # decoy: a mass no small C/H/O polyketide reaches
    peaks.append(Peak("decoy@322.1", ion_mz(323.1234, ADDUCTS["[M-H]-"]), ("[M-H]-",), 5.0, name="decoy"))

    res = match(clusters, peaks)
    calls = res.discovery_calls()
    print(f"{'peak':16s} {'verdict':14s} {'struct?':8s} attribution")
    print("-" * 78)
    for p in peaks:
        c = calls.get(p.id)
        if c is None:
            print(f"{p.id:16s} {'UNEXPLAINED':14s} {'-':8s} no cluster under current grammars")
            continue
        tb, tflat = truth.get(p.id, (None, None))
        struct_ok = "YES" if tflat and tflat in {_flat(s) for s in c["structures"]} else "no"
        attrib = ("AMBIGUOUS: " + ",".join(c["clusters"])) if c["cluster_ambiguous"] else c["clusters"][0]
        print(f"{p.id:16s} {c['state'].value:14s} {struct_ok:8s} {attrib}")
    print(f"\nunexplained peaks: {res.unexplained_peaks or 'none'}")
    print(f"total scored edges: {len(res.edges)}")
    print("\nReading: the structure is recovered per peak (struct?=YES); where PR clusters share the\n"
          "{KS,AT,DH,KR,ACP} alphabet, mass alone cannot say which produced it (AMBIGUOUS) -- the\n"
          "signal that cluster attribution needs sequence specificity, MS/MS, or expression.")


if __name__ == "__main__":
    main()
