"""Lanthipeptide (RiPP) grammar -- SCAFFOLD for the pre-registered cross-family crossing (paper Sec. 16.14).

What is sound here (asserts no chemistry):
  * precursor parsing (core peptide -> Ser/Thr dehydration sites, Cys bridge donors);
  * the FREED topology DOF -- every injective Cys->Dha pairing, permissive, no ring-size prior
    (Remark 16.14a), so #pairings is a conservative kappa_mass UPPER BOUND and the true topology
    is guaranteed inside the enumerated set (the soundness gate can fire when renderers arrive);
  * the mass bookkeeping: dehydration Delta = -H2O each; thia-Michael lanthionine bridge is
    mass-NEUTRAL (no leaving group) -> every topology of a core shares ONE formula, so
    kappa_mass = #pairings exactly. Topology-U is automatic for any >=2-ring core.

What is STUBBED (the gated 20%): the structural renderers that build modified-peptide SMILES
(Ser->Dha, lanthionine bridge, leader cleavage). They raise NotImplementedError, typed against
the mass deltas above, and wait on chemist-signed SMARTS + an INDEPENDENT deposited ATOMIC structure
clearing the ~=-match gate -- NOT a connectivity anchor (sequence + bridge table is the engine's own
input, so matching a render against it is a mirror that certifies nothing; cf. the PT-naphthalene
"skeleton-correct, flagged" precedent). This module renders NOTHING and claims no citable datapoint;
it computes the sound combinatorics and frames the partition.

Pre-registered prediction (Sec. 16.14): core-residue identity -> genome-modality (R, engineerable
by codon swap); ring topology -> (m_ctrl, m_conf) = (trajectory, product): mass/genome-invisible
(kappa_mass = #pairings, computed here), product-resolvable (MS/MS across rings -- PENDING renderers).
NOT the fungal (trajectory, trajectory) cell. Falsifiers in Sec. 16.14.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from lpi.search.generate import Candidate, GenResult  # interface currency (re-exported by base)

__all__ = ["LanthiAlphabet", "LanthiProgram", "LanthiGrammar", "LANTHI", "partition"]

H2O = 18.010565          # dehydration mass loss; lanthionine bridge is mass-neutral (Delta = 0)
DEHYDRATABLE = {"S", "T"}  # Ser -> Dha, Thr -> Dhb
BRIDGE_DONOR = "C"         # Cys thiol (thia-Michael donor)


@dataclass(frozen=True)
class LanthiAlphabet:
    """Genome-mining input for a lanthipeptide cluster: the gene-encoded core peptide (1-letter
    sequence) and the dehydratase's target positions. LanB/LanC or LanM domain content only
    *confirms* the family; the CORE SEQUENCE is the alphabet (the RiPP analog of per-module sequence)."""

    core: str
    dehydrated: tuple[int, ...]            # 0-based positions of Ser/Thr -> Dha/Dhb

    @property
    def cys(self) -> tuple[int, ...]:
        return tuple(i for i, a in enumerate(self.core) if a == BRIDGE_DONOR)


@dataclass(frozen=True)
class LanthiProgram:
    """Provenance for one lanthipeptide program: dehydration set + the Cys->Dha bridge assignment
    (the freed topology DOF). Distinct pairings are distinct ring topologies / distinct molecules."""

    core: str
    dehydrated: tuple[int, ...]
    pairing: tuple[tuple[int, int], ...]   # (cys_pos, dha_pos) thioether bonds


def _pairings(cys: tuple[int, ...], dha: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    """Permissive topology enumeration (Remark 16.14a): every injective Cys->Dha assignment
    (mature fully-bridged form), no ring-size prior. ~=-gated downstream; ring-size legality only
    SHRINKS this set, so len(_pairings) is a conservative kappa_mass upper bound with the true
    topology guaranteed inside it. Requires len(dha) >= len(cys) to bridge every Cys."""
    n = len(cys)
    if n == 0 or len(dha) < n:
        return [] if n else [()]
    seen = {tuple(sorted(zip(cys, dsel))) for dsel in itertools.permutations(dha, n)}
    return sorted(seen)


# ---- structural renderers: STUBS, typed against the mass deltas, flagged for chemist sign-off ----
def _render_dehydration(*_a: Any, **_k: Any):
    raise NotImplementedError(
        "dehydration renderer pending chemist-signed SMARTS (Ser->Dha / Thr->Dhb, Delta=-H2O); "
        "scaffold asserts no lanthipeptide chemistry")


def _render_lanthionine_bridge(*_a: Any, **_k: Any):
    raise NotImplementedError(
        "lanthionine-bridge renderer pending chemist-signed SMARTS (Cys-S across Dha, thia-Michael, "
        "mass-NEUTRAL Delta=0); scaffold asserts no lanthipeptide chemistry")


def _render_leader_cleavage(*_a: Any, **_k: Any):
    raise NotImplementedError("leader-cleavage renderer pending chemist-signed cut site")


class LanthiGrammar:
    """Lanthipeptide grammar -- SCAFFOLD. Sound DOF enumeration + mass bookkeeping; structural
    execution stubbed (raises, pending chemist-signed renderers). Implements the Grammar protocol
    so it plugs into ``engine.infer_cluster`` unchanged the moment the renderers drop in."""

    name = "lanthipeptide"

    def alphabet_from_domains(self, domains: Any, core: str = "", *_a: Any, **_k: Any) -> LanthiAlphabet:
        core = core.upper()
        dehydrated = tuple(i for i, a in enumerate(core) if a in DEHYDRATABLE)  # v0: all Ser/Thr dehydrate
        return LanthiAlphabet(core=core, dehydrated=dehydrated)

    def enumerate(self, alphabet: LanthiAlphabet, min_len: int = 0, max_len: int = 0,
                  target_cho: tuple[int, int, int] | None = None) -> GenResult:
        raise NotImplementedError(
            "lanthipeptide structural execution pending chemist-signed renderers. The freed topology "
            "DOF is enumerated soundly by .topology_dof()/partition() (kappa_mass = #pairings, exact "
            "by mass-neutrality), but rendering candidates to SMILES needs the dehydration + lanthionine "
            "SMARTS (flagged). The engine's mass/MS-MS/verdict path runs unchanged once they arrive.")

    def mass_prefilter_keys(self, neutral_mass: float, ppm: float) -> list[tuple[int, ...]]:
        raise NotImplementedError(
            "pending renderers; product formula = core - n_dehydration*H2O (bridge mass-neutral)")

    # ------------------------------------------------------------------ sound, chemistry-free part
    def topology_dof(self, alphabet: LanthiAlphabet) -> list[LanthiProgram]:
        """The freed topology DOF: every legal Cys->Dha bridge assignment. len() == kappa_mass."""
        return [LanthiProgram(alphabet.core, alphabet.dehydrated, p)
                for p in _pairings(alphabet.cys, alphabet.dehydrated)]


#: module singleton.
LANTHI = LanthiGrammar()


def partition(core: str, dehydrate: tuple[int, ...] | None = None) -> dict:
    """Sec. 16.14 partition harness over genome / product / trajectory modalities, computed on the
    SOUND DOF (no chemistry). kappa_mass = #pairings is exact (mass-neutral bridging => all pairings
    share one formula). The product (MS/MS) resolution is PENDING the structural renderers."""
    core = core.upper()
    dha = dehydrate if dehydrate is not None else tuple(i for i, a in enumerate(core) if a in DEHYDRATABLE)
    cys = tuple(i for i, a in enumerate(core) if a == BRIDGE_DONOR)
    pairings = _pairings(cys, dha)
    kappa_mass = len(pairings)
    return {
        "core": core, "n_cys": len(cys), "n_dha": len(dha), "n_rings": len(cys),
        "kappa_mass": kappa_mass,                       # SOUND: all topologies share one formula
        "topology_U": kappa_mass >= 2,                  # automatic for >=2-ring cores (mass-neutrality)
        "genome":     {"axis": "core-residue identity", "verdict": "R",
                       "why": "core sequence is genome-read; composition kappa=1 given the ORF"},
        "mass":       {"kappa": kappa_mass, "verdict": "U" if kappa_mass >= 2 else "R",
                       "why": "thia-Michael bridge is mass-neutral => every topology shares the formula"},
        "product":    {"axis": "MS/MS fragmentation across rings", "verdict": "PENDING-renderers",
                       "predicted": "R (distinct pairings = distinct molecules = product-resolvable)"},
        "trajectory": {"axis": "control / cyclase", "verdict": "trajectory-controlled",
                       "why": "pairing is enzyme-determined (Repka); no genome edit reprograms it"},
        "cell": "(m_ctrl, m_conf) = (trajectory, product)   [product confirmation pending renderers]",
    }


if __name__ == "__main__":
    print("Lanthipeptide SCAFFOLD -- sound topology DOF (kappa_mass = #pairings); renderers stubbed.\n")
    for core in ["SC", "SCSC", "SCSCSC", "ITSAACDLCTPGC"]:
        p = partition(core)
        print(f"  core={core:14s} cys={p['n_cys']} dha={p['n_dha']} "
              f"kappa_mass={p['kappa_mass']:3d}  topology_U={str(p['topology_U']):5s}  {p['cell']}")
    print("\n  (renderer stubs raise pending chemist-signed SMARTS; no SMILES rendered, no datapoint claimed)")
