"""BGC<->peak matching: the metabologenomics layer that makes the engine a discovery tool.

The paper's engine answers one cluster + one observation. Real discovery is *many clusters x many
peaks*: a genome yields a set of biosynthetic clusters, an untargeted-metabolomics run yields a set
of peaks, and the question is which cluster produces which peak (and what the structure is). This
module builds that scored assignment graph by running the shared engine over every (cluster, peak)
pair:

    edge(cluster c, peak p) exists  iff  c can produce a structure consistent with p's mass
                                          (verdict VERIFIED or UNDER-OBSERVED; OUT-OF-GRAMMAR -> no edge)

Each edge carries the three-state verdict, the candidate structure(s), and a confidence score
(higher for a VERIFIED, near-unique edge). From the graph we read off the *discovery call* -- the
best-scoring cluster for each peak -- and the *unexplained* peaks (no cluster under the current
grammars). It is grammar-agnostic: clusters may be PKS, NRPS, or hybrid; only ``infer_ms`` runs
underneath.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lpi.engine import State
from lpi.grammars import PKS, Grammar
from lpi.observe import MSObservables, infer_ms


@dataclass
class Cluster:
    """A biosynthetic gene cluster: a domain alphabet under a family grammar, with a step band."""

    id: str
    domains: Any                       # the cluster's catalytic domain set
    grammar: Grammar = PKS
    min_len: int = 1
    max_len: int = 8
    name: str = ""

    def alphabet(self):
        return self.grammar.alphabet_from_domains(self.domains)


@dataclass
class Peak:
    """An observed metabolomics peak: a measured ion m/z under one or more adduct hypotheses."""

    id: str
    mz: float
    adducts: tuple[str, ...] = ("[M+H]+", "[M-H]-")
    ppm: float = 5.0
    msms: tuple[float, ...] | None = None
    name: str = ""


@dataclass
class Assignment:
    """A scored cluster->peak edge: the cluster can make a structure at the peak's mass."""

    cluster_id: str
    peak_id: str
    state: State
    n_candidates: int
    top_smiles: str | None
    score: float


@dataclass
class MatchResult:
    edges: list[Assignment] = field(default_factory=list)        # all edges, ranked by score desc
    unexplained_peaks: list[str] = field(default_factory=list)   # peaks no cluster explains

    def best_per_peak(self) -> dict[str, Assignment]:
        """The single highest-scoring cluster for each peak (arbitrary tie-break; see
        :meth:`discovery_calls` for the honest, ambiguity-aware version)."""
        best: dict[str, Assignment] = {}
        for e in self.edges:
            cur = best.get(e.peak_id)
            if cur is None or e.score > cur.score:
                best[e.peak_id] = e
        return best

    def edges_for_peak(self, peak_id: str) -> list[Assignment]:
        return [e for e in self.edges if e.peak_id == peak_id]

    def discovery_calls(self) -> dict[str, dict]:
        """Per peak, the honest discovery call separating *structure* from *cluster attribution*:
        the top-scoring structure(s) and the cluster(s) tied at the top. When several
        grammar-degenerate clusters tie, the structure is determined but the producing cluster is
        not (``cluster_ambiguous``) -- the signal that disambiguation needs sequence/MS-MS/expression.
        """
        by_peak: dict[str, list[Assignment]] = {}
        for e in self.edges:
            by_peak.setdefault(e.peak_id, []).append(e)
        out: dict[str, dict] = {}
        for pid, es in by_peak.items():
            top = max(e.score for e in es)
            tied = [e for e in es if abs(e.score - top) < 1e-9]
            out[pid] = {
                "state": tied[0].state,
                "score": top,
                "structures": sorted({e.top_smiles for e in tied if e.top_smiles}),
                "clusters": sorted({e.cluster_id for e in tied}),
                "cluster_ambiguous": len({e.cluster_id for e in tied}) > 1,
                "n_edges": len(es),
            }
        return out


# verdict priority for scoring: a VERIFIED edge beats an under-observed one.
_STATE_BASE = {State.VERIFIED: 1.0, State.UNDER_OBSERVED: 0.4}


def _edge_score(state: State, n_candidates: int) -> float:
    """Confidence that the cluster produces the peak: high for a VERIFIED, near-unique edge;
    decays as the candidate set grows (a peak many programs could hit is weak evidence)."""
    base = _STATE_BASE.get(state, 0.0)
    return base / (1.0 + math.log1p(max(0, n_candidates - 1)))


def match(clusters: list[Cluster], peaks: list[Peak]) -> MatchResult:
    """Score every (cluster, peak) edge by running the shared engine; build the assignment graph."""
    edges: list[Assignment] = []
    explained: set[str] = set()
    for c in clusters:
        alpha = c.alphabet()
        for p in peaks:
            res = infer_ms(alpha,
                           MSObservables(observed_mz=p.mz, adducts=p.adducts, ppm=p.ppm,
                                         msms_peaks=p.msms),
                           c.grammar, c.min_len, c.max_len, ladder=False)
            if res.state is State.OUT_OF_GRAMMAR:
                continue                                  # this cluster cannot make this peak
            top = res.candidates[0].smiles if res.candidates else None
            edges.append(Assignment(c.id, p.id, res.state, len(res.candidates), top,
                                    _edge_score(res.state, len(res.candidates))))
            explained.add(p.id)
    edges.sort(key=lambda e: (-e.score, e.cluster_id, e.peak_id))
    unexplained = [p.id for p in peaks if p.id not in explained]
    return MatchResult(edges, unexplained)
