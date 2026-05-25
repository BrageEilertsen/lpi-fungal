"""Core-match scoring: does a predicted PKS core embed in the isolated product?

Rationale (Brage's reframe): the executor predicts the *polyketide synthase core*; MIBiG
records the isolated natural product = core + post-PKS tailoring (oxidation, halogenation,
prenylation, ring expansion, NRPS-appended amino acids, ...). Exact match (core == y) is
the airtight executor-correctness check but under-counts the trainable set, because it
penalises tailoring the model never claimed to predict.

Core-match instead asks: is the predicted core's **carbon skeleton a subgraph of the
product's carbon skeleton** (core subset-of y)? This absorbs *additive* tailoring (extra
substituents, fused rings, an appended amino acid) while still requiring the product to
contain the predicted backbone connectivity.

Under core-match the verifier is **core-sound, not exact-sound**: a core-match certifies
the backbone is reproducible, not the full isolated structure.

Limitation (reported honestly): a bare linear core embeds permissively. We therefore (a)
match on the carbon skeleton *with ring-membership preserved* (a core ring atom must map
to a ring atom of y), and (b) expose ``min_core_carbons`` so callers can require a
substantial core. Skeleton-rearranging tailoring (ring cleavage, dimerisation) is NOT
absorbed -- those targets are unreachable even at core level (the genuine ceiling).
"""

from __future__ import annotations

import networkx as nx
from rdkit import Chem


def carbon_skeleton_graph(mol: Chem.Mol) -> nx.Graph:
    """Carbon-only graph: nodes = carbon atoms (attr: in_ring), edges = C-C bonds.

    Bond order is ignored (tailoring/aromatisation changes it); heteroatoms and H are
    dropped (additive tailoring is absorbed). Ring membership is kept as a node attribute
    so a core ring carbon must map to a product ring carbon.
    """
    g = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            g.add_node(atom.GetIdx(), in_ring=atom.IsInRing())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 6:
            g.add_edge(a.GetIdx(), b.GetIdx())
    return g


def _node_match(core_attr: dict, y_attr: dict) -> bool:
    # A core ring carbon must map to a ring carbon of y; a core chain carbon may map to
    # either (a chain carbon in the core can be part of a ring in the larger product).
    if core_attr.get("in_ring"):
        return bool(y_attr.get("in_ring"))
    return True


def core_in_target(core: Chem.Mol, target: Chem.Mol, min_core_carbons: int = 4) -> bool:
    """True if the core's carbon skeleton is a subgraph of the target's carbon skeleton."""
    cg = carbon_skeleton_graph(core)
    if cg.number_of_nodes() < min_core_carbons:
        return False
    return _graph_in(cg, carbon_skeleton_graph(target))


def _graph_in(cg: nx.Graph, tg: nx.Graph) -> bool:
    if cg.number_of_nodes() > tg.number_of_nodes():
        return False
    matcher = nx.algorithms.isomorphism.GraphMatcher(tg, cg, node_match=_node_match)
    return matcher.subgraph_is_isomorphic()


# --------------------------------------------------------------------- core library
from dataclasses import dataclass  # noqa: E402

from lpi.chem.program import Cycle, Program, ReductionState as _R, Release  # noqa: E402
from lpi.executor import core as _core  # noqa: E402


@dataclass
class CoreEntry:
    program: Program
    smiles: str
    graph: nx.Graph
    n_carbons: int
    is_cyclic: bool


def build_core_library(max_cycles: int = 14) -> list[CoreEntry]:
    """Distinct producible carbon SKELETONS from representative programs.

    Reductions do not change a chain's carbon skeleton, so linear cores use a fixed
    reduction; cyclised cores use a reduction pattern that makes the release fire. This
    is representative, not exhaustive over methyl positions (documented). Linear cores are
    intentionally permissive; cyclised cores (ring systems) carry the meaningful signal.
    """
    progs: list[Program] = []
    # linear straight chains of increasing length (skeleton = path)
    for n in range(1, max_cycles + 1):
        progs.append(Program("acetyl", tuple(Cycle(_R.KETO) for _ in range(n)),
                             Release.HYDROLYSIS))
    # one methylated linear variant per length (a single branch point)
    for n in range(1, max_cycles + 1):
        cyc = tuple(Cycle(_R.KETO, c_methyl=(i == n // 2)) for i in range(n))
        progs.append(Program("acetyl", cyc, Release.HYDROLYSIS))
    # cyclised cores (ring systems) -- the meaningful skeletons
    progs += [
        Program("acetyl", (Cycle(_R.KETO),) * 2, Release.LACTONIZATION),      # pyranone C6
        Program("acetyl", (Cycle(_R.KETO),) * 3, Release.ALDOL_AROMATIC),     # resorcylate C8
        Program("acetyl", (Cycle(_R.KR), Cycle(_R.KETO), Cycle(_R.KR), Cycle(_R.KETO)),
                Release.DIHYDROISOCOUMARIN),                                   # mellein C10
        Program("acetyl", (Cycle(_R.KETO),) * 4, Release.PT_NAPHTHALENE),     # naphthalene C10
    ]
    lib: dict[str, CoreEntry] = {}
    for prog in progs:
        try:
            mol = _core.exec(prog)
        except Exception:  # noqa: BLE001
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in lib:
            continue
        g = carbon_skeleton_graph(mol)
        is_cyclic = any(d.get("in_ring") for _, d in g.nodes(data=True))
        lib[smi] = CoreEntry(prog, smi, g, g.number_of_nodes(), is_cyclic)
    return list(lib.values())


def best_core_match(target: Chem.Mol, library: list[CoreEntry],
                    min_core_carbons: int = 6) -> CoreEntry | None:
    """Best matching core by COVERAGE (matched-core carbons / target carbons).

    Coverage is the principled signal: a small ring embeds in almost any aromatic product
    (subgraph match alone is permissive), but a high-coverage core means the predicted
    backbone IS most of the molecule and the tailoring is minor/additive. We return the
    embedding core with the most carbons (= highest coverage for a fixed target).
    """
    tg = carbon_skeleton_graph(target)
    n_y = tg.number_of_nodes()
    candidates = [c for c in library if min_core_carbons <= c.n_carbons <= n_y]
    for entry in sorted(candidates, key=lambda c: -c.n_carbons):  # largest core first
        if _graph_in(entry.graph, tg):
            return entry
    return None


def coverage(core: CoreEntry, target: Chem.Mol) -> float:
    n_y = carbon_skeleton_graph(target).number_of_nodes()
    return core.n_carbons / n_y if n_y else 0.0
