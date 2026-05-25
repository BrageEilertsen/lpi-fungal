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


def _node_match(y_attr: dict, core_attr: dict) -> bool:
    # GraphMatcher(Yg, Cg) calls node_match(y_node_attr, core_node_attr).
    # A core RING carbon must map to a ring carbon of y (cyclisation cannot be undone);
    # a core CHAIN carbon may map to either (ring-closure tailoring can ringify it).
    if core_attr.get("in_ring"):
        return bool(y_attr.get("in_ring"))
    return True


_MAX_MAPPINGS = 20000  # cap monomorphism enumeration per (core, target) pair


def _monomorphisms(cg: nx.Graph, tg: nx.Graph):
    """Yield monomorphisms {y_node: core_node}: core edges -> y edges (y may have extra
    edges among matched atoms = ring-closure tailoring). Capped for tractability."""
    if cg.number_of_nodes() > tg.number_of_nodes():
        return
    gm = nx.algorithms.isomorphism.GraphMatcher(tg, cg, node_match=_node_match)
    for i, mapping in enumerate(gm.subgraph_monomorphisms_iter()):
        if i >= _MAX_MAPPINGS:
            return
        yield mapping


def any_subgraph_match(core: Chem.Mol, target: Chem.Mol, min_core_carbons: int = 4) -> bool:
    """DIAGNOSTIC upper bound: core skeleton embeds *somewhere* in y (permissive)."""
    cg = carbon_skeleton_graph(core)
    if cg.number_of_nodes() < min_core_carbons:
        return False
    for _ in _monomorphisms(cg, carbon_skeleton_graph(target)):
        return True
    return False


def skeleton_complete_match(core: Chem.Mol, target: Chem.Mol,
                            min_core_carbons: int = 4) -> bool:
    """Brage's tightened metric: the core accounts for ALL of y's carbon skeleton modulo
    *separable* additive decorations.

    A monomorphism of the core into y exists (core bonds subset y bonds; ring-aware nodes)
    such that every unmatched y-carbon component attaches to the matched core by exactly
    ONE bond -- i.e. it is a single-bond-separable decoration (prenyl, O-/C-methyl,
    glycosyl, halogen-bearing carbon, NRPS amino acid). A component attached by >=2 bonds
    is fused/bridging -- part of the skeleton, not a decoration -- so the core is NOT a
    complete account (that is the genuine ceiling: cleavage / rearrangement).
    """
    return skeleton_complete_extra_bonds(core, target, min_core_carbons) is not None


def skeleton_complete_extra_bonds(core: Chem.Mol, target: Chem.Mol,
                                  min_core_carbons: int = 4,
                                  max_extra_ring_bonds: int | None = None) -> int | None:
    """Min number of EXTRA ring-closure bonds (bonds in y among matched core atoms beyond
    the core's own bonds) over all skeleton-complete mappings; None if none qualifies.

    ``max_extra_ring_bonds`` caps how many ring-closure bonds count as allowable
    tailoring: 0 = the core's exact structure (incl. its rings) must appear (induced),
    forbidding a linear core from reconstructing a ring; None = unlimited (most permissive,
    lets a chain explain any backbone -- the swainsonine leak). This is the metric knob.
    """
    cg = carbon_skeleton_graph(core)
    if cg.number_of_nodes() < min_core_carbons:
        return None
    core_edges = cg.number_of_edges()
    tg = carbon_skeleton_graph(target)
    best: int | None = None
    for mapping in _monomorphisms(cg, tg):
        matched = set(mapping)
        unmatched = set(tg) - matched
        ok = True
        for comp in nx.connected_components(tg.subgraph(unmatched)):
            attach = sum(1 for n in comp for nb in tg.neighbors(n) if nb in matched)
            if attach != 1:  # >=2 = fused/bridging = part of skeleton, not a decoration
                ok = False
                break
        if not ok:
            continue
        extra = tg.subgraph(matched).number_of_edges() - core_edges
        if max_extra_ring_bonds is not None and extra > max_extra_ring_bonds:
            continue
        if best is None or extra < best:
            best = extra
            if best == 0:
                break
    return best


# --------------------------------------------------------------------- core library
from dataclasses import dataclass  # noqa: E402

from lpi.chem.program import Cycle, Program, ReductionState as _R, Release  # noqa: E402
from lpi.executor import core as _core  # noqa: E402


@dataclass
class CoreEntry:
    program: Program
    smiles: str
    mol: Chem.Mol
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
        lib[smi] = CoreEntry(prog, smi, mol, g, g.number_of_nodes(), is_cyclic)
    return list(lib.values())


def best_skeleton_complete(target: Chem.Mol, library: list[CoreEntry],
                           min_core_carbons: int = 6) -> CoreEntry | None:
    """Largest core that SKELETON-COMPLETELY matches y (the trainable-set metric)."""
    n_y = carbon_skeleton_graph(target).number_of_nodes()
    candidates = [c for c in library if min_core_carbons <= c.n_carbons <= n_y]
    for entry in sorted(candidates, key=lambda c: -c.n_carbons):  # largest = most coverage
        if skeleton_complete_match(entry.mol, target):
            return entry
    return None


def best_any_subgraph(target: Chem.Mol, library: list[CoreEntry],
                      min_core_carbons: int = 6) -> CoreEntry | None:
    """Largest core embedding ANYWHERE in y (DIAGNOSTIC upper bound, permissive)."""
    n_y = carbon_skeleton_graph(target).number_of_nodes()
    candidates = [c for c in library if min_core_carbons <= c.n_carbons <= n_y]
    for entry in sorted(candidates, key=lambda c: -c.n_carbons):
        if any_subgraph_match(entry.mol, target):
            return entry
    return None


def coverage(core: CoreEntry, target: Chem.Mol) -> float:
    """Fraction of y's carbons accounted for by the predicted core."""
    n_y = carbon_skeleton_graph(target).number_of_nodes()
    return core.n_carbons / n_y if n_y else 0.0
