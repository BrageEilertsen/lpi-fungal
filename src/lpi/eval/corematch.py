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
    tg = carbon_skeleton_graph(target)
    if cg.number_of_nodes() > tg.number_of_nodes():
        return False
    matcher = nx.algorithms.isomorphism.GraphMatcher(tg, cg, node_match=_node_match)
    return matcher.subgraph_is_isomorphic()
