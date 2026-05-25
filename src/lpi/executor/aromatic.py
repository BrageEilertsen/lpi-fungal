"""NR-PKS product-template (PT) domain aromatic cyclization.

The PT domain is part of the NR-PKS megasynthase (not post-PKS tailoring), so its
aromatic cyclizations are PKS-intrinsic chemistry and legitimately in scope. PT directs
which intramolecular aldol/Claisen regiochemistry fires; Crawford et al. (2009) describe
~5 regiochemical classes across chain lengths (tetra-/penta-/hexa-/hepta-/octaketide).

Status:
  * Tetraketide single-ring class (orsellinic / 6-MSA) is handled in
    :mod:`lpi.executor.cyclize` (one productive mode at that chain length).
  * Pentaketide -> naphthalene (the T4HN / scytalone melanin class) is implemented here.
    The carbon SKELETON produced is a correct naphthalene (validated), which is what
    core-match needs. The exact hydroxyl REGIOCHEMISTRY differs from 1,3,6,8-T4HN by one
    position -- the precise PT fold is FLAGGED for Brage in QUESTIONS.md (Q8); until
    confirmed, this release is skeleton-correct but not exact-match-correct, so it is used
    for core-match but NOT counted as an exact-match hit.
  * Hexa-/hepta-/octaketide PT classes (anthrone/aflatoxin precursors) are NOT yet
    implemented -- they need the tethered (pre-release) Claisen path and confirmed
    regiochemistry (QUESTIONS.md Q8).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

PT_REGIOSELECTIVITY_CLASSES = ("C2-C7 (tetraketide)", "naphthalene (pentaketide)",
                               "C4-C9", "C6-C11", "octaketide-anthrone")

# Non-reduced pentaketide thioester -> tetrahydroxynaphthalene. Two ring-closing bonds
# (starter-methyl carbon becomes a bridgehead; C5-keto oxygen is the leaving water; the
# thioester is released by Claisen). Skeleton = naphthalene (correct); OH regiochemistry
# is pending Brage's PT-fold confirmation (QUESTIONS.md Q8).
_PT_NAPHTHALENE = (
    "[CH3:1][C:2](=[O:11])[CH2:3][C:4](=[O:12])[CH2:5][C:6](=[O:13])[CH2:7]"
    "[C:8](=[O:14])[CH2:9][C:10](=[O:15])[S:16][#0:17]"
    ">>[c:1]12[c:10]([OH:15])[cH:9][c:8]([OH:14])[cH:7][c:6]1[cH:5][c:4]([OH:12])"
    "[cH:3][c:2]2[OH:11]"
)


class PTDomainNotImplemented(NotImplementedError):
    pass


def cyclize_naphthalene(tethered_pentaketide: Chem.Mol) -> Chem.Mol | None:
    """Pentaketide thioester -> tetrahydroxynaphthalene (skeleton-correct)."""
    rxn = AllChem.ReactionFromSmarts(_PT_NAPHTHALENE)
    rxn.Initialize()
    products: dict[str, Chem.Mol] = {}
    for (prod,) in rxn.RunReactants((tethered_pentaketide,)):
        for a in prod.GetAtoms():
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(prod)
        except Exception:  # noqa: BLE001
            continue
        products[Chem.MolToSmiles(prod)] = prod
    if len(products) == 1:
        return next(iter(products.values()))
    return None
