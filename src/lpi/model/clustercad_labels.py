"""Per-module action labels (from structure) + domain features for the differential PoC.

For each ClusterCAD module we read the GROUND-TRUTH per-step action from the intermediate
structure (not from domain presence -- that is the whole point: domain present != active),
and build the domain feature vector. Reduction state is classified from the beta-position of
the most-recently-added unit, located at the active thioester C(=O)[S].
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_THIOESTER = Chem.MolFromSmarts("[CX3](=O)[#16]")

# domain features the per-step action can depend on (KS/AT/ACP are ubiquitous -> uninformative
# but kept; reductive + MT are the discriminative ones).
FEATURE_DOMAINS = ("AT", "KR", "DH", "ER", "cMT", "oMT", "nMT")
REDUCTION_CLASSES = ("KETO", "KR", "DH", "ER")
STEREO_CLASSES = ("none", "R", "S")


@dataclass
class ModuleLabel:
    reduction: str  # KETO / KR / DH / ER / None (unparseable / loading)
    stereo: str  # none / R / S (chirality at the beta-carbon for KR-type)
    alpha_methyl: bool  # an alpha-methyl branch is present on the new unit
    features: dict[str, int]


def _carbonyl_alpha_beta(mol: Chem.Mol):
    """Return (carbonyl_idx, alpha_idx, beta_idx) for the active thioester, or None."""
    ms = mol.GetSubstructMatches(_THIOESTER)
    if not ms:
        return None
    c_idx, _o, _s = ms[-1]  # the last thioester = the active chain terminus
    carbonyl = mol.GetAtomWithIdx(c_idx)
    alpha = None
    for nb in carbonyl.GetNeighbors():
        if nb.GetAtomicNum() == 6:  # carbon neighbour that is not =O / S
            alpha = nb
            break
    if alpha is None:
        return None
    beta = None
    for nb in alpha.GetNeighbors():
        if nb.GetIdx() != c_idx and nb.GetAtomicNum() == 6:
            beta = nb
            break
    return c_idx, alpha.GetIdx(), (beta.GetIdx() if beta else None)


def reduction_state(mol: Chem.Mol) -> tuple[str, str]:
    """(reduction class, stereo) from the structure at the active thioester."""
    ab = _carbonyl_alpha_beta(mol)
    if ab is None:
        return "KETO", "none"
    c_idx, a_idx, b_idx = ab
    if b_idx is None:
        return "KETO", "none"  # only one carbon before thioester (loading / acetoacetyl edge)
    alpha = mol.GetAtomWithIdx(a_idx)
    beta = mol.GetAtomWithIdx(b_idx)
    # enoyl: alpha=beta double bond
    bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
    if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
        return "DH", "none"
    # beta substituents
    beta_has_keto = any(
        b.GetBondType() == Chem.BondType.DOUBLE and
        mol.GetAtomWithIdx(b.GetOtherAtomIdx(b_idx)).GetAtomicNum() == 8
        for b in beta.GetBonds())
    beta_has_oh = any(
        b.GetBondType() == Chem.BondType.SINGLE and
        mol.GetAtomWithIdx(b.GetOtherAtomIdx(b_idx)).GetAtomicNum() == 8
        for b in beta.GetBonds())
    if beta_has_keto:
        return "KETO", "none"
    if beta_has_oh:
        tag = beta.GetChiralTag()
        stereo = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: "R",
                  Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "S"}.get(tag, "none")
        return "KR", stereo
    return "ER", "none"  # saturated methylene, no beta-oxygen


def has_alpha_methyl(mol: Chem.Mol) -> bool:
    ab = _carbonyl_alpha_beta(mol)
    if ab is None:
        return False
    _c, a_idx, _b = ab
    alpha = mol.GetAtomWithIdx(a_idx)
    # an alpha carbon bearing a CH3 branch (degree-1 carbon neighbour)
    return any(nb.GetAtomicNum() == 6 and nb.GetDegree() == 1 for nb in alpha.GetNeighbors())


def domain_features(domains: list[str]) -> dict[str, int]:
    s = set(domains)
    return {d: int(d in s) for d in FEATURE_DOMAINS}


def label_module(domains: list[str], intermediate_smiles: str) -> ModuleLabel | None:
    mol = Chem.MolFromSmiles(intermediate_smiles)
    if mol is None:
        return None
    red, stereo = reduction_state(mol)
    return ModuleLabel(reduction=red, stereo=stereo,
                       alpha_methyl=has_alpha_methyl(mol),
                       features=domain_features(domains))
