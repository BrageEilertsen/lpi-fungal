"""Deterministic operator library: the per-step chemistry, as reaction SMARTS.

Each operator is a *fixed rule*, never learned (Section 4.1). Every operator is
anchored on the single active thioester terminus ``...C(=O)S*`` (sentinel dummy atom),
which is unique in any well-formed intermediate, so each reaction matches exactly one
site and is deterministic.

Atom-position convention relative to the active thioester carbonyl C1:
    ...[C(beta)]( =O )[C(alpha)]H2[C1]( =O )[S][*]
    KR/DH/ER act on the beta-position; C-MeT methylates the alpha-CH2.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

# --- starter acyl-thioester SMILES (tethered to the sentinel dummy carrier) -------
STARTERS: dict[str, str] = {
    "acetyl": "CC(=O)S*",
    "propionyl": "CCC(=O)S*",
    "butyryl": "CCCC(=O)S*",
    "hexanoyl": "CCCCCC(=O)S*",  # fatty-acyl starter (e.g. olivetolic acid, alkylresorcylics)
    "benzoyl": "c1ccccc1C(=O)S*",
}

# --- reaction SMARTS --------------------------------------------------------------
# Decarboxylative Claisen extension (malonyl): append -CH2-C(=O)- to the active
# thioester carbonyl, making the old carbonyl the new beta-keto. Net +C2H2O.
_EXTEND_MALONYL = "[C:1](=[O:2])[S:3][#0:4]>>[C:1](=[O:2])[CH2][CX3](=O)[S:3][#0:4]"

# Methylmalonyl extension: like malonyl but installs an alpha-methyl branch from the
# extender (a bacterial-AT feature; reserved for the ClusterCAD pretraining side).
_EXTEND_METHYLMALONYL = (
    "[C:1](=[O:2])[S:3][#0:4]>>[C:1](=[O:2])[CH1]([CH3])[CX3](=O)[S:3][#0:4]"
)

# C-MeT: genuine SAM-dependent alpha-methylation of the sp3 alpha-CH2. Net +CH2.
_C_METHYL = "[CH2:1][CX3:2](=[O:3])[S:4][#0:5]>>[CH1:1]([CH3])[CX3:2](=[O:3])[S:4][#0:5]"

# KR: reduce the beta-keto to beta-hydroxyl. alpha is any carbon (CH2 or methylated
# CH). Net +H2.
_KR = (
    "[CX3:1](=[O:2])[#6:3][CX3:4](=[O:5])[S:6][#0:7]"
    ">>[CX4:1]([OH:2])[#6:3][CX3:4](=[O:5])[S:6][#0:7]"
)

# DH: dehydrate beta-hydroxyl + alpha-H to the alpha,beta-enoyl. Net -H2O.
_DH = (
    "[CX4:1]([OH:2])[#6:3][CX3:4](=[O:5])[S:6][#0:7]"
    ">>[C:1]=[C:3][CX3:4](=[O:5])[S:6][#0:7]"
)

# ER: reduce the alpha,beta-enoyl to the methylene. Net +H2.
_ER = (
    "[C:1]=[C:3][CX3:4](=[O:5])[S:6][#0:7]"
    ">>[CX4:1][CX4:3][CX3:4](=[O:5])[S:6][#0:7]"
)

# TE hydrolysis: release the chain as a carboxylic acid (the linear checkpoint).
_HYDROLYSIS = "[C:1](=[O:2])[S:3][#0:4]>>[C:1](=[O:2])[OH]"


def _reaction(smarts: str) -> AllChem.ChemicalReaction:
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    return rxn


_RXN = {
    "extend_malonyl": _reaction(_EXTEND_MALONYL),
    "extend_methylmalonyl": _reaction(_EXTEND_METHYLMALONYL),
    "c_methyl": _reaction(_C_METHYL),
    "kr": _reaction(_KR),
    "dh": _reaction(_DH),
    "er": _reaction(_ER),
    "hydrolysis": _reaction(_HYDROLYSIS),
}


class OperatorError(RuntimeError):
    """Raised when an operator cannot be applied or is non-deterministic."""


def _apply(name: str, mol: Chem.Mol) -> Chem.Mol:
    """Apply a named single-site reaction, requiring exactly one distinct product.

    Determinism guard: if the reaction produces more than one *distinct* product
    (by canonical SMILES) we raise, because that means the operator's match site was
    ambiguous -- a bug we want to surface, not paper over.
    """
    rxn = _RXN[name]
    outcomes = rxn.RunReactants((mol,))
    if not outcomes:
        raise OperatorError(
            f"operator {name!r} did not match intermediate {Chem.MolToSmiles(mol)!r}"
        )
    products: dict[str, Chem.Mol] = {}
    for (prod,) in outcomes:
        # RDKit reaction SMARTS carries the reactant's H count onto product atoms
        # whose bond order changed (e.g. an alpha-CH2 that gains a double bond keeps
        # both H -> valence 5). These tethered intermediates are plain aliphatic
        # molecules with no explicit-H-dependent features, so we clear explicit H and
        # let sanitization recompute implicit H from valence.
        for atom in prod.GetAtoms():
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(prod)
        except Exception as exc:  # noqa: BLE001
            raise OperatorError(f"operator {name!r} produced unsanitizable mol: {exc}") from exc
        products[Chem.MolToSmiles(prod)] = prod
    if len(products) != 1:
        raise OperatorError(
            f"operator {name!r} was ambiguous on {Chem.MolToSmiles(mol)!r}: "
            f"{sorted(products)}"
        )
    return next(iter(products.values()))


def load_starter(name: str) -> Chem.Mol:
    if name not in STARTERS:
        raise OperatorError(f"unknown starter {name!r}; known: {sorted(STARTERS)}")
    mol = Chem.MolFromSmiles(STARTERS[name])
    if mol is None:
        raise OperatorError(f"could not build starter {name!r}")
    return mol


def extend(mol: Chem.Mol, methylmalonyl: bool = False) -> Chem.Mol:
    return _apply("extend_methylmalonyl" if methylmalonyl else "extend_malonyl", mol)


def c_methylate(mol: Chem.Mol) -> Chem.Mol:
    return _apply("c_methyl", mol)


def ketoreduce(mol: Chem.Mol) -> Chem.Mol:
    return _apply("kr", mol)


def dehydrate(mol: Chem.Mol) -> Chem.Mol:
    return _apply("dh", mol)


def enoylreduce(mol: Chem.Mol) -> Chem.Mol:
    return _apply("er", mol)


def hydrolyse(mol: Chem.Mol) -> Chem.Mol:
    """TE hydrolysis -> carboxylic acid (the released linear chain)."""
    return _apply("hydrolysis", mol)
