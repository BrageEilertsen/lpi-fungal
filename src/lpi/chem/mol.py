"""Molecule representation, canonicalization, and similarity helpers.

The growing chain is modelled as an ACP-tethered thioester: the carrier is a single
*sentinel dummy atom* (RDKit atomic number 0) attached to the thioester sulfur,
``...C(=O)S*``. Because there is exactly one active thioester at any time, the
sentinel uniquely anchors every operator's reaction SMARTS (see executor.operators).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

SENTINEL_ATOMIC_NUM = 0  # RDKit dummy atom ('*') used as the ACP carrier placeholder


def mol_from_smiles(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """Parse SMILES into an RDKit Mol, raising on failure."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    """Canonical SMILES (no atom maps), the primary structure-match key."""
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(work)


def inchi(mol: Chem.Mol) -> str:
    """InChI string, used to absorb tautomer/representation noise in matching."""
    return Chem.MolToInchi(mol, logLevel=None, treatWarningAsError=False)


def inchikey(mol: Chem.Mol) -> str:
    return Chem.InchiToInchiKey(inchi(mol))


def has_sentinel(mol: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == SENTINEL_ATOMIC_NUM for a in mol.GetAtoms())


def molecular_formula(mol: Chem.Mol) -> str:
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.CalcMolFormula(mol)


def exact_mass(mol: Chem.Mol) -> float:
    """Monoisotopic exact mass. Dummy/sentinel atoms contribute 0."""
    return Descriptors.ExactMolWt(mol)


def structures_match(mol_a: Chem.Mol, mol_b: Chem.Mol) -> dict[str, bool]:
    """Compare two molecules on canonical SMILES and InChI.

    Returns a dict with both signals so callers can require either or both. We treat
    a match on *either* canonical SMILES or InChI as a structure match, because InChI
    absorbs tautomer/representation differences that the canonical SMILES does not.
    """
    smi_a, smi_b = canonical_smiles(mol_a), canonical_smiles(mol_b)
    try:
        ik_a, ik_b = inchikey(mol_a), inchikey(mol_b)
        inchi_ok = bool(ik_a) and ik_a == ik_b
    except Exception:
        inchi_ok = False
    smiles_ok = smi_a == smi_b
    return {
        "smiles": smiles_ok,
        "inchi": inchi_ok,
        "match": smiles_ok or inchi_ok,
    }


def tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> float:
    """Morgan/ECFP Tanimoto similarity."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_a = gen.GetFingerprint(mol_a)
    fp_b = gen.GetFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def scaffold_match(mol_a: Chem.Mol, mol_b: Chem.Mol) -> bool:
    return murcko_scaffold_smiles(mol_a) == murcko_scaffold_smiles(mol_b)
