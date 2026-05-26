"""Deterministic NRPS peptide executor (v0).

The peptide counterpart of the PKS executor (:mod:`lpi.executor.core`): a pure,
deterministic function from a discrete program to a molecule, *validated* by mass
bookkeeping and known-product round-trips, never trained. It exists to prove the engine
generalizes -- the same observability (mass / MS-MS) and three-state verdict layers mine
through it unchanged.

Program model
-------------
A non-ribosomal peptide program is an ordered list of amino-acid *condensation modules*
(each an A-domain selection) plus a terminal release. The backbone is built by amide
condensation: residue ``i``'s carbonyl bonds to residue ``i+1``'s alpha-amine, losing one
water per bond.

Scope (v0), with the genuine chemistry calls flagged in ``QUESTIONS.md`` (workflow rule 5):
* achiral backbone -- D/L epimerization (E domains) is not modelled, exactly as the PKS
  executor is achiral by design;
* a small set of proteinogenic residues (no proline / secondary-amine residues, whose ring
  N breaks the linear backbone assumption);
* two releases -- linear hydrolysis (free peptide acid) and head-to-tail macrolactamization
  (a cyclic peptide; a 2,5-diketopiperazine for ``n=2``);
* deferred: heterocyclization (Cy -> thiazoline/oxazoline), N-methylation, side-chain and
  branch macrocyclizations, non-proteinogenic monomers, and starter/tailoring chemistry.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rdkit import Chem, RDLogger

from lpi.chem import mol as M

RDLogger.DisableLog("rdApp.*")

# Residue table: name -> (alpha-carbon sidechain SMILES fragment, free amino-acid formula
# (C, H, N, O)). The sidechain is inserted into the backbone unit ``N C(<sidechain>) C(=O)``;
# glycine has no sidechain. SMILES are achiral by design (no @/@@), consistent with the PKS
# executor. The free-AA formulas are textbook values used as the *independent* analytic source
# -- a property test confirms the assembled-SMILES formula matches, catching any sidechain typo.
RESIDUES: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "Gly": ("", (2, 5, 1, 2)),
    "Ala": ("C", (3, 7, 1, 2)),
    "Val": ("C(C)C", (5, 11, 1, 2)),
    "Leu": ("CC(C)C", (6, 13, 1, 2)),
    "Ser": ("CO", (3, 7, 1, 3)),
    "Thr": ("C(C)O", (4, 9, 1, 3)),
    "Phe": ("Cc1ccccc1", (9, 11, 1, 2)),
    "Tyr": ("Cc1ccc(O)cc1", (9, 11, 1, 3)),
}

_WATER = (0, 2, 0, 1)  # (C, H, N, O) removed per amide bond formed


class Release(Enum):
    """Terminal chain-release mode for the assembled peptidyl chain."""

    HYDROLYSIS = "hydrolysis"      # TE hydrolysis -> free linear peptide acid
    MACROLACTAM = "macrolactam"    # head-to-tail amide -> cyclic peptide (DKP for n=2)


@dataclass(frozen=True)
class Peptide:
    """A complete NRPS program: ordered residues + a terminal release."""

    residues: tuple[str, ...] = ()
    release: Release = Release.HYDROLYSIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "residues", tuple(self.residues))

    @property
    def n_modules(self) -> int:
        return len(self.residues)


class NRPSExecError(RuntimeError):
    pass


def _backbone_unit(residue: str) -> str:
    """The internal backbone fragment ``N-Calpha(sidechain)-C(=O)`` for one residue."""
    try:
        side, _ = RESIDUES[residue]
    except KeyError:
        raise NRPSExecError(f"unknown residue {residue!r}; have {sorted(RESIDUES)}") from None
    return f"NC({side})C(=O)" if side else "NCC(=O)"


def linear_smiles(residues: tuple[str, ...]) -> str:
    """SMILES of the free linear peptide acid (H2N-...-COOH)."""
    return "".join(_backbone_unit(r) for r in residues) + "O"


def cyclic_smiles(residues: tuple[str, ...]) -> str:
    """SMILES of the head-to-tail cyclic peptide (last carbonyl bonded to first amine).

    The macrocycle uses the two-digit ring-closure label ``%99`` so it cannot collide with
    single-digit ring bonds inside aromatic sidechains (e.g. Phe/Tyr ``c1ccccc1``).
    """
    units = [_backbone_unit(r) for r in residues]
    body = "N%99" + units[0][1:] + "".join(units[1:])  # ring-label the N-terminal nitrogen
    return body + "%99"                                  # ...and the C-terminal carbonyl carbon


def peptide_formula(residues: tuple[str, ...], release: Release) -> tuple[int, int, int, int]:
    """Analytic (C, H, N, O) of the released peptide -- no RDKit call.

    Sum of free-AA formulas minus one water per amide bond: ``n-1`` bonds for a linear acid,
    ``n`` for a head-to-tail macrocycle. Used as the engine's exact formula prefilter for NRPS.
    """
    c = h = n = o = 0
    for r in residues:
        rc, rh, rn, ro = RESIDUES[r][1]
        c += rc; h += rh; n += rn; o += ro
    bonds = len(residues) if release is Release.MACROLACTAM else max(0, len(residues) - 1)
    h -= _WATER[1] * bonds
    o -= _WATER[3] * bonds
    return (c, h, n, o)


def formula_chno(mol: Chem.Mol) -> tuple[int, int, int, int]:
    """(C, H, N, O) of a molecule -- the NRPS molecular-formula key (the mass observable).

    The peptide analogue of :func:`lpi.search.beam._formula_cho`; includes nitrogen, which
    is what distinguishes peptide isomers by mass.
    """
    c = h = n = o = 0
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        h += a.GetTotalNumHs()
        if z == 6:
            c += 1
        elif z == 7:
            n += 1
        elif z == 8:
            o += 1
        elif z == 1:
            h += 1
    return (c, h, n, o)


def run(peptide: Peptide) -> Chem.Mol:
    """Execute a peptide program into a molecule. Pure and deterministic.

    Mirrors ``exec(z)`` in the paper. Macrolactamization requires at least two residues
    (a one-residue head-to-tail closure would be a strained alpha-lactam, out of scope).
    """
    if peptide.release is Release.MACROLACTAM:
        if peptide.n_modules < 2:
            raise NRPSExecError("macrolactam requires >= 2 residues")
        smi = cyclic_smiles(peptide.residues)
    else:
        if peptide.n_modules < 1:
            raise NRPSExecError("a peptide needs >= 1 residue")
        smi = linear_smiles(peptide.residues)
    try:
        return M.mol_from_smiles(smi)
    except ValueError as exc:  # should not happen for table residues; surfaced loudly if it does
        raise NRPSExecError(f"executor built invalid SMILES {smi!r}: {exc}") from exc


def exec(peptide: Peptide) -> Chem.Mol:  # noqa: A001 - mirrors lpi.executor.core.exec
    """Convenience alias: the program's released product molecule."""
    return run(peptide)
