"""Deterministic PKS-NRPS hybrid executor (v0): typed composition of two grammars.

A hybrid synthase assembles a polyketide chain (PKS), hands the acyl chain off to an NRPS module
that condenses one or more amino acids by an amide bond, then releases. We model this as the typed
COMPOSITION of the existing executors, not as new chemistry:

    z_hybrid = z_PKS  o  h_handoff  o  z_NRPS  o  r_release

* ``z_PKS``     -- the PKS *linear acid* (the paper's linear checkpoint, which is exactly the
  PKS-NRPS handoff point: lpi.executor.core.run(...).linear);
* ``h_handoff`` -- a typed amide-condensation operator joining the PK carboxyl to the peptide's
  N-terminus (explicit RWMol surgery, never a hidden special case);
* ``z_NRPS``    -- the NRPS peptide backbone, reusing lpi.executor.nrps building blocks;
* ``r_release`` -- hydrolysis (the linear hybrid acid). The fungal tetramic-acid Dieckmann release
  (e.g. pretenellin A from the TenS hybrid) is a known scope boundary: it is declared as a *typed*
  release that is NOT implemented, so requesting it raises :class:`ReleaseNotImplemented` -- an
  explicit operator gap, never a silent failure.

Conservative by design: the point is typed compositionality with a shared observability/verdict
layer, not PKS-NRPS biochemical completeness.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.chem.program import Program
from lpi.executor import core as _pks
from lpi.executor import nrps as _nrps
from lpi.search.beam import linear_acid_formula

RDLogger.DisableLog("rdApp.*")

_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")        # PK free carboxyl (carbonyl C, =O, -OH)
_PRIMARY_AMINE = Chem.MolFromSmarts("[NX3;H2]")        # peptide N-terminus


class HybridRelease(Enum):
    """Terminal release of the assembled hybrid chain."""

    HYDROLYSIS = "hydrolysis"          # free linear hybrid acid (implemented)
    TETRAMIC_ACID = "tetramic_acid"    # Dieckmann -> tetramic acid (typed scope boundary; unimplemented)


#: the releases the executor can actually render; others are typed-but-unimplemented operators.
IMPLEMENTED_RELEASES = frozenset({HybridRelease.HYDROLYSIS})


@dataclass(frozen=True)
class Hybrid:
    """A hybrid program: a PKS sub-program, the NRPS handoff residues, and a release."""

    pks: Program
    residues: tuple[str, ...] = ()
    release: HybridRelease = HybridRelease.HYDROLYSIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "residues", tuple(self.residues))


class HybridError(RuntimeError):
    pass


class ReleaseNotImplemented(HybridError):
    """Requested a typed release operator that is not implemented (a declared scope boundary)."""


def handoff_amide(pk_acid: Chem.Mol, residues: tuple[str, ...]) -> Chem.Mol | None:
    """Typed handoff operator: amide-couple the PK linear acid's carboxyl to the peptide N-terminus.

    Returns ``None`` if the PK product exposes no free carboxyl or the peptide no primary amine
    (so the caller can raise a typed handoff failure rather than fail silently).
    """
    pep = M.mol_from_smiles(_nrps.linear_smiles(residues))
    acid = pk_acid.GetSubstructMatch(_ACID)
    amine = pep.GetSubstructMatch(_PRIMARY_AMINE)
    if not acid or not amine:
        return None
    c_carbonyl, _o_dbl, o_h = acid
    rw = Chem.RWMol(Chem.CombineMols(pk_acid, pep))
    n_off = pk_acid.GetNumAtoms()
    rw.AddBond(c_carbonyl, n_off + amine[0], Chem.BondType.SINGLE)  # the new amide C-N bond
    rw.RemoveAtom(o_h)                                              # drop the carboxyl -OH (water)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def hybrid_formula(pks: Program, residues: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Analytic (C, H, N, O) of the linear hybrid acid -- composes the two grammars' analytic
    formulas: PK linear acid + free peptide, minus the one handoff-condensation water."""
    pc, ph, po = linear_acid_formula(pks)                           # PK (C, H, O)
    qc, qh, qn, qo = _nrps.peptide_formula(residues, _nrps.Release.HYDROLYSIS)  # peptide (C,H,N,O)
    return (pc + qc, ph + qh - 2, qn, po + qo - 1)                  # - H2O for the handoff amide


def run(h: Hybrid) -> Chem.Mol:
    """Execute a hybrid program into a molecule. Pure and deterministic.

    Raises :class:`ReleaseNotImplemented` for a typed-but-unimplemented release (scope boundary),
    and :class:`HybridError` if the handoff cannot be applied.
    """
    if h.release not in IMPLEMENTED_RELEASES:
        raise ReleaseNotImplemented(
            f"hybrid release {h.release.value!r} is a declared but unimplemented operator "
            f"(scope boundary: e.g. the tetramic-acid Dieckmann release of TenS)")
    if not h.residues:
        raise HybridError("a hybrid program needs >= 1 handoff residue")
    pk_acid = _pks.run(h.pks).linear        # the linear checkpoint = the PKS-NRPS handoff point
    mol = handoff_amide(pk_acid, h.residues)
    if mol is None:
        raise HybridError("handoff failed: PK product has no free carboxyl or peptide no N-terminus")
    return mol


def exec(h: Hybrid) -> Chem.Mol:  # noqa: A001 - mirrors core.exec / nrps.exec
    return run(h)
