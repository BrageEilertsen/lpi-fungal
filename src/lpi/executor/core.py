"""The deterministic chemical executor: ``exec(program, operator_library) -> Mol``.

This is the *verifier* (Section 4.1). It is a pure function of the program: no RNG,
no global mutable state. Given a :class:`~lpi.chem.program.Program` it walks the
elongation cycles applying fixed operator rules, then exposes two release products:

* ``linear``  -- the released linear chain (TE hydrolysis -> carboxylic acid),
  computed for *every* program regardless of its declared release. This is the
  linear-chain checkpoint: it isolates the controller's actual scientific output
  (chain length + per-cycle reduction/methylation program) from downstream cyclization.
* ``final``   -- the product of the program's declared release operator. Equals
  ``linear`` for HYDROLYSIS; a cyclized structure otherwise. ``None`` if a
  best-effort cyclization could not be applied.

Cycle order (Section 3): extend -> C-MeT (optional) -> KR -> DH -> ER. The reductive
cascade ER>DH>KR is enforced by construction (a cycle declares one reachable state).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem, RDLogger

from lpi.chem.program import Extender, Program, ReductionState, Release
from . import operators as op

# RDKit logs intended atom deletions (dropped OH/S sentinel) as warnings; silence.
RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Output of :func:`run`. ``linear`` is always present; ``final`` may be None."""

    program: Program
    tethered: Chem.Mol  # final ACP-tethered intermediate, pre-release
    linear: Chem.Mol  # released linear chain (TE hydrolysis), the checkpoint
    final: Chem.Mol | None  # product of the declared release operator
    cyclization_ok: bool  # whether the declared (non-hydrolysis) release succeeded


class ExecError(RuntimeError):
    pass


def _apply_reduction(mol: Chem.Mol, state: ReductionState) -> Chem.Mol:
    """Apply the reductive cascade up to ``state`` (KR, then DH, then ER)."""
    if state is ReductionState.KETO:
        return mol
    mol = op.ketoreduce(mol)  # all reduced states pass through KR
    if state is ReductionState.KR:
        return mol
    mol = op.dehydrate(mol)  # DH presupposes KR
    if state is ReductionState.DH:
        return mol
    mol = op.enoylreduce(mol)  # ER presupposes DH
    return mol


def run(program: Program, operator_library=op) -> ExecResult:
    """Execute a program into molecules. Pure and deterministic.

    ``operator_library`` defaults to the module-level operator set; it is a parameter
    so Phase 2 can pass a library whose thin learned heads have annotated, e.g.,
    stereochemistry. The connectivity rules here are never learned.
    """
    mol = operator_library.load_starter(program.starter)

    for i, cycle in enumerate(program.cycles):
        methylmalonyl = cycle.extender is Extender.METHYLMALONYL
        try:
            mol = operator_library.extend(mol, methylmalonyl=methylmalonyl)
            if cycle.c_methyl:
                # C-MeT acts on the sp3 alpha-CH2 at the beta-keto stage, before KR.
                mol = operator_library.c_methylate(mol)
            mol = _apply_reduction(mol, cycle.reduction)
        except op.OperatorError as exc:
            raise ExecError(
                f"program failed at cycle {i} ({cycle}): {exc}"
            ) from exc

    tethered = mol
    linear = operator_library.hydrolyse(tethered)

    final: Chem.Mol | None
    cyclization_ok: bool
    if program.release in (Release.HYDROLYSIS, Release.NONE):
        final = linear if program.release is Release.HYDROLYSIS else None
        cyclization_ok = True
    else:
        from . import cyclize

        final = cyclize.release(linear, program.release)
        cyclization_ok = final is not None

    return ExecResult(
        program=program,
        tethered=tethered,
        linear=linear,
        final=final,
        cyclization_ok=cyclization_ok,
    )


def exec(program: Program, operator_library=op) -> Chem.Mol:
    """Convenience: return the program's final product (or linear if no cyclization).

    Mirrors ``exec(z, O)`` in the paper (Eq. 2). Returns ``final`` when available,
    else the linear checkpoint.
    """
    result = run(program, operator_library)
    return result.final if result.final is not None else result.linear
