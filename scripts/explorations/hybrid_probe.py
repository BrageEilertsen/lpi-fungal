"""EXPLORATORY (Direction D): does the hybrid bridge compose to non-trivial real hybrids?

The bridge (lpi.executor.hybrid) composes z_PKS o handoff o z_NRPS o release. This probe tests its
reach empirically against the demands of real composed-biosynthesis products and characterizes the
barriers (the finding the prompt asks for either way).

Real targets considered (MIBiG): tenellin (BGC0001049, fungal 2-pyridone/tetramic-acid PKS-NRPS),
desmethylbassianin (BGC0001136), aspyridone A (BGC0000959) -- all tetramic-acid/pyridone PKS-NRPS;
and meroterpenes (austinol class) -- polyketide + terpene composition.
"""
from __future__ import annotations

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState, Release
from lpi.executor import hybrid as hyb


def main() -> None:
    print("Direction D -- hybrid bridge reach (empirical) + barrier characterization\n")

    # PK sub-program: acetyl + one keto extension -> linear acid (the handoff checkpoint).
    pk = Program("acetyl", (Cycle(ReductionState.KETO),), Release.HYDROLYSIS)

    # (1) minimal hydrolysis hybrid: PK acid + glycine handoff -> renders?
    try:
        mol = hyb.run(hyb.Hybrid(pk, ("Gly",), hyb.HybridRelease.HYDROLYSIS))
        print(f"(1) hydrolysis PK-NRPS (acetoacetyl + glycine): RENDERS -> {M.canonical_smiles(mol)}")
    except hyb.HybridError as e:
        print(f"(1) hydrolysis hybrid FAILED: {e}")

    # (2) the SAME backbone with the tetramic-acid (Dieckmann) release -- tenellin's actual release.
    try:
        hyb.run(hyb.Hybrid(pk, ("Gly",), hyb.HybridRelease.TETRAMIC_ACID))
        print("(2) tetramic-acid release: RENDERS (unexpected)")
    except hyb.ReleaseNotImplemented as e:
        print(f"(2) tetramic-acid release: OUT-OF-GRAMMAR (typed, unimplemented) -- {e}")

    print("""
FINDING -- the bridge composes for the MINIMAL case but does NOT reach the real fungal hybrids; the
barriers are specific and typed:

 BARRIER 1 (release operator). Tenellin / desmethylbassianin / aspyridone are tetramic acids / 2-pyridones
   formed by a Dieckmann (tetramic-acid) release, which is a DECLARED-but-UNIMPLEMENTED operator
   (raises ReleaseNotImplemented above). The PK-NRPS backbone reconstructs to the handoff; the terminal
   cyclative release does not. This is the dominant barrier for the tetramic-acid PKS-NRPS class.

 BARRIER 2 (single-residue handoff). The grammar's max_residues defaults to 1 (single-amino-acid handoff).
   Multi-module NRPS peptide continuations are not enumerated; real multi-residue hybrids are out of reach
   without lifting that bound (and an NRPS module/condensation-domain alphabet).

 BARRIER 3 (post-assembly tailoring). Tenellin's mature scaffold involves a ring expansion (acyl-pyridone)
   and oxidative/trans-acting tailoring beyond the assembly line -- the same tailoring-in-the-latent gap
   the PKS diagnostic already documents, now compounded on the hybrid backbone.

 BARRIER 4 (meroterpenes need terpenes). Austinol-class meroterpenes compose a polyketide with a TERPENE
   moiety. Per Direction C, the terpene cyclization does not fit the program-space abstraction at all, so
   meroterpene composition is blocked upstream -- not by the bridge, but by the absence of a sound,
   composable terpene grammar to compose WITH.

So: the architecture COMPOSES soundly at the typed-handoff level (minimal hydrolysis hybrid renders), and
every failure is an EXPLICIT typed gap (a named missing operator), not a silent failure -- which is the
honest, useful characterization. But "handles non-trivial real hybrids" is FALSE at v0: the tetramic-acid
release, multi-residue handoff, post-assembly tailoring, and terpene composition are all unbuilt.
""")


if __name__ == "__main__":
    main()
