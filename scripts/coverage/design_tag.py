"""(m_ctrl, m_conf) modality tagging of the 1-edit design neighbourhood -- the corpus-wide test of
Theorem 16.13 (paper Sec. 16.13; companion Results row 7). This is a run, not a pre-registration: the
theorem was already validated on the 6-MSA neighbourhood; here it is banked across all 142 designs.

Theorem 16.13: m_ctrl(e) = modality(f(e)) -- a design edit changes a program feature living in either
  * the GENOME modality   -- domain set / loading-AT starter / extender-AT / release cyclase: all read
    off phi(B), genome-realizable (swap the domain/module);  or
  * the TRAJECTORY modality -- which cycle a *present* domain fires on (per-cycle reduction / C-MeT) or
    the iteration count, under a FIXED domain set: genome-invisible, the iteration-program frontier.
Hence 'engineerable' (structural-only) <=> genome-m_ctrl, and 'frontier' (>=1 control edit) <=> trajectory.

The test is non-vacuous: each edit's modality is derived INDEPENDENTLY from its feature (the kind->modality
map below = Def 16.7b's domain-content-vs-cycle-reprogramming criterion, classified from the biology), then
checked against the Axis label realizability.py assigned by its own reasoning. Agreement across every edit
is Theorem 16.13 holding corpus-wide; a single disagreement -- an edit whose feature is trajectory yet was
labeled structural, or vice versa -- is a real exception worth knowing. Confirms 49 genome / 69 trajectory.
The second column m_conf (product if the edit moves the constitution, else latent) is independent of m_ctrl
(a trajectory-m_ctrl edit that alters an oxidation state is still product-confirmable -- the (0,1) case).
"""
from __future__ import annotations

import collections

from lpi.chem import mol as M
from lpi.executor import core
from lpi.realizability import Axis, design_space, natural_manifold

# Def 16.7b, operationalized by which program feature the edit changes:
#   GENOME     -- domain set / loading-AT / extender-AT / release cyclase  (in phi(B), genome-realizable)
#   TRAJECTORY -- per-cycle firing or iteration count under a FIXED domain set (genome-invisible)
GENOME_KINDS = {"starter", "extender", "add_reductive", "add_cmt", "release"}
TRAJ_KINDS = {"reduction", "c_methyl", "cycle_add", "cycle_remove"}


def edit_modality(kind: str) -> str:
    if kind in GENOME_KINDS:
        return "genome"
    if kind in TRAJ_KINDS:
        return "trajectory"
    raise KeyError(f"unmapped edit kind {kind!r} -- classify it before tagging")


def m_ctrl(r) -> str:
    """Design-level control modality: trajectory if ANY edit is trajectory-modality, else genome."""
    return "trajectory" if any(edit_modality(e.kind) == "trajectory" for e in r.edits) else "genome"


def _product(prog) -> str | None:
    try:
        return M.canonical_smiles(M.strip_stereo(core.exec(prog)))
    except Exception:  # noqa: BLE001 - an edited program whose cyclization cannot fire
        return None


def m_conf(r) -> str:
    """Confirm modality: 'product' if the edit moves the constitutional structure (mass/MS/NMR reads it),
    else 'latent' (confirmable only by the same genome/trajectory channel that controls it)."""
    pt, pd_ = _product(r.nearest.program), _product(r.target)
    if pt is None or pd_ is None:
        return "exec_fail"
    return "product" if pt != pd_ else "latent"


def main() -> None:
    print("Theorem 16.13 corpus-wide test: (m_ctrl, m_conf) tagging of the 1-edit neighbourhood\n")
    d = design_space(natural_manifold())
    eng, fro, spec = d["engineerable"], d["frontier"], d["speculative"]
    print(f"designs: {d['n_designs']}  (engineerable {len(eng)}, frontier {len(fro)}, speculative {len(spec)})")

    # --- per-edit Theorem-16.13 check: feature-modality must equal the axis label, for EVERY edit ---
    exceptions, n_edits = [], 0
    for r in eng + fro + spec:
        for e in r.edits:
            n_edits += 1
            mod = edit_modality(e.kind)
            axis_says = "genome" if e.axis is Axis.STRUCTURAL else "trajectory"
            if mod != axis_says:
                exceptions.append((r.nearest_id, e.kind, e.detail, f"feature={mod}", f"axis={axis_says}"))
    print(f"\nper-edit check over {n_edits} edits: {len(exceptions)} exceptions "
          f"(an edit whose feature-modality disagrees with its structural/control axis)")
    for ex in exceptions:
        print(f"  EXCEPTION: {ex}")

    # --- design-level m_ctrl partition: engineerable == genome, frontier == trajectory ---
    eng_mc = collections.Counter(m_ctrl(r) for r in eng)
    fro_mc = collections.Counter(m_ctrl(r) for r in fro)
    print("\nm_ctrl partition (derived from edit modalities, checked vs verdict):")
    print(f"  engineerable -> {dict(eng_mc)}   (expect all genome = {len(eng)})")
    print(f"  frontier     -> {dict(fro_mc)}   (expect all trajectory = {len(fro)})")
    print(f"  => {eng_mc['genome']} genome / {fro_mc['trajectory']} trajectory")

    # --- (m_ctrl, m_conf) cells over the 118 engineerable+frontier designs (the second column) ---
    cells = collections.Counter((m_ctrl(r), m_conf(r)) for r in eng + fro)
    print("\n(m_ctrl, m_conf) cells over the 118 engineerable+frontier designs:")
    for mc in ("genome", "trajectory"):
        for cf in ("product", "latent", "exec_fail"):
            if cells[(mc, cf)]:
                print(f"  ({mc:10s}, {cf:9s}): {cells[(mc, cf)]}")
    print("  [(trajectory, product) is the live (0,1)-edit cell: frontier to engineer, product to confirm]")

    ok = (len(exceptions) == 0 and eng_mc["genome"] == len(eng) == 49
          and fro_mc["trajectory"] == len(fro) == 69)
    print(f"\n{'='*60}\nTheorem 16.13 REPRODUCES (0 exceptions; 49 genome / 69 trajectory): "
          f"{'YES' if ok else 'NO -- see above'}")


if __name__ == "__main__":
    main()
