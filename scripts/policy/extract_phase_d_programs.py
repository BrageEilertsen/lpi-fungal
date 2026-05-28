"""Phase D1: program extraction for the 16 GOLD/SILVER inventory clusters.

For each cluster with |Z*| at alphabet+mass in [1, 10], re-run the engine to capture
the full candidate set (the Z* programs). The marginal-likelihood objective in Phase
D2 will use Z* as the supervision target: a cluster with |Z*|=k contributes k
programs to the loss sum, with the model learning to put mass on any one of them.

Adds Brage's structural cheat sheet from PHASE_D_PREP.md as authoritative biological
annotations + caveats per cluster. Out-of-grammar biology (e.g. strobilurin's benzoyl
starter under our acetyl-only inventory grammar) is flagged in ``caveats``.

Output: data/policy/ground_truth_programs.json -- a per-BGC dictionary mapping
bgc_id -> {compound, formula, candidates: [{starter, cycles, release, smiles, score},
...], chemistry_notes, caveats}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem.program import ReductionState as Rs, Release  # noqa: E402
from lpi.search.generate import Alphabet, generate  # noqa: E402

INVENTORY_PARQUET = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_JSON = ROOT / "data" / "policy" / "ground_truth_programs.json"

# Same grammar the inventory used so candidate counts match.
GRAMMAR = Alphabet(
    reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
    releases=(Release.HYDROLYSIS, Release.ALDOL_AROMATIC, Release.LACTONIZATION),
    starters=("acetyl",),
    allow_cmet=True,
)

# Per-BGC starter overrides. The inventory ran with acetyl-only for tractability, but
# specific BGCs use a non-acetate starter in vivo. Updating the grammar boundary
# condition (NOT the data point) lets the verifier-grounded inverse compiler self-
# correct to the biologically authentic program space.
BGC_STARTER_OVERRIDES = {
    "BGC0001909": ("benzoyl",),  # strobilurin A: phenylalanine -> benzoyl-CoA starter
    "BGC0002065": ("benzoyl",),
}

# Carbons contributed by each starter (the loaded acyl group; the thioester -S* sentinel
# is dropped on TE hydrolysis and replaced by -OH).
STARTER_CARBONS = {
    "acetyl": 2,
    "propionyl": 3,
    "butyryl": 4,
    "hexanoyl": 6,
    "benzoyl": 7,
}


def grammar_for(bgc: str) -> Alphabet:
    """Return the candidate grammar for ``bgc``, applying per-BGC starter overrides."""
    starters = BGC_STARTER_OVERRIDES.get(bgc, ("acetyl",))
    return Alphabet(
        reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
        releases=(Release.HYDROLYSIS, Release.ALDOL_AROMATIC, Release.LACTONIZATION),
        starters=starters,
        allow_cmet=True,
    )


def cycle_bounds_for(C: int, starters: tuple[str, ...]) -> tuple[int, int]:
    """Compute (lo, hi) cycle range for a target carbon count + starter set.

    Each extension cycle adds ~2 carbons (malonyl); C-MeT adds +1 per fire (modest). The
    bounds bracket the n_est = (C - starter_C) / 2 prediction with +/- slack.
    """
    sc_max = max(STARTER_CARBONS.get(s, 2) for s in starters)
    sc_min = min(STARTER_CARBONS.get(s, 2) for s in starters)
    lo = max(1, (C - sc_max) // 2 - 1)
    hi = min(10, (C - sc_min) // 2 + 2)
    return lo, hi

# Brage's structural cheat sheet from PHASE_D_PREP.md / the cheat-sheet message.
# Each entry tags the canonical biology + any caveats that affect the engine match.
CHEMISTRY_NOTES = {
    "BGC0001121": dict(
        family="orsellinic_acid",
        notes="C8 tetraketide; starter acetyl + 3 malonyl; all KETO; C2-C7 aldol "
              "aromatization. The non-reducing baseline.",
    ),
    "BGC0002212": dict(family="orsellinic_acid", notes="same chemistry as BGC0001121"),
    "BGC0002595": dict(family="orsellinic_acid", notes="same chemistry as BGC0001121 "
                                                       "(formula_source: smiles_recovery)"),
    "BGC0001275": dict(
        family="6_methylsalicylic_acid",
        notes="C8 tetraketide; starter acetyl + 3 malonyl; cycle-2 KR (one ketoreduction); "
              "C2-C7 aldol + aromatization expels the cycle-2 OH (no programmed DH).",
    ),
    "BGC0001276": dict(family="6_methylsalicylic_acid", notes="same chemistry as BGC0001275"),
    "BGC0001606": dict(
        family="gibepyrone_A",
        notes="Pyrone from F. graminearum PKS8. Pentaketide-ish core; LACTONIZATION "
              "release (alpha-pyrone). C10H12O2 implies extensive reduction.",
        caveats=["alpha-pyrone topology may not match our LACTONIZATION operator exactly"],
    ),
    "BGC0002191": dict(
        family="prolipyrone_B",
        notes="Same F. graminearum PKS8 architecture as gibepyrone-A; partially-reducing "
              "pentaketide. C10H10O5 has more oxygens than gibepyrone-A so fewer reductions.",
        caveats=["formula_source: smiles_recovery"],
    ),
    "BGC0002852": dict(
        family="3_6_dialkyl_alpha_pyrone",
        notes="Clean alpha-pyrone model. Possible C-MeT events on the backbone (SAM-driven); "
              "engine grammar has allow_cmet=True so this is in scope.",
    ),
    "BGC0001338": dict(
        family="citrinin",
        notes="C13H14O5; classical PR pentaketide core with extensive post-PKS tailoring "
              "(extra methylations, oxidations). Engine returns the PKS core; the tailoring "
              "step is NOT in the Phase D supervision target.",
        caveats=["literature program extends beyond the PKS core to the tailored product"],
    ),
    "BGC0000161": dict(
        family="isoterrein",
        notes="C8 cyclopentenone-based core derived from an aromatic hexa/pentaketide via "
              "oxidative ring contraction. The PKS core ladder is what we enumerate; the "
              "ring contraction is post-PKS.",
        caveats=["post-PKS oxidative ring contraction is out of grammar; supervision target "
                 "is the pre-contraction core"],
    ),
    "BGC0002180": dict(
        family="asperlin",
        notes="(+)-asperlin C10H12O5; hexaketide-derived fatty-acid-like chain with "
              "lactonization. Reduction state alternates between KETO and ER sections.",
        caveats=["epoxide + acetate moieties are post-PKS tailoring; supervision target is "
                 "the linear PKS core (formula_source: smiles_recovery)"],
    ),
    "BGC0001273": dict(
        family="asperlactone",
        notes="C9H12O4; related fatty-acid-derived lactone to asperlin, smaller chain. "
              "Mixed reduction states.",
    ),
    "BGC0001909": dict(
        family="strobilurin_A",
        notes="C16H18O3. Phenyl ring + diene + methoxy ester. Phenyl ring derived from "
              "phenylalanine -> benzoyl-CoA in vivo. STARTER_OVERRIDE applied: re-enumerated "
              "with starters=(benzoyl,); cycle bounds shifted to match the +7-carbon "
              "contribution of the benzoyl starter. Inventory's |Z*|=7 under acetyl is "
              "superseded by the new, biologically authentic candidate set.",
        caveats=["formula_source: smiles_recovery"],
    ),
    "BGC0002065": dict(
        family="strobilurin_A",
        notes="same chemistry as BGC0001909; STARTER_OVERRIDE applied",
        caveats=["formula_source: smiles_recovery"],
    ),
    "BGC0002238": dict(
        family="dihydroxy_methoxypropiophenone",
        notes="C10H12O4. Salicyclic-like aromatic core with propanone side chain.",
        caveats=["formula_source: smiles_recovery"],
    ),
    "BGC0003109": dict(
        family="stachysalicyloid_C",
        notes="C16H20O2. Salicylic acid derivative; similar to 6-MSA / orsellinic family "
              "but with extended side chain (likely from C-MeT events or a longer extender).",
        caveats=["audit the cyclization chemistry against the 6-MSA-class operator"],
    ),
}


def program_to_dict(prog) -> dict:
    return dict(
        starter=prog.starter,
        cycles=[dict(reduction=c.reduction.value, c_methyl=bool(c.c_methyl),
                     extender=c.extender.value) for c in prog.cycles],
        release=prog.release.value,
        n_cycles=prog.n_cycles,
    )


def extract_candidates(bgc: str, C: int, H: int, O: int,
                       cycle_lo: int, cycle_hi: int) -> tuple[list[dict], tuple[str, ...], tuple[int, int]]:
    """Re-run generate() and return the full Z* candidate set.

    Honors per-BGC starter overrides: when a starter override is in effect, the
    cycle bounds are recomputed from the override's starter-carbon contribution so
    the search space matches the new biology rather than the acetyl assumption.
    """
    grammar = grammar_for(bgc)
    if bgc in BGC_STARTER_OVERRIDES:
        cycle_lo, cycle_hi = cycle_bounds_for(C, grammar.starters)
    res = generate(grammar, cycle_lo, cycle_hi, target_cho=(C, H, O))
    cands = [dict(smiles=c.smiles, program=program_to_dict(c.program),
                  score=float(c.score)) for c in res.candidates]
    return cands, grammar.starters, (cycle_lo, cycle_hi)


def main() -> None:
    df = pd.read_parquet(INVENTORY_PARQUET)
    # The 16 GOLD/SILVER targets: |Z*| in [1, 10] (i.e. GOLD_unique + all SILVER tiers).
    targets = df[df.z_star_alphabet_plus_mass.between(1, 10)].copy()
    targets = targets.sort_values(["z_star_alphabet_plus_mass", "bgc_id"]).reset_index(drop=True)
    print(f"Extracting programs for {len(targets)} GOLD/SILVER clusters...\n")

    out: dict[str, dict] = {}
    for _, r in targets.iterrows():
        bgc = r.bgc_id
        cands, used_starters, used_bounds = extract_candidates(
            bgc, int(r.C), int(r.H), int(r.O),
            int(r.cycle_lo), int(r.cycle_hi))
        meta = CHEMISTRY_NOTES.get(bgc, {})
        starter_note = ""
        if bgc in BGC_STARTER_OVERRIDES:
            starter_note = f" [STARTER_OVERRIDE -> {','.join(used_starters)}]"
        out[bgc] = dict(
            bgc_id=bgc,
            compound_name=r.compound_name,
            formula=r.formula,
            formula_source=r.formula_source,
            C=int(r.C), H=int(r.H), O=int(r.O),
            mass=float(r.mass),
            tier=r.tier,
            z_star_size_inventory=int(r.z_star_alphabet_plus_mass),
            z_star_size=len(cands),  # actual; may differ from inventory after override
            grammar_starters=list(used_starters),
            cycle_lo=used_bounds[0],
            cycle_hi=used_bounds[1],
            inventory_cycle_lo=int(r.cycle_lo),
            inventory_cycle_hi=int(r.cycle_hi),
            candidates=cands,
            family=meta.get("family"),
            chemistry_notes=meta.get("notes", "(no annotation)"),
            caveats=meta.get("caveats", []),
        )
        flag = " <-- caveat" if meta.get("caveats") else ""
        print(f"  {bgc}  C{r.C:>2}H{r.H:>2}O{r.O:>2} "
              f"inv|Z*|={int(r.z_star_alphabet_plus_mass):>2} "
              f"new|Z*|={len(cands):>2}  "
              f"{meta.get('family', '?'):<30} {starter_note}{flag}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} -- {len(out)} BGCs")

    # Summary statistics for the prep doc.
    print()
    print(f"  total BGCs:                {len(out)}")
    print(f"  total candidate programs:  {sum(len(v['candidates']) for v in out.values())}")
    print(f"  unique chemistries:        {len({v['family'] for v in out.values()})}")
    print(f"  BGCs with caveats:         {sum(1 for v in out.values() if v['caveats'])}")
    print()
    print("Caveat breakdown:")
    for bgc, v in out.items():
        for cav in v["caveats"]:
            print(f"  {bgc} ({v['family']:<22}): {cav}")


if __name__ == "__main__":
    main()
