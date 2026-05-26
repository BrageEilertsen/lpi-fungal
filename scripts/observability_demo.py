"""Track 2 candidate-collapse benchmark: the three-state verdict as an experiment planner.

The whole thesis, operationally: a grammar gives possibility; an accurate mass (adduct-aware,
ppm-tolerant) collapses it; MS/MS disambiguates the residual; and the verdict says what to do next.
The same observability stack runs over PKS and NRPS candidates -- only the grammar changes.

The mass rung is a sound Z*_eps (mass-window formula prefilter), not an exact verifier with a
cosmetic tolerance. Where the full program space is combinatorial (highly-reducing), the genome-only
count is reported as a censored lower bound; the mass-conditioned verdict is still sound. A
sensitivity sweep shows the collapse is stable across realistic mass tolerances and that MS/MS
resolution is threshold-transparent (not a hand-picked cutoff).
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.engine import contains, frag_fingerprint
from lpi.grammars import NRPS, PKS
from lpi.observe import ADDUCTS, MSObservables, exact_mass, infer_ms, ion_mz

RDLogger.DisableLog("rdApp.*")

_ORS = "Cc1cc(O)cc(O)c1C(=O)O"
_OLI = "CCCCCc1cc(O)cc(O)c1C(=O)O"
_DKP = "O=C1CNC(=O)CN1"
_PHETYR = "O=C1NC(Cc2ccc(O)cc2)C(=O)NC1Cc1ccccc1"

_NOISE = (33.3, 71.7, 150.5)   # deterministic in-source / contaminant peaks added to each spectrum
_TAU = 0.7                      # MS/MS modified-cosine threshold (see the sweep for sensitivity)

# label, grammar, domains, truth SMILES, observed adduct, search adducts, (min,max), kind
CASES = [
    ("orsellinic acid", PKS, {"KS", "AT", "ACP", "PT", "TE"}, _ORS, "[M-H]-", ("[M-H]-",), (3, 3), "normal"),
    ("olivetolic acid", PKS, {"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"}, _OLI, "[M+H]+", ("[M+H]+",), (3, 6), "normal"),
    ("cyclo(Gly-Gly)", NRPS, {"C", "A", "T", "TE"}, _DKP, "[M+H]+", ("[M+H]+",), (2, 2), "normal"),
    ("cyclo(Phe-Tyr)", NRPS, {"C", "A", "T", "TE"}, _PHETYR, "[M+H]+", ("[M+H]+",), (2, 2), "normal"),
    ("mass withheld", NRPS, {"C", "A", "T", "TE"}, _DKP, None, (), (2, 3), "withhold"),
    ("wrong mass (+50 Da)", PKS, {"KS", "AT", "ACP", "PT", "TE"}, _ORS, "[M-H]-", ("[M-H]-",), (3, 3), "wrongmass"),
    ("wrong adduct (obs Na, search H)", PKS, {"KS", "AT", "ACP", "PT", "TE"}, _ORS, "[M+Na]+", ("[M+H]+",), (3, 3), "wrongadduct"),
]


def _fmt(n):
    return "  -" if n is None else str(n)


def run_table():
    hdr = f"{'case':33s} {'gram':5s} {'genome':>8s} {'+mass':>6s} {'+MSMS':>6s} {'rank':>5s}  verdict / next observable"
    print(hdr)
    print("-" * len(hdr))
    for label, grammar, domains, smi, obs_add, search_add, (lo, hi), kind in CASES:
        alpha = grammar.alphabet_from_domains(domains)
        give_msms = kind in ("normal",)
        ladder = kind in ("normal", "withhold")
        if kind == "withhold":
            ms = MSObservables()
        elif kind == "wrongmass":
            mz = ion_mz(exact_mass(smi) + 50.0, ADDUCTS[obs_add])
            ms = MSObservables(observed_mz=mz, adducts=search_add, ppm=5)
        else:  # normal / wrongadduct
            mz = ion_mz(exact_mass(smi), ADDUCTS[obs_add])
            peaks = (tuple(frag_fingerprint(smi)) + _NOISE) if give_msms else None
            ms = MSObservables(observed_mz=mz, adducts=search_add, ppm=5, msms_peaks=peaks, msms_tau=_TAU)
        res = infer_ms(alpha, ms, grammar, lo, hi, ladder=ladder)
        genome = res.ladder["genome"]
        genome_s = (f">={genome}" if res.censored and genome is not None else _fmt(genome))
        rank = contains(res, smi) if kind not in ("wrongmass",) else None
        tail = res.state.value.upper() + (f"  (next: {res.next_observable})" if res.next_observable else "")
        print(f"{label:33s} {grammar.name:5s} {genome_s:>8s} {_fmt(res.ladder['+mass']):>6s} "
              f"{_fmt(res.ladder['+msms']):>6s} {str(rank):>5s}  {tail}")


def run_sweep():
    print("\nSensitivity sweep on olivetolic acid (PKS, [M+H]+):")
    alpha = PKS.alphabet_from_domains({"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"})
    mz = ion_mz(exact_mass(_OLI), ADDUCTS["[M+H]+"])
    print("  mass tolerance (no MS/MS) -- collapse is stable across realistic ppm:")
    for ppm in (2, 5, 10, 20):
        r = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=ppm), PKS, 3, 6, ladder=False)
        print(f"    {ppm:2d} ppm -> +mass = {r.ladder['+mass']}  ({', '.join(r.formulas)})")
    print("  MS/MS threshold (5 ppm + noisy spectrum) -- MS/MS resolves the residual:")
    peaks = tuple(frag_fingerprint(_OLI)) + _NOISE
    for tau in (0.3, 0.5, 0.7, 0.9):
        r = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5,
                                          msms_peaks=peaks, msms_tau=tau), PKS, 3, 6, ladder=False)
        print(f"    tau={tau} -> +MSMS = {r.ladder['+msms']:3d}  [{r.state.value}]")


def main():
    print("Track 2 -- metabolomics observability: candidate collapse + three-state verdict\n")
    run_table()
    run_sweep()


if __name__ == "__main__":
    main()
