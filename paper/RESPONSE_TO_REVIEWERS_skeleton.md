# Response to reviewers — SKELETON (pre-review)

Drafted at the forward-index revision (tag `lpi-v0.24-forward-index`, branch merged to `main`). This is a
skeleton: the anticipated load-bearing answers are stubbed now while the forward-index work is fresh;
flesh out against the actual reviews when they land. Every number below is reproducible from one command
and locked to the tag (see Appendix A); the full change log is `REVISION_NOTES_forward_index.md`.

---

## Anticipated R1 — "exact reconstruction is only 9/326; is the method just low-accuracy / is this a coverage limit?"

**This is the question the revision answers directly, and the answer is now a proof, not a hedge.**

The beam verifier is *sound but incomplete*: it can only under-count (κ_β ≤ κ), so a low reconstruction
count cannot, by itself, distinguish a true coverage gap from a program the ranking starved. We removed
that ambiguity by running the executor **forward** — enumerating Θ₀ programs to a size bound (N=9, C≤20)
and rendering every one. Forward enumeration is ranking-free, hence **complete by construction up to the
bound**, so:

1. **The demonstrated reach rises from 9 to a sound reach of 11** — the complete search recovers two
   in-grammar cores the beam's carbon-closeness ranking starved (incl. zearalenone, where the beam gave
   κ_β=0 through β=2×10⁵). Search-incompleteness was *exactly two cores*.
2. **The boundary becomes a certified map, not a heuristic score**: a frontier of **117** cores — **83
   provable structural-gaps** (formula-feasible, exhaustively unbuilt up to the bound → each needs a new
   operator) and **34 mass-infeasible** (analytic) — plus **86** larger cores reported as
   bounded-inconclusive (C>20, above the size bound), *never* miscounted as gaps. An empty enumeration
   bucket is a **provable absence certificate** — the one verdict the starving beam categorically cannot
   issue.

**On the three numbers a careful reader will see (8, 9, 11), they are three distinct quantities, not
drift** (§\ref{sec:coverage-structure}): the beam *places* 9; the complete forward search *reaches* 11
(the two beam-starved cores recovered); R_term=8 is the finer *verified-terminal* count, below both because
olivetolic admits κ_O≥2 under the full operator universe.

`make forward-index` reproduces 11 / 117 / 86 from a clean clone (verified).

## Anticipated R2 — "are the structural-gaps real, or artifacts of the search?"

Stub: they are **provable up to (Θ₀, N=9, C≤20)** — complete-by-construction, not heuristic. A timed-out
exec degrades its formula to *inconclusive*, never to a false gap (sound guardrail). The 117 frontier is
also turned into a **monotone build-queue** (`make rung2-cover`: 19 rung-2 / 62 rung-3-open / 34
rung-3-mass), and registering a sound operator can only ADD reach — R_term non-decreasing in Θ, a
*runnable* check (`make forward-index-monotone`). Build-loop step 1 (the PT-naphthalene closure) delivers
it: +3 deposited tetrahydroxynaphthalenes, sound reach 11→14, zero demotions.

## Anticipated R3 — reproducibility / "every number from one command"

Stub: tag `lpi-v0.24-forward-index`; `make forward-index` (the map), `make duality` (Z⋆=D(y;O), **460/460**
distinct formulas across PKS/NRPS/hybrid), `make forward-index-monotone`, `make rung2-cover`. Clean-clone
reproduction confirmed (117/11). Appendix A lists each command.

## Anticipated R4 — "the C>20 cores are unresolved; isn't the map incomplete?"

Stub (and this is the honest, deliberate boundary): the 86 inconclusive cores (C=21…41) are **bounded, not
gaps**. We characterize *why* they resist completion: pinning a target formula removes the carbon-cap
combinatorics, but the residual is the **ordering of the formula-fixed cycle multiset — the per-cycle
program itself**, whose count is astronomical for large highly-reducing cores. This is the
iteration-program combinatorics (the paper's central wall) resurfacing at the structural frontier — the
same degeneracy the §POMDP outlook frames and the pocket lever (citrinin, n=1) **pre-registers** as the
sequel. Completing this tail is future work requiring a state observable, not an enumeration fix. *(A
bounded formula-directed run resolves the tractable subset with sound verdicts; fold its count in here only
if it is a clean one-liner — it does not change the argument.)*

## Proactive disclosures (factual corrections in this revision — stated, not buried)

- **PT-naphthalene**: "skeleton-correct, hydroxyl regiochemistry flagged" → **witness-exact** (round-trips
  the deposited 1,3,6,8-tetrahydroxynaphthalene, BGC0001257/0001258). The earlier status was a stale belief
  the deposited witness refutes; the cores had been excluded partly on it.
- **Resorcylic macrolactone**: "reach unchanged (+0)" → **+2** (the beam-starved cores forward indexing
  surfaces; sound reach 9→11).
- **Macrolactonization**: "not implemented" → **implemented**.

All three are documented with their cause in `REVISION_NOTES_forward_index.md`.
