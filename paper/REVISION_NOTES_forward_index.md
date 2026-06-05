# Revision notes — forward sound indexing milestone

Branch `paper-revision-forward-index`. Every number below is locked to tag
`lpi-v0.24-forward-index` and re-checked against its artifacts (not the PDF):
`results/forward_index.csv` (Θ₀), `results/forward_index_theta1.csv` (Θ₁),
`make rung2-cover`, `results/duality_regression.log`, `make reachability`.

## What is added (sound/proven, not pre-registered)

**Forward sound indexing.** The inverse beam is sound but *incomplete* (κ_β ≤ κ): zearalenone is
in-grammar at Θ₀ yet the beam reports κ_β = 0 through β = 2×10⁵ (a ranking/starvation artifact, not
missing coverage — widening β changed 0/205 classifications). The fix runs the executor *forward*
(it is sound), enumerating Θ₀ programs to a size bound (N = 9 cycles, C ≤ 20), formula-gating against
the corpus, executing the survivors, and bucketing by structure. Forward enumeration is ranking-free,
hence **complete by construction up to (Θ₀, N=9, C≤20)** — so an empty bucket is a *provable* absence
certificate, which the starving beam categorically cannot give.

Delivered numbers (Θ₀, `make forward-index`):
- **sound reach 11** = 9 beam-reachable + 2 beam-starved cores the forward index recovers (incl.
  zearalenone, where the beam gave κ_β = 0). Search-incompleteness was *exactly two cores*.
- **certified frontier of 117** = 83 provable structural-gaps + 34 mass-infeasible (analytic).
- **86 inconclusive** (C > 20, above the size bound) — bounded, never counted as a gap.
- A **monotone build-loop**: `make forward-index THETA=1` adds the witness-validated PT-naphthalene
  operator → **reach 11 → 14** (+3 deposited THN cores), gaps 117 → 114, **0 demotions** (Prop 16.4c
  as a runnable check, `make forward-index-monotone`). Build-queue: 19 rung-2 / 62 rung-3-open /
  34 rung-3-mass (`make rung2-cover`).
- Duality unchanged and re-confirmed: Z⋆ = D(y;O), **460/460** (`make duality`).

## Factual corrections to the submitted manuscript (transparent, not silent)

These correct claims in the submitted source that the deposited witnesses / the forward index refute:

1. **PT-naphthalene — "skeleton-correct, hydroxyl regiochemistry flagged" → witness-exact.** The
   operator round-trips the deposited 1,3,6,8-tetrahydroxynaphthalene (BGC0001257/0001258, C₁₀H₈O₄).
   The earlier "regiochemistry-flagged" status was a stale docstring belief the deposited witness
   refutes. Consequence: the THN cores had been excluded from `beam._ALL_RELEASES` *partly on that
   stale belief*; the witness arbitrates, and they are now recovered (+3 at Θ₁). **The witness is the
   sounder chemist** — the operator was built witness-first, not from recall.

2. **Resorcylic macrolactone increment — "reach unchanged (+0)" → +2.** The beam does not surface the
   highly-reducing program at practical width, so the submitted text reported the closure as adding no
   reachable core. Forward sound indexing *does* surface it: the closure's true increment is the
   beam-starved set (sound reach 9 → 11). Only the fold-gated rung-3 closures stay open above the size
   bound.

3. **Macrolactonization — "not implemented" → implemented** (registered; reach effect now sorted by
   the forward index rather than left as a tally).

4. **Zearalenone (§POMDP) — "outside the reachable set" → recovered.** Forward indexing recovers it;
   the κ_β = 0 at practical width was a *search artifact*, not missing coverage.

## The PT-closure projection — delivered (in BOTH papers), with over-claims removed

**CORRECTION to my earlier read.** The projection is the **main paper's own** — `section_duality.tex`
L153-154 (which is `\input` into `main.tex`), *and* `companion.tex` L210. My earlier grep checked
`main.tex` directly and missed the `\input` file, so I wrongly reported it as "companion-only." There is
**no PDF/source divergence**: it is the paper's own forecast, now delivered — which makes the strong
framing ("the paper's own projection, now delivered") cleaner than crediting the companion.

The projection read "R_term 8 → 11 (reach 9→12) via PT." The artifacts measure **forward reach**
(recovered), a *distinct, broader* metric than R_term (verified-terminal). The honest reconciliation:

| metric | value | source |
|---|---|---|
| R_term (verified-terminal, Θ₀) | **8** | `make reachability` |
| beam reach (Θ₀) | **9** | `make reachability` |
| forward sound reach (Θ₀) | **11** | `make forward-index` (= 9 beam + 2 beam-starved) |
| forward sound reach (Θ₁, +PT) | **14** | `make forward-index THETA=1` |

Edited in BOTH papers (`section_duality.tex` L153-154 and `companion.tex` L210) to distinguish
R_term=8 (verified-terminal) from forward reach 11 → 14, rather than reasserting "R_term 8 → 11"
(which conflated R_term with reach). The projection's *spirit* — PT lifts the reachable set — is
delivered (+3 via PT: 11 → 14), from a higher base (11, not the forecast 9) because the forward index
recovers the beam-starved cores.

**Two over-claims REMOVED (verify-before-freeze — no artifact backs them):**
- *"they also become beam-reachable, 9→12"* (was in `main.tex` build-loop paragraph): PT is held OUT
  of `beam._ALL_RELEASES`, so there is no beam+PT run; the forward index proves forward-reachability
  (14), not beam-reachability (12). Removed.
- *"R_term lifts to 11 (Θ₁)" / "reach 9→12"* (the projection's literal numbers): `make reachability`
  was not re-run at Θ₁, so R_term=11 and beam-reach=12 are unverified. Recast to the delivered forward
  reach (11 → 14) in both papers.

**Please confirm the metric framing** — it touches the R_term framework; I distinguished the metrics
honestly but did not want to silently reassert unverified Θ₁ numbers.

Also added to `section_duality.tex`: the duality machine-check $Z^\star = D(y;O)$ on **460/460** distinct
product formulas across PKS/NRPS/hybrid (`make duality`).

## C>20 frontier — tractability finding (informs the "86 inconclusive" framing; NOT yet a manuscript claim)

Dry-count (`make forward-index` with `--drycount`, env `CMAX`/`NMAX`): brute forward enumeration is a
super-exponential wall against the carbon cap — C≤20: 6.7M combos; C≤22: 38M (enumerable); C≤24: ≥60M
and climbing ~5–6× per +2 carbons → intractable; C≤80 (needed to close the 214-core map) is hopeless.
Of the 86 inconclusive cores (C = 21…41), a tractable C≤22 run resolves 16; the other 70 need a
**formula-directed** enumerator (search toward the target formulas, not brute force). The "86
inconclusive" claim stays as-is (bounded, never a gap); no C>20 result is asserted until run.

## Pre-registered (future work — never reported as a result)

The unified AF2 fold-bet (rung-3-open regiochemistry + the conformational residue = one fold-gated DOF;
n ≈ 1) remains a pre-registration in §POMDP/Outlook. rung-3-mass (missing enzyme) and rung-4 (soundly
inexpressible) are out of reach. None of this is a result.
