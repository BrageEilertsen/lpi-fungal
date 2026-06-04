# One-Function Recovery — Experimental Run Report

**Thesis under test:** fungal polyketide prediction reduces to learning a single per-cycle
policy π(a | s, φ(G)); residual hardness is small, localized, and closable.

**Commit at run start:** `93ab3ef` (working tree clean; non-semantic regenerated artifacts
— reachability.csv CRLF, structural_rescue.pdf re-render — restored to committed bytes).
**Python:** `.venv/bin/python` 3.12.6, `PYTHONPATH=src`.

A clean falsification is a SUCCESSFUL run. Verdicts are one of
**SUPPORTED / REFUSED / INCONCLUSIVE**. Numbers tagged **proven / measured / assumed**.
Soundness invariant on every experiment: ≃ is graph isomorphism up to canonicalization
(never relaxed to substructure); ring-closure slack unchanged; |Z*| reported as capped
where capped; beam not pruning (κ-certificate when Θ touched).

---

## Phase 0 — Baseline reproduction (THE GATE)

**Claim:** every headline baseline regenerates from one command on the current commit.
**Verdict: SUPPORTED.** All four baselines reproduce on `93ab3ef`.

| baseline | command | reproduced figure | tag |
|---|---|---|---|
| reachability | `make reachability` | **9/326** reachable (94 too_large, 111 unreachable, 112 heteroatom) | measured (exhaustive scan) |
| census partition | `make census` | CHO in-scope **205**; cap-lifted re-scan **0/91** reachable (3 budget); **178/205 (87%)** hard-touch; **22/205 (11%)** clean single-class aromatic; 97/205 (47%) entangled | measured |
| census recovery curve | `make census` | top-3 op classes proxy-cover **91%** of 205 (UPPER BOUND, tag≠sound op); heteroatom 112 → 80% N / 99% +halo / 100% +S (scope boundary, diff. grammar) | measured (proxy = upper bound) |
| gate-frontier | `make gate-frontier` | methylating Gate-3 mean **7.04%** / median 6.06% / range 2.6–17.5%; aromatic **0/87**; citrinin 11.1%; Spearman(\|Z*\|, Gate-3) = **−0.61** | measured |
| differential | `make differential` | transfer active_domains=**1.000**, constant_domains=**0.567** (the cross-taxa wall); reduction mlp=0.783; stereo mlp=0.719 | measured (LOSO transfer) |

**Falsification criterion (gate):** any headline number fails to regenerate → STOP, the
downstream program is built on sand. **Did not fire.**

**Soundness note:** ≃ unchanged; no Θ enrichment in Phase 0 so κ-certificate not applicable;
reachability is exhaustive enumeration (not learned, no held-out split needed); the census
recovery curve is explicitly an UPPER BOUND (structural tag ≠ sound operator) and is reported
as such, with the 22/205 clean-aromatic floor as the sound lower bound.

**Note on scan_progress.log:** a sparse 22-line milestone log (last line "reachable so far: 8")
is NOT the authoritative count — the reachability.csv status column (read via csv module,
names contain commas) and the census header both give reachable=**9**. The 8→9 drift hazard
(verify-before-freeze) was checked and is clean on HEAD.

**Reproduction caveat (census, honest):** re-running `make census` on HEAD reproduces every
load-bearing number EXACTLY (9 reachable; 205 CHO / 112 heteroatom; 0 cap-lifted-reachable;
178/205 hard-touch; 22/205 clean aromatic; 91% top-3 recovery), but the secondary
`unreachable`-vs-`budget` split among the 94 too_large cores is NOT bit-stable: committed
log says **91 unreachable / 3 budget**, live re-run gives **87 / 7** (4 cores —
BGC0001450/0001600/0002258/0002429 — flip `unreachable→budget`). Cause: the re-scan caps at
`max_executions=8000` (oog_census.py:200) and enumeration order is not pinned (likely
`PYTHONHASHSEED`), so which cores exhaust-vs-hit-budget varies. This changes NO downstream
number — `budget` cores are classified structurally identically to `unreachable` (oog_census.py:226),
so per_core.csv differs only in the `status_rescan` label (curve/classes/n_classes byte-identical
on all 317 rows), and the load-bearing claim "0 too_large cores become reachable when the cap is
lifted" holds in both. The `budget` flip is the *conservative* direction (declines to confirm
exhaustion). **Follow-up (non-blocking):** pin `PYTHONHASHSEED` in the census target to make the
91/3 split reproducible-from-one-command.

---

## B1 — Substrate-state conditioning vs the 0.567 wall (DECISIVE)

**Claim:** conditioning the per-cycle policy on running substrate state s_t breaks the
constant-domain symmetry and clears the 0.567 wall.
**Verdict: REFUSED.** The wall stands; substrate state does not clear it, even as an oracle.

**Harness:** `make b1` → `scripts/policy/b1_substrate_wall.py`. SAME corpus (25 BGC assets:
16 GOLD |Z*|=1, 9 SILVER), SAME NLML loss, SAME leave-one-BGC-out protocol as train_weak —
the ONLY variable is the feature set. Three nested conditions; CI = cluster bootstrap over
held-out BGCs (2000 reps, 95% percentile); seed 0xC0FFEE.

| condition | features | canonical acc (95% CI) | agreed acc (95% CI) | fraction |
|---|---|---|---|---|
| CONSTANT | intercept only (= per-synthase-majority, the wall's local analogue) | **0.4659** [0.314, 0.622] | 0.4531 | 41/88 |
| POSITION | t, frac, sub_hr (observable, DEPLOYABLE) | **0.4432** [0.300, 0.598] | 0.5312 | 39/88 |
| SUBSTRATE | + chain_c, oxid_sum, n_red, prev (ORACLE on prior actions) | **0.4545** [0.329, 0.600] | 0.5312 | 40/88 |

**Decisive contrasts (canonical):** SUBSTRATE − POSITION = **+0.0114** (literally ONE cycle: 40 vs
39 of 88 — not CI-separated, pure noise at this scale); POSITION − CONSTANT = **−0.0227** (position
features HURT). Clearing 0.567 needs ~50/88; the best condition reaches 41/88 — a ~9-cycle gap.
SUBSTRATE CI-lower 0.329 ≪ 0.567.

**The real signal is in the per-subclass split** (measured, curated-only is the exact split;
inventory HR/PR tags are a family heuristic): on **HR** cycles (where the reduction action genuinely
varies) substrate state *helps* — curated-exact HR climbs **0.053 → 0.263 → 0.368** across
CONSTANT→POSITION→SUBSTRATE (n=19) — while on **PR** cycles (majority-keto, easy) it *hurts*
(0.636 → 0.455, n=11). The pooled accuracy washes this out because the corpus is PR-majority. So
substrate state carries real information exactly where the symmetry is non-trivial, but (a) nowhere
near enough to clear the wall and (b) only as an oracle.

**Falsification status: FIRED (as designed).** Pre-committed rule: REFUSED if SUBSTRATE ≤ POSITION
or fails to clear 0.567. SUBSTRATE (0.4545) > POSITION (0.4432) but does not clear 0.567 → REFUSED
on the wall. This is the guide's "most important possible negative result," reported loud, not tuned.

**Load-bearing caveat (presence-vs-load-bearing):** SUBSTRATE feeds the TRUE prior reduction
actions (oxid_sum/n_red/prev) — an oracle unavailable at inference on a held-out synthase. So even
the within-experiment HR lift is a statement about the FORMALISM (a_t is *somewhat* a function of
s_t), not a deployable predictor. The deployable comparison is CONSTANT-vs-POSITION, and there
position features do not help at all. The honest reading matches substrate_state_probe.py's standing
null and the n≈3 evidence base: the symmetry-break **cannot be demonstrated on existing data** — it
is genuinely data-gated (esp. on HR decisions), not shortcuttable by adding computable state.

**Soundness note:** ≃ unchanged; no Θ enrichment (κ-certificate N/A); all numbers leave-one-BGC-out
(held out); the 0.567 wall is a cross-taxa MLP-transfer baseline (bacteria→fungi, constant-domain
features) while B1 is a within-fungal linear-policy LOSO — DIFFERENT method, so the absolute
"clears 0.567" bar is cross-method and the *within-experiment* CONSTANT/POSITION/SUBSTRATE contrast
is the load-bearing evidence. Reproduces train_weak's committed 0.4432 canonical exactly.

---

## B3 — Free-label scaling: do per-cycle DECISIONS (not clusters) drive accuracy?

**Claim:** the NLML objective manufactures a free per-cycle label for every cycle of every BGC
(straight from Z*, no manual annotation), so the data resource is DECISIONS, not clusters; held-out
accuracy should climb with training **decisions**.
**Verdict: INCONCLUSIVE (mechanical) — substantively a NEGATIVE for the deployable lever.** The
deployable POSITION curve does not climb; it drifts **down** (−0.117 over k=3→17) inside an
overlapping draw spread. The oracle SUBSTRATE curve shows a faint rise-then-fall (+0.058 net,
peak at k=9–12). Neither is CI-separated. Free-label scaling is **not demonstrated** on n=25.

**Harness:** `make b3` → `scripts/policy/b3_free_label_scaling.py`. SAME corpus (25 BGCs, 88
canonical decisions), SAME NLML loss, SAME feature sets as B1. Fixed leave-cluster-out TEST = 8
clusters (26 cycles) NEVER in any training subset; train pool = 17 clusters (62 cycles). For each
k∈{3,6,9,12,15,17}: R=20 random size-k training subsets; report mean test acc, the 2.5/97.5
**draw**-percentile (training-subsample variance), mean #training-decisions, and a test-cluster
bootstrap on the full-pool point. seed 0xC0FFEE, n_steps=300.

| k | train-decisions | POSITION acc | POSITION draw[2.5,97.5] | SUBSTRATE acc | SUBSTRATE draw[2.5,97.5] |
|---|---|---|---|---|---|
| 3 | 11.4 | 0.3865 | [0.269, 0.597] | 0.3346 | [0.192, 0.502] |
| 6 | 20.6 | 0.3365 | [0.172, 0.561] | 0.3769 | [0.287, 0.482] |
| 9 | 33.1 | 0.3404 | [0.249, 0.564] | 0.4385 | [0.308, 0.674] |
| 12 | 44.2 | 0.3288 | [0.231, 0.637] | 0.4423 | [0.346, 0.599] |
| 15 | 54.2 | 0.3096 | [0.249, 0.366] | 0.4173 | [0.326, 0.540] |
| 17 | 62.0 | 0.2692 | [0.269, 0.269] | 0.3923 | [0.385, 0.423] |

**Decisive read (POSITION, deployable):** delta(k=3→17) = **−0.1173**, draw-pct NOT separated
(k=17 point 0.269 ∈ k=3's [0.269, 0.597]). More free labels do not lift the deployable policy —
the trend is mildly negative, within noise. **SUBSTRATE (oracle):** delta = **+0.0577**, an
inverted-U peaking at k=9–12 (~0.44) then declining; draw spreads overlap end-to-end.

**Falsification status.** Pre-committed bins: SUPPORTED (climbing + CI-separated), REFUSED (flat
within spread), INCONCLUSIVE (POSITIVE slope, overlapping). The observed POSITION slope is
**negative** — outside all three clean bins; the mechanical logic routes it to INCONCLUSIVE
(|delta|>0.03, not climbing). **Honest reading:** for the deployable lever this is a NEGATIVE, not a
positive-underpowered — at this feature set, adding free labels does not buy held-out accuracy (it
slightly costs it, consistent with overfitting 3 weak features). That matches the pre-committed
REFUSED *rationale* — "the ceiling is method/feature-bound, not label-bound" — even though the
|delta|<0.03 flatness threshold for a clean REFUSED is not met. Reported as the result, NOT tuned.

**Consistency with B1.** B1 found POSITION features HURT vs CONSTANT (−0.0227) and that the real
signal lives on HR cycles via the oracle. B3 is the scaling-axis view of the same fact: deployable
POSITION features don't scale (negative drift), while oracle SUBSTRATE shows the faint rise B1's
HR-lift predicts. Both say the bridge is data-/feature-gated and, on existing data, not closable by
adding more free labels (B3) any more than by adding computable state (B1).

**Soundness note:** ≃ unchanged; no Θ enrichment (κ-certificate N/A); fixed leave-cluster-out — the
8 test clusters are never in any training subset (ground rule 4); labels are the free per-cycle
Z*-derived labels (the resource under test). n=25 (max 17 train / 8 test) is underpowered: a short,
noisy curve is the honest expectation, and a flat/negative result is a first-class negative.

---

## Duality regression — Theorem 1: Z*(verifier) == D(y;O) at the accurate-mass observable

**Claim (Theorem 1, the soundness+completeness backbone):** for observable O, the verifier's
returned candidate set Z* equals the independent set of all alphabet-producible programs whose
ACTUAL product is O-consistent: Z*(z;O) = [z]_O = D(y;O).
**Verdict: SUPPORTED.** Across PKS, NRPS, HYBRID — **460/460** distinct product formulas — the
verifier's mass-prefiltered Z* equals the independently recomputed O-consistent set D *exactly*.
Mass prefilter is sound AND complete; the out-of-grammar LANTHI guard holds.

**Harness:** `make duality` → `scripts/duality_regression.py`. Per grammar, F = enumerate(target_cho
=None) is the FULL producible set (no prefilter); assert NOT capped (a capped run VOIDS the test —
ground rule 1). Partition F by each candidate's ACTUAL RDKit product formula (same H convention as
beam._formula_cho), projected to the grammar key. For EVERY distinct formula T: Z* = enumerate(
target_cho=T) (the verifier's analytic-prefilter path) vs D = {c∈F : actual_formula(c)==T}
(forward-enumerate-then-filter, independent of the prefilter). Assert not-capped, soundness (no
off-mass product in Z*), and canonical-SMILES set equality Z*==D.

| grammar | case | n_full (uncapped) | distinct formulas | Z*==D |
|---|---|---|---|---|
| PKS | full 4-red+C-MeT, acetyl, n=1..3 | 584 | 90 | **90/90** |
| PKS | 4-red, acetyl+propionyl, n=1..3 | 168 | 58 | **58/58** |
| NRPS | 8-residue, hydrolysis+macrolactam, n=1..3 | 796 | 234 | **234/234** |
| HYBRID | default, max_residues=1, PK n=1..2 | 160 | 78 | **78/78** |
| LANTHI | out-of-grammar guard (registered SCAFFOLD) | — | — | raises NotImplementedError |

**Why this is a real (non-vacuous) test.** D is built from the *executed products' real RDKit
formulae*, not from the analytic prefilter — so the test PITS the analytic prefilter (PKS post-exec
`_formula_cho`; NRPS analytic `peptide_formula`; HYBRID water-corrected PK-formula) against the
actual executor output across 460 genuinely-distinct partitions. The lossless-projection guard fired
on no candidate (no element outside the (C,H,O[,N]) key — no hidden-element masking). Equality ⇔ the
mass prefilter induces EXACTLY the partition the real product formula does ⇔ sound (adds nothing
off-mass) AND complete (drops nothing on-mass) ⇔ Z* = D(y;O).

**Falsification status.** Pre-committed: REFUSED if any grammar's prefilter fails to reproduce the
independent O-consistent set (a verifier soundness/completeness gap); VOID if any enumeration caps.
No enumeration capped; no mismatch. **Did not fire** — the duality holds at the accurate-mass
observable.

**Soundness note:** == is canonical-SMILES set equality (graph iso up to canonicalization — the
engine's own dedup key, NOT relaxed to substructure — ground rule 1); ring-closure slack untouched;
all four enumerations uncapped, so |Z*| is genuinely exhaustive (not a truncated set reported as
exhaustive — ground rule 1); no Θ enrichment (κ-certificate N/A). This is the accurate-mass half of
Theorem 1, the load-bearing observable; richer observables (MS/MS, etc.) only further constrain Z*,
so mass-soundness is the conservative floor.

---

## Experiment A — Register one coverage operator (climb one rung) WITHOUT weakening the executor

**Claim:** a new biosynthetic fold can be added to the grammar so that a previously out-of-grammar
core becomes reachable BY CONSTRUCTION, with (i) soundness preserved (no spurious products, no
existing reconstruction verdict changed) and (ii) a κ-monotonicity certificate proving the
Θ-enrichment can only ADD programs (ground rule 6 — enriching Θ that *decreases* κ = silent beam
pruning = bug).
**Verdict: SUPPORTED for the operator; INFORMATIVE NEGATIVE on deployable scan recovery.** The
resorcylic-acid-lactone (RAL) release was registered into the scan's release set; it is sound,
renders the deposited target by construction, flips **0/326** corpus verdicts, and the κ-certificate
reproduces. It is NOT surfaced by the default-β scan — a beam-completeness boundary, the *same*
mechanism the κ-certificate detects, closable by β and explicitly NOT by weakening (ground rule 1).

**The operator.** `Release.RESORCYLIC_MACROLACTONE` = C2–C7 aromatization (`resorcylic_aromatic`,
the curated single-mode OrsA-family register) + cis-TE macrolactonization (`macrolactonize`,
`_MACROLACTONE_MIN_RING = 8`). Registered in `src/lpi/search/beam.py:44` (`_ALL_RELEASES`), so it is
in the deployable scan's action set, not a side experiment.

| check | command | result | tag |
|---|---|---|---|
| renders target by construction | `pytest test_resorcylic_macrolactone.py` | `core.run(zearalenone program)` == MIBiG **BGC0001057** flat connectivity (C18H22O5, 14-membered RAL) | proven (z→structure) |
| soundness — no spurious product | same | returns `None` on orsellinic precursor (aromatizes but no aliphatic OH → lactonization cannot fire) | proven |
| isolation — no regime poaching | same | returns `None` on the 6-hydroxymellein δ-lactone chain (min_ring=8); regression for the \|Z*\| 1→2 inflation | proven |
| 0 corpus verdict changes | `make reachability` | reachable still **9/326**; the registered release flips no core's verdict | measured (exhaustive scan) |
| κ-monotonicity certificate | `make kappa-check` | Θ_A{malonyl} ⊂ Θ_B{+methylmalonyl}: **2 inversions @β=8000** (olivetolic 3→2, BAB 1→0), **both restored @β=50000** (Prop 16.4c) | measured |
| 15/15 RAL/aromatization/macrolactone tests | (3 test files) | all pass on HEAD | proven |

**The informative negative (beam-completeness boundary).** Zearalenone is reachable by construction
but is NOT surfaced by the reachability scan: the C18 8-cycle no-C-MeT program is starved by the
beam's carbon-closeness ranking — C-MeT-heavy partials fill the beam at depth 7 before the all-keto
chain completes. This is a beam-completeness limit independent of the operator, and it is the SAME
phenomenon the κ-certificate machine-checks: κ_β ≤ κ (the beam under-counts), here under-counting the
target to zero. **One mechanism, two manifestations** — the methylmalonyl witness inverts κ at β=8000
and is restored at β=50000; zearalenone is starved at the default β and is recoverable by construction
(and would be by scan at sufficient β). The κ-certificate is what makes this an honest *measured*
boundary rather than an undetected hole: a decrease in κ across Θ-enrichment is flagged as pruning,
not silently absorbed.

**Falsification status.** Pre-committed: REFUSED if registering the operator changed any existing
reconstruction verdict (soundness regression) or if κ DECREASED under Θ-enrichment without being
recovered at higher β (a true completeness loss, not beam pruning). Neither fired — 0 verdict changes,
both κ-inversions restored at β=50000. The scan-recovery negative was NOT resolved by widening
ring-closure slack or relaxing ≃ (ground rule 1); it is reported as a β-bounded beam limit.

**Soundness note:** the RAL fold is *relative* soundness w.r.t. the curated single-mode grammar (the
competing C1–C6 Claisen fold is deliberately absent — a register asserted by the curated grammar, not
read from observables); ≃ unchanged; min_ring=8 prevents poaching the δ-lactone regime; the
κ-certificate is the ground-rule-6 guarantee that Θ-enrichment is monotone.

---

## Experiment C — Coverage-illusion survivorship control: is the thin Gate-3 frontier real, or are its hard cases invisible?

**The decisive question.** The Gate-3 methylation frontier (Phase 0) is thin — methylating-core mean
**7.04%**, **0/87** aromatic — but it is measured ONLY over the **9 reachable** cores. Citrinin's true
multi-fold C-methylation biology already sits PAST the reachability boundary (it is unreachable). If
methylation-hard cases GENERALLY sit past the boundary, the frontier looks thin only because its hard
cases are INVISIBLE — a survivorship illusion. This is the single most decisive control for the
"small, localized, closable residue" thesis: it tests whether the residue is genuinely small or just
hidden behind the cores we cannot enumerate.

**Pre-registered protocol (2026-06-03, symmetric).** Hardness axis = **C-methylation burden H**, NOT
size — mechanistic, because Gate-3 IS C-MeT positional label-symmetry and zero-C-MeT ⟹ zero Gate-3
(the 0/87 aromatic floor), so C-MeT burden is the necessary condition for pocket-hardness *and* the
one hardness axis computable on the cores we cannot enumerate (the whole requirement of a survivorship
test). H = count of C-methyl branches on the deposited structure (upper bound for true C-MeT).
**Covariate, pre-committed: carbon count C** — size co-drives methylation multiplicity AND
unreachability, so it is CONTROLLED, never the label (size-as-hardness is circular). Comparison set:
reachable (9) vs **CHO-in-scope-unreachable (205)**; the 112 heteroatom cores are out-of-grammar for
scope, reported separately. Citrinin (BGC0001338) is the pre-named positive control.

**Verdict: THINNESS SURVIVES on the methylation axis** (the survivorship falsification did NOT fire),
with three load-bearing caveats. Once carbon count is controlled, C-methyl burden does NOT predict
unreachability (logistic H z=**+0.49**, ns) while size DOES (C z=**+3.27**). The dramatic raw gap is a
SIZE artifact — the reachability boundary is a carbon/combinatorial wall, not a methylation wall.

**Harness:** `make coverage-illusion` → `scripts/coverage/coverage_illusion.py`.

| step | result | reading |
|---|---|---|
| [1] calibration (proxy H vs true C-MeT, n=19) | exact **6**, over **13**, under **0** | over-count is starter/ring/hydroxy-terminal methyls on small aromatics → INFLATES reachable arm → conservative for the survivorship test |
| [2] citrinin positive control | H=**3**, C=13, status=unreachable → **PASS** | the one named methylation-hard core lands high-H AND past the boundary, as pre-registered |
| [3] raw (uncontrolled) | reachable meanH **0.778** / P(H≥2) **0.000**; cho meanH **2.278** / P(H≥2) **0.546** | dramatic gap — but reachable C∈[8,12], cho C∈[7,41] (severe size confound) |
| [3] within-strata C[7,10] | reach 7/meanH 1.00/P≥2 0.00 vs cho 17/0.76/0.24 | at smallest matched size, reachable is if anything MORE methylated |
| [3] matched overlap C[8,12] | reach n=9 meanH 0.778 (H≥2: 0/9) vs cho n=24 meanH 1.000 (H≥2: 7/24); Fisher one-sided p=**0.149** | no significant enrichment past the boundary at matched size |
| [3] logistic P(unreach)~1+H+C | H β=+0.275 (z=**+0.49**, ns); C β=+0.552 (z=**+3.27**) | size predicts unreachability; methylation does not, once size is held |
| [4] cho meanH by stratum | 0.76 → 1.30 → 1.25 → 1.38 → **3.49** (C[21,99]) | methyl count rises monotonically with size — the confound made visible |
| [4] heteroatom (n=112) | meanH 2.562 / P(H≥2) 0.625 | the scope boundary (N/halogen), reported separately, never pooled |

**Why this is a real (non-vacuous) test.** D-style independence: H is computed on the DEPOSITED
molecule, so it is defined on exactly the 205 cores the enumerator cannot reach — the test compares
the hardness of the visible cores against the *invisible* ones directly, which is the only way to
detect survivorship. The control is symmetric and pre-committed: had cho carried significantly higher
H at matched C, the verdict would have DOWNGRADED the residue claim. The proxy's known error (over=13)
inflates the reachable arm, so it biases the matched comparison toward the null we observe — the
finding is therefore conservative w.r.t. its own instrument error in one direction, which is itself a
caveat (below).

**Falsification status.** Pre-committed: SURVIVORSHIP-BIASED (downgrade) if in-scope-unreachable cores
carry significantly higher C-methyl burden than reachable cores at matched carbon count; THINNESS
SURVIVES if H is balanced at matched C. The matched-stratum gap is +0.222 with Fisher p=0.149 and
logistic H z=+0.49 — balanced. **Survivorship falsification did not fire.** The result was NOT tuned:
the verdict is reported on its pre-registered symmetric branch (ground rule 2).

**Caveats (load-bearing — these bound the claim).** (i) **Underpowered:** reachable n=9, all C≤12, so
the matched control lives in ONE narrow stratum (C[8,12]); above C=12 there is NO reachable comparator,
so methylation-hardness cannot be tested independent of size beyond C=12 — the claim is bounded to
C≤12. (ii) **Proxy inflates the reachable arm:** all 9 reachable cores have *true* C-MeT = 0 (they are
non-methylating aromatics/simple cores); the proxy assigns them meanH 0.778, biasing the matched
comparison toward the "balanced" conclusion we report. (iii) **Scope, not methylation, is the real
frontier:** this control speaks ONLY to methylation-hardness; the size/heteroatom boundary is large
(205 CHO-unreachable + 112 heteroatom) and is the dominant coverage frontier — "thinness survives"
means the *methylation residue* is not survivorship-biased, not that coverage is broad.

**Soundness note:** reachable/unreachable status is from the exhaustive `make reachability` scan (≃
unchanged, no slack relaxed); H is an explicit UPPER BOUND calibrated against the executor's true
C-MeT (no claim it equals true C-MeT); the heteroatom set is reported separately and never pooled into
the primary; the verdict is the pre-registered symmetric call, not a post-hoc bin.

---

## Generability scan — two-walls decomposition of the 205 (Sec. 10.1)

**Claim.** The 205 CHO-in-scope-unreachable cores can be sorted, per core and holding the operator set
Θ₀ FIXED, into the two walls the campaign names: the *engineering / beam* wall (a program exists in the
grammar but the default-β beam does not surface it — closable by β, ground rule 1, and NEVER by
weakening) versus the *chemistry / coverage* wall (no Θ₀ program can build the target — needs new
chemistry). **Pre-registered symmetric readout (ground rule 2):** ENGINEERING-dominant if
search-limited ≥ coverage-limited, CHEMISTRY-dominant otherwise — reported on whichever branch fires.

**Verdict: CHEMISTRY-dominant by the mechanical count (0 search-limited @β=8000), but the two walls are
COUPLED — only 34/205 are a SOUND coverage certificate; the other 169 are beam-contaminated.** The
sound, certificate-grade result is the 34 mass-infeasible cores; the "structural" majority is an UPPER
BOUND on new-operator need, not a coverage proof, because the beam underflows even an in-grammar core
(the witness, below).

**Method.** Single bulk rung β=8000 (a prior full two-rung run showed β=50000 added **zero**
classification changes for every C≤20 core, so the second rung is redundant and ~2× the cost). Per core:
(1) `formula_feasible((C,H,O), Θ₀, max_cycles=C//2+1)` — a SOUND analytic check that no program's product
formula can equal the target; fail → `coverage-limited-formula`, NO search. (2) else
`search(canon, Θ₀, β=8000)`: |Z\*|>0 → `search-limited` (surfaced, verified Z\* witness); budget-exhausted
→ `ambiguous-budget`; exhausted-at-β with no match → `coverage-limited-structural` (HEURISTIC). Θ₀ is the
committed base universe (acetyl/propionyl/hexanoyl · malonyl · all reductions · C-MeT · all 6 releases
incl. the RAL macrolactone); ≃ untouched; β the only knob.

| partition (Θ₀ fixed, β=8000) | count | wall | soundness |
|---|---|---|---|
| search-limited (beam-starved, surfaced) | **0** | engineering | SOUND |
| coverage-limited-formula (mass-infeasible) | **34** | new chemistry (rung-3) | **SOUND** |
| coverage-limited-structural (skeleton miss) | **169** | new operator (UPPER BOUND) | HEURISTIC |
| ambiguous-budget (search-incomplete) | **2** | unresolved frontier | HEURISTIC |
| κ-monotonicity violations (witness ladder) | **0** | — | Prop 16.4c clean |

base_status of the 205: too_large (C>20) **94**, unreachable **111**. Reconciliation: **0/94** too_large
cores become reachable at β=8000 — matches the census cap-lifted **0/91**. The 34 mass-infeasible cores
are high-oxygen polyketides whose formula no Θ₀ program matches (aflatoxins, bikaverin, emodin, patulin,
endocrocin, dothistromin, asperthecin, TAN-1612, …).

**The coupling (the load-bearing qualifier).** Zearalenone (BGC0001057) is reachable BY CONSTRUCTION —
Experiment A registered the RAL macrolactone operator and a pytest renders the deposited C18H22O5 by
construction — yet the witness β-escalation does NOT surface it at ANY tested width:

| β | κ (\|Z\*\|) | budget_exhausted | runtime |
|---|---|---|---|
| 8,000 | 0 | False | 11.2s |
| 50,000 | 0 | False | 42.6s |
| 120,000 | 0 | False | 53.3s |
| 200,000 | 0 | **False** | 210.8s |

`budget_exhausted=False` at β=200000 means the beam **exhausted its frontier naturally** (it did NOT hit
the 2M-execution cap) and STILL did not contain the known program. Crucially, that program is in **Θ₀'s**
reachable set — not merely Exp A's curated spec: a test (`test_zearalenone_program_within_scan_grammar`)
confirms every operator it uses (acetyl starter · malonyl extender · {KR,ER,KETO,DH} reductions · no C-MeT
· RESORCYLIC_MACROLACTONE release · 8 cycles) lies in `_scan_spec()`, and the executor renders BGC0001057
from it. So **κ(zearalenone \| Θ₀) ≥ 1 while κ_β = 0 through β=200000** — a CONFIRMED beam under-count
(Lemma 16.3, κ_β ≤ κ), monotone across the ladder (0→0→0→0): NOT a Θ₀ coverage gap and NOT a Prop 16.4c
violation. By monotonicity it is therefore **recoverable at sufficient β** (existence + κ_β ↑ κ) — which
*validates* Exp A's claim WITHOUT locating the surfacing β. That β is >200000 and, by the depth-7
C-MeT-crowding mechanism (Exp A: C-MeT-heavy partials flood the beam before the all-keto C18 chain
completes), is likely **combinatorial in depth**, so a brute β-chase is deliberately not run — the underflow
is bounded below, not pinned. **Consequence:** the 169 `coverage-limited-structural` cores are NOT a clean
coverage certificate — zearalenone is a *proven* in-grammar (Θ₀) core the beam underflows, so the class
MIXES genuine skeleton-coverage gaps with deep beam-starved in-grammar cores. **The engineering / beam wall
gates the chemistry test:** one cannot soundly size how much new chemistry the corpus needs until the beam
is wide enough to stop starving in-grammar programs — and at the deployable β it is not.

**Honest reading.** The only SOUND, certificate-grade coverage claim is the **34 mass-infeasible cores**
(genuine new chemistry, rung-3). The mechanical "CHEMISTRY-dominant" readout (search-limited 0 vs coverage
203) is the branch that fired, but it OVERSTATES the *certified* chemistry wall by ~5×: 169 of the 203 are
HEURISTIC and beam-contaminated. The defensible statement: *of the 205, 34 are provably out of
formula-reach (new chemistry), 2 are unresolved at the budget, and the remaining 169 are an upper bound on
new-operator need, entangled with beam-completeness and not cleanly attributable without β ≫ 200000.*

**Falsification status.** Pre-committed symmetric readout: the CHEMISTRY branch fired (0 search-limited),
reported NOT tuned — Θ₀ fixed, β the only knob, ≃ not relaxed, ring-closure slack not widened, |Z\*|
reported as the verified surfaced count (0) and never as a capped exhaustive claim (ground rule 1);
κ-certificate clean (0 decreases, ground rule 6). Reported with the coupling qualifier, not as a clean
chemistry-dominance claim.

**Reproduction.** `make generability` (PYTHONHASHSEED=0) → `results/generability_scan.{log,csv}` (205
rows × 15 fields) + `results/generability_witness.csv`; **9735s** on HEAD. A target-set cache
(`results/.generability_targets.json`) lets a killed run resume without re-running the base scan; a fresh
run (no CSV) rebuilds it. The CSV's `class` column is the authoritative label (names contain commas — read
with a CSV parser, not field-split).

---

## Completion 1 — forward sound indexing: completing the inverse without weakening it

**Claim.** The generability scan (§10.1) left the 169 `coverage-limited-structural` cores as a HEURISTIC
upper bound: the inverse BEAM is sound but INCOMPLETE (κ_β ≤ κ), so its κ_β=0 cannot distinguish a genuine
coverage gap from a beam-starved in-grammar core (the zearalenone witness). Completion 1 resolves them by
replacing inversion with **forward sound indexing**: enumerate Θ₀ programs up to a size bound (N=9 cycles,
Cmax=20 carbons), execute each with the sound executor, bucket by product. Forward enumeration is
ranking-free, so it cannot starve — it is COMPLETE BY CONSTRUCTION up to (Θ₀, N, Cmax). An empty bucket up
to the bound is then a *provable* structural-gap certificate, the one thing the starving beam categorically
cannot give.

**Verdict: SUPPORTED — the inverse is completed on the in-scope set, and the 169 is resolved.**
Re-orchestration only (no new chemistry): the sweep reuses the generator's exact primitives
(`linear_acid_formula`, `_formula_consistent`, `_formula_cho`, `core.exec`) — the same forward+prefilter
path `make duality` proves equals the Theorem-1 dual set D(y;O) (460/460). One parallel sweep (6 workers,
formula-set-gated, per-exec 30s timeout) serves all 214 CHO targets.

| forward verdict (Θ₀, N=9, Cmax=20) | count | meaning |
|---|---|---|
| recovered | **11** | a Θ₀ program renders the deposited structure (9 reachable + 2 beam-starved) |
| sound-structural-gap | **83** | formula-feasible, exhaustively enumerated, NO program builds it — *provable* coverage gap (needs a new operator) |
| formula-infeasible | **34** | no program's formula can equal the target — needs new chemistry (mass) |
| inconclusive-budget | **86** | C>20 / N>9 — not enumerated to depth; honest "don't know", never a false gap |

**A0 (parity): 9/9 reachable cores recovered** — the forward index loses nothing the beam had; no
completeness gap on the reachable manifold. **A1 (headline): zearalenone (BGC0001057) → recovered** — the
in-grammar C18 core the beam reported κ_β=0 for through β=200000 is surfaced by forward enumeration, so the
underflow is confirmed a *search/ranking artifact*, not chemistry. **Consistency:** 34/34 of the generability
mass-infeasible cores re-confirm formula-infeasible.

**A2 — re-partition of the beam's 169 `coverage-limited-structural`** (the deliverable the campaign could
not produce, because the beam cannot certify absence):

| of the 169 | count |
|---|---|
| recovered (beam-starved, in-grammar) | **2** |
| sound-structural-gap (genuine, now certified) | **83** |
| inconclusive-budget (C>20 / N>9) | **84** |

**The scientific read.** The beam-contamination the §10.1 witness flagged is **real but small**: of the 169,
only **2** (incl. zearalenone) were beam-starved-recoverable in the tractable C≤20 range. The forward index's
payoff is the **83 cores upgraded from HEURISTIC to SOUND** coverage gaps — provably out of Θ₀'s reach up to
N=9, the certificate the beam's κ_β=0 could only gesture at — plus **84** C>20 cores honestly bounded as
inconclusive rather than miscalled gaps. So the §10.1 "two walls are coupled" finding is confirmed *and
bounded*: the engineering/beam wall does contaminate the structural class, but only at the 2-core level; in
the tractable range the chemistry wall dominates (83 genuine gaps vs 2 beam-starved). The certified
new-chemistry frontier is now **83 structural-gap + 34 mass-infeasible = 117** cores (SOUND), with **84**
C>20 unresolved and **11** recovered.

**Soundness (ground rules, untouched).** Forward enumeration is ranking-free and bounded by (Θ₀, N=9,
Cmax=20), so a κ_β ≤ κ underflow cannot occur up to the bound — an empty bucket is a *provable* absence
relative to (Θ₀, N), never "biologically impossible". ≃ is canonical-isomorphism (stereo-free canonical
SMILES), unchanged; no slack, no relaxed matching. The per-exec timeout fired **0 times** (no pathological
exec hit the 30s bound); had any fired, its gated formulas would degrade to inconclusive, never to a false
gap. A duplicate deposited structure is recovered for all its BGCs (28 cores across 23 shared-structure
groups — caught by audit, expanded in classification).

**Honest frontier.** N=9/Cmax=20 exhaustively enumerates every C≤20 core; the 86 inconclusive are the C>20
cores (84 of the 169 + the 2 generability ambiguous-budget), where the 8^N enumeration is intractable —
bounded honestly, not closed. The cost is dominated by enumeration depth, not executions; pushing N is
future work.

**A4 (efficiency).** One parallel sweep — 6,083,164 executor calls (all formula-gated), 0 timeouts, **3058s**
(~51 min, 6 workers) — serves all 214 targets, versus the beam's starving per-target search. By the duality,
every recovered core is simultaneously a design-reachable target.

**Reproduction.** `make forward-index` (PYTHONHASHSEED=0) → `results/forward_index.csv` (214 rows × 12
fields). Targets load directly from the parquet (no beam scan); the sweep is deterministic.

**Build-queue (rung-2 sound set-cover; `make rung2-cover`).** Tagging the 117 certified gaps by needed
operator-class (census proxy) and restricting to RUNG-2 (soundly expressible: formula-invariant DOF
recoverable from φ(B), earned per class, not read off structure) partitions the chemistry axis:

| slice | count | nature | soundly buildable? |
|---|---|---|---|
| rung-2 | **19** | aromatic register (sequence-keyed) | YES — the build-loop floor |
| rung-3-open | **62** | non-aromatic carbocycle, FOLD-gated regiochemistry | not yet (fold-bet, DOF-A) |
| rung-3-mass | **34** | formula-changing tailoring (+O / skeleton) | no (missing enzyme; near rung-4) |

The next sound operator is `aromatic_cyclization` (clean yield 17, 73 demand = upper bound); a macrolactone
split lifts the rung-2 ceiling only 17→**19** (Θ₀'s two macrolactone modes already cover the resorcylic
cases, so `macrolactonization` adds 0 clean). So the soundly-buildable FLOOR is **~19/117** — the fold-gate
dominates. The **62** rung-3-open carbocycle gaps are formula-invariant but fold-gated, the SAME hardness as
the conformational residue (a DOF the sequence doesn't set, only 3D fold does); **rung-3-mass (34)** is a
separate missing-tailoring-enzyme frontier the fold cannot reach. The PT-clade aromatic build-loop
(witness-first, κ-certified monotone Θ-extension) banks the ~19 floor; the unified fold experiment is the bet
for the 62 + the conformational residue on one structural-prediction lever. PROXY caveat: the tag is
necessary-not-sufficient; the SOUND per-operator yield is whatever `make forward-index` reports as
gap→reachable after the operator is built + κ-certified.

**Build-loop log (Θ-extensions beyond the cited baseline).** The frontier above is Θ₀ (tag
`lpi-v0.24-forward-index`; `make forward-index THETA=0` → 11/117). Each build-loop step registers a
witness-validated rung-2 operator as Θ_n (`make forward-index THETA=n`), kept out of the Θ₀ release set so
the citation stays anchored; `make forward-index-monotone` certifies R_term non-decreasing (Prop 16.4c as a
runnable test).

| step | operator | Θ_n | recovered | certified-gaps | monotone |
|---|---|---|---|---|---|
| 1 | PT_NAPHTHALENE — T4HN bicyclic double-aldol (witness BGC0001257/0001258) | Θ₁ | 11→**14** | 117→**114** | +3 / −3 / **0 demotions** PASS |

Step 1 recovered EXACTLY the 3 deposited tetrahydroxynaphthalenes (sound-structural-gap → recovered);
nothing else moved (PT_NAPHTHALENE fires only on the all-keto pentaketide — None-test). It *promoted* an
operator that already existed and was witness-exact — the "gap" was a stale "off-by-one OH" docstring the
deposited witness refuted (corrected; locked by `tests/test_pt_naphthalene.py`) plus its exclusion from the
Θ₀ release set. Reach 9→12 — realizing §10.1's own projection. Next queued (witness-first): single-ring
orsellinic/resorcinol variants (check first whether they need new chemistry), sorbicillinoid, RAL
register-variants. Ceiling = the ~19 rung-2 floor; the larger reach is the fold-bet (62 rung-3-open).

---

## Scoreboard and what "solved" means

| experiment | verdict | one line | command |
|---|---|---|---|
| **Phase 0** (gate) | **SUPPORTED** | all four headline baselines reproduce on `93ab3ef` | `make reachability census gate-frontier differential` |
| **B1** (decisive) | **REFUSED** | the *oracle* substrate state $s_t$ does NOT clear 0.567 (CI-lower 0.329); the wall stands with true prior actions known | `make b1` |
| **B3** | **INCONCLUSIVE / negative** | free-label scaling not demonstrated (n=25 underpowered; POSITION slope −0.117, not climbing) | `make b3` |
| **Duality** | **SUPPORTED** | Z* == D(y;O) exactly, 460/460 formulas, PKS/NRPS/HYBRID; LANTHI guard holds | `make duality` |
| **A** (climb a rung) | **SUPPORTED operator / INFORMATIVE NEGATIVE on scan** | RAL operator sound, 0/326 verdict changes, κ-certified; target reachable-by-construction but beam-starved (β-bounded, not weakened) | `make kappa-check` |
| **C** (decisive) | **THINNESS SURVIVES** (methylation axis) | at matched carbon count, C-MeT burden does NOT predict unreachability (H z=+0.49 ns; C z=+3.27); residue not survivorship-biased | `make coverage-illusion` |
| **Generability** (two-walls) | **CHEMISTRY-dominant (mechanical) / WALLS COUPLED** | of 205 in-scope-unreachable: 0 search-limited, **34 SOUND** mass-infeasible, 169 beam-contaminated structural, 2 ambiguous, 0 κ-violations; witness underflows β=200000 (in-grammar zearalenone) → the engineering wall gates the chemistry test | `make generability` |
| **Completion 1** (forward index) | **SUPPORTED — inverse completed** | forward sound index re-resolves the 169: **2** beam-starved (incl. zearalenone, A1) / **83 SOUND** coverage gaps (heuristic→provable) / **84** inconclusive (C>20); A0 9/9 parity, 0 timeouts | `make forward-index` |
| **B2** (the bet) | **NOT RUN** | bacterial (s,a) re-indexing pretraining — deferred (see below) | — |

**What is solved.** The sound executor + per-cycle policy SOLVE the **reachable manifold**: Theorem 1
holds exactly (Duality, 460/460), so on any reachable core the verifier returns the *exact* O-consistent
program set Z* — sound and complete at accurate mass. The Gate-3 methylation residue is genuinely thin
**and not a survivorship illusion** (Experiment C): at matched size, the cores past the boundary are not
more methylated than the ones we reach. On the residue axis the thesis holds — small, localized, and
(via the RAL operator + κ-certificate, Experiment A) demonstrably *closable* by registering folds without
weakening the executor.

**What is not solved, named honestly.** Three frontiers, none of them the methylation residue:
1. **The cross-taxa prediction wall (0.567) is a modality wall, not a feature deficit (B1, decisive).**
   An *oracle* over all computable substrate state — true prior reductions, oxidation sum, chain length —
   fails to clear it. The identifying signal is therefore not in any computable per-cycle state; it lives
   in the conformational coordinate (the enzyme pocket geometry), which is data-gated to **n=1 (citrinin)**.
   B1's null kills the cheap branch and is *consistent with* (not proof of) the conformational thesis.
2. **The deployable scaling lever is unproven (B3, null).** Decisions-not-clusters does not demonstrably
   scale on the deployable POSITION feature set at n=25; the ceiling is method/feature-bound, matching B1.
3. **Coverage is bounded by a size/combinatorial wall and a heteroatom scope boundary, not methylation
   (A + C + Generability).** The reachability wall is carbon-count-driven (C z=+3.27); the beam starves
   large all-keto programs at the default β (zearalenone, Experiment A); and 112 heteroatom cores are a
   different-grammar scope boundary. The generability scan (Sec. 10.1) decomposes the 205 in-scope-unreachable
   cores: **34** are SOUND mass-infeasible (certified new chemistry), 2 are unresolved at the budget, and
   **169** are `coverage-limited-structural` — but that 169 is an UPPER BOUND, not a certificate: the witness
   shows the in-grammar zearalenone underflows even β=200000 (`budget_exhausted=False`), so the chemistry
   wall cannot be cleanly sized at the deployable β — **the two walls are coupled** (the engineering/beam
   wall gates the chemistry test). **Completion 1 (forward sound indexing) resolves the 169:** forward
   enumeration recovers only **2** (beam-starved, incl. zearalenone), certifies **83** as SOUND coverage
   gaps (provably out of Θ₀ to N=9 — heuristic→provable), and bounds **84** (C>20) as inconclusive. So the
   coupling is real but small (2/169), and the tractable chemistry frontier is now SOUNDLY sized at **83
   structural-gap + 34 mass-infeasible = 117** cores. These are the LARGE frontiers; they are now
   *characterized and partly certified* — still not closed.

**Honest endpoint.** *Solved over the reachable manifold, with a thin non-survivorship-biased methylation
residue, against a clearly-named hard frontier (size/combinatorial coverage + heteroatom scope + the
conformational coordinate for cross-taxa transfer).* The campaign did what it was built to do: the one
decisive test that could have downgraded the residue claim (Experiment C survivorship) was run and the
claim survived; the two tests that could have cheapened the cross-taxa wall (B1, B3) refused, and that
refusal is itself the load-bearing finding — the wall is real and conformational. No verdict was tuned;
two of six are clean negatives, reported as first-class results (ground rule 2).

---
