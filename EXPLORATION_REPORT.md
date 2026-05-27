# Exploration report: four extensions beyond v0.15-planner

*Session 2026-05-27. Branch `exploration`. Exploratory code in `scripts/explorations/` (not shipped to
`src/`); no `src/` changes this session, so the full suite is unchanged. Discipline: primary sources
fetched, soundness boundaries named, null/blocked reported as such, no extrapolation from single examples.*

**Bottom line up front.** The prior-favourite (A, the sequence head that could close the 0.57 wall) is
**blocked on sequence data** and produced **no result**. The direction that actually produced a working
new capability is **B (retro-biosynthesis)**. C and D produced **decisive characterizations** (one a clean
negative, one a precise barrier map) rather than capabilities. Because A did not move the 0.57 wall, the
wall stays a wall — so **the paper currently being drafted (triage-with-named-limits) remains the right
paper**; B is a candidate added capability, C and D sharpen the boundary claims.

Ranking by actual results obtained (not prior expectation): **B > C > D > A.**

---

## Direction A — iteration-grammar induction via protein language model

**Attempted.** Audited the ESM-2 head; checked the assembled dataset; live-retried the ClusterCAD
sequence endpoint (path a).

**Achieved (audit, concrete).**
- `src/lpi/model/esm_head.py` is **complete and runnable**: frozen `facebook/esm2_t6_8M_UR50D`,
  mean-pooled KR-domain embeddings, tiny linear head, leave-cluster-out, balanced-accuracy + ROC-AUC vs
  majority/domain-rule baselines.
- The labelled set exists: `data/processed/clustercad_kr_examples.parquet`, **1039 KR modules** (962
  active / 77 inactive; stereo 156 R / 175 S / 708 none) — labels parsed from cached ClusterCAD pages.

**Blocked (the decisive fact).** **0 / 1039 rows have a sequence** (`kr_sequence` all null). The ESM head
embeds sequences; with none, `evaluate_head()` filters to zero rows and trains nothing. The sequences come
from ClusterCAD `…/pks/domainLookup/?domainid=…`, which I **live-probed this session: HTTP 404** — the same
blocker as the original attempt, still dead. So **bacterial test accuracy, fungal transfer accuracy, and
the 0.57 → ? delta are all UNMEASURED.** Reported as blocked, not null.

**Second, separate gap (scope).** Even with sequences, the existing head predicts **bacterial KR stereo /
inactive-domain firing** (leave-cluster-out within bacteria). That is *not* the prompt's experiment
(train bacterial → evaluate the 30 fungal cycles' per-cycle reduction, the 0.57 baseline). The 0.57 probe
lives in `src/lpi/model/differential.py`; wiring a bacterial-ESM predictor into that fungal eval is **new
code that does not yet exist.** So A is two unbuilt steps from a number: (1) sequences, (2) the
bacterial→fungal transfer harness.

**Soundness.** N/A — not reached. The sequence head is correctly framed in-repo as the **unsound learned
layer** over the sound type system; if built, the moat is preserved (predictions are still verified by the
sound executor; the boundary stays typed).

**Paths to the data (not pursued to completion this session, honestly):**
- (a) ClusterCAD endpoint — **dead (404 live).**
- (b) MIBiG loci → NCBI CDS → HMMER (PF08659 KR) → count-matched join. Scaffolded in
  `model/kr_reconstruct.py`; memory records it "doesn't come clean." **The real unblock route, but it is a
  multi-hour data-engineering effort with no guarantee, not a session task.**
- (c) a newer PKS per-domain annotation/sequence resource (e.g. a refreshed ClusterCAD dump or an
  antiSMASH-DB export). **TODO — not surveyed this session; flagged, not claimed.**

**Assessment: highest ceiling, blocked on data — pursue, but only after a dedicated data-assembly effort.**
A is the one direction that could *reframe the paper* (close the wall). It is not abandoned and not
falsified — it is **untested**, gated on path (b) or (c). Do not let "high ceiling" read as "promising
result": there is no result.

**Next steps.** (1) Commit to path (b): assemble ≥100 bacterial KR sequences via NCBI+HMMER, accept it as
a 1–2 day data task. (2) Build the bacterial→fungal transfer harness around `differential.py`'s 30 cycles.
(3) Only then train and report the 0.57 delta, honestly whichever way it lands.

---

## Direction B — retro-biosynthesis under grammar constraints

**Attempted.** Built a retro interface (`scripts/explorations/retro_biosynthesis.py`) that inverts the
**sound** PKS executor via the existing beam verifier over the full PKS operator library, returning a
three-state verdict. Tested on a 12-molecule corpus (positives / negatives / probes).

**Achieved (concrete).**
| category | result |
|---|---|
| positives (known producible cores) | **4/4 VERIFIED** — orsellinic, 6-MSA, mellein, BAB; each recovers a producing program |
| negatives (out-of-grammar chemistry) | **4/4 OUT_OF_GRAMMAR** — limonene, glucose, glycylglycine, caffeine |
| probes (polyketide-shaped, non-manifold) | **2/4 VERIFIED** (C8 linear triketo-acid; propionyl resorcylic acid), 2 OUT_OF_GRAMMAR |

**The catch (and why it matters).** On the first run, **BAB (a known-producible C12 HR core) came back
OUT_OF_GRAMMAR** — a false negative. Diagnosed: at the default `beam_width=2000` the beam **prunes the
producing program**; at `beam_width≥20000` it is recovered (`|Z*|`: 0 → 1). So the beam inversion is
**SOUND (every VERIFIED hit is a real producing program — it is the inverse of a sound function) but NOT
COMPLETE**: a finite beam can miss a program and report a false OUT_OF_GRAMMAR. Therefore the **negative**
verdict is only trustworthy when the target is **formula-infeasible** (the four negatives here — N-bearing
or wrong C:H:O, rejected pre-search) or the search is exhaustive. The corpus above uses the widened beam;
the negatives are sound by formula, not by an exhaustiveness claim.

**Soundness.** Preserved. No unsound shortcuts. The honest framing is **sound positives, heuristic
negatives** — a typed distinction, not a smear.

**On "novel synthetic hits".** The two VERIFIED probes are **grammar-accessible polyketides**. Whether
either is a *novel* compound with *no natural producer* is a **separate database query (TODO)** — not
claimed here. Per the primary-source rule, I will not assert novelty I have not verified against a
natural-products database.

**Scope.** PKS family only. NRPS and hybrid retro need the same enumerate-and-match over their grammars
(the machinery exists: `grammar.enumerate` + structural match) — scoped, not built.

**Assessment: the most promising actual result — pursue as a strong side capability, possibly a paper
subsection.** It is a genuinely new, sound capability (inverting the executor over arbitrary targets) that
fell out of existing machinery. It does not change the core framing; it extends the design contribution.

**Next steps.** (1) Productionize into `src/` with tests (and a completeness bound: report a beam width /
exhaustiveness certificate so OUT_OF_GRAMMAR can be made sound, or label it explicitly heuristic).
(2) Extend to NRPS/hybrid retro. (3) For any VERIFIED non-natural probe, run the novelty check and report
hits with predicted program + required cluster alphabet.

---

## Direction C — terpene synthase grammar

**Attempted.** Fetched the primary source and probed whether the program-space abstraction fits terpenes
(`scripts/explorations/terpene_probe.py`).

**Primary source (fetched).** D. W. Christianson, "Structural and Chemical Biology of Terpenoid Cyclases,"
*Chem. Rev.* 2017, 117(17):11570–11648, doi:10.1021/acs.chemrev.7b00287 (PMID 28841019). Verbatim:
terpenoid cyclases "catalyze the most complex chemical reactions in biology, in that **more than half of
the substrate carbon atoms undergo changes in bonding and hybridization during a single enzyme-catalyzed
cyclization reaction**." Class I monoterpene synthases ionize GPP to a carbocation that cyclizes and
rearranges (hydride shifts, methyl/Wagner-Meerwein migrations, cation-π stabilization) and terminates by
deprotonation or water capture.

**Achieved (concrete).** Mass bookkeeping for the gate case is trivially sound: `GPP → product + PPi`
balances (residual ~0.0000) for all five canonical C10 monoterpenes (limonene, α-/β-pinene, terpinolene,
myrcene). **But all five are C10H16** — identical formula and monoisotopic mass (136.125).

**Finding (the result).** **Program-space fits PKS and does NOT fit terpene cyclization.**
- In PKS, distinct products come from distinct **operator sequences** (a composable program over discrete,
  mass-changing steps) — exactly what the executor enumerates and the verifier infers.
- In terpenes, distinct products come from the **same substrate** via different carbocation steering by the
  enzyme's 3D template. The "operators" (cyclization, hydride shift, Wagner-Meerwein) are **mass-neutral
  isomerizations** of non-isolable carbocations — no per-step observable, no composable inner program. The
  products are **formula/mass-degenerate**, so the architecture's observables (mass, MS/MS) cannot
  disambiguate them either. The abstraction **collapses to a 1:1 synthase→product lookup**: there is nothing
  to enumerate or infer.
- A sound terpene PRODUCT VALIDATOR (per known synthase, net reaction) is achievable; a sound terpene
  PROGRAM SPACE is not. A real terpene predictor needs a per-synthase structural/templating or ML model — a
  different abstraction.

**Soundness.** The net-reaction validator is sound; the absence of a sound composable grammar is the finding,
reported as such (per the prompt: "if you cannot achieve soundness for any non-trivial terpene class, that
is itself the finding").

**Assessment: abandon as an extension; KEEP as a scoping/limitations result.** It is a clean, decisive,
primary-source-grounded negative that sharpens the architecture's boundary — valuable for the paper's
limitations section, not a build direction.

**Next steps.** None as an extension. Cite the finding in §7 as the principled edge of the program-space
abstraction (compositional assembly-line grammars yes; concerted carbocation cascades no).

---

## Direction D — hybrid biosynthesis at non-trivial scale

**Attempted.** Audited the hybrid bridge and tested its reach empirically
(`scripts/explorations/hybrid_probe.py`) against real composed-biosynthesis products.

**Achieved (concrete).**
- The bridge **composes soundly at the minimal typed-handoff level**: a hydrolysis PK-NRPS hybrid
  (acetoacetyl + Gly) renders → `CC(=O)CC(=O)NCC(=O)O`.
- **0 real fungal hybrids reconstruct.** The tetramic-acid release (tenellin's actual chemistry) is a
  **typed, declared, unimplemented** operator → explicit OUT_OF_GRAMMAR (raises `ReleaseNotImplemented`),
  not a silent failure.

**Finding — four specific, typed barriers** (the useful characterization):
1. **Release operator.** Tenellin / desmethylbassianin / aspyridone are tetramic acids / 2-pyridones via a
   Dieckmann (tetramic-acid) release — declared but unimplemented (`IMPLEMENTED_RELEASES = {HYDROLYSIS}`).
   The PK-NRPS backbone reaches the handoff; the cyclative release does not.
2. **Single-residue handoff.** `max_residues = 1`; multi-module NRPS continuations are not enumerated.
3. **Post-assembly tailoring.** Tenellin's mature scaffold needs ring expansion + oxidative/trans-acting
   tailoring — the same tailoring-in-the-latent gap as the PKS diagnostic, compounded on the hybrid backbone.
4. **Meroterpenes need terpenes.** Austinol-class meroterpenes compose a polyketide with a terpene moiety;
   per Direction C there is no sound composable terpene grammar to compose with — blocked upstream.

**Soundness.** Preserved: every failure is an explicit typed operator gap, never a silent or unsound
substitution.

**Assessment: side experiment; the barriers are buildable but each is real work.** "The architecture
composes naturally to handle real complex hybrids" is **FALSE at v0**. "Composition runs into specific named
barriers" is the honest result — and the typed-gap discipline means the architecture **fails legibly**,
which is itself a (smaller) positive.

**Next steps (incremental, ordered by leverage).** (1) Implement the tetramic-acid Dieckmann release — it
single-handedly unlocks the tenellin/aspyridone tetramic-acid class. (2) Lift `max_residues` + add an NRPS
condensation alphabet for multi-residue handoffs. (3) Meroterpenes wait on a terpene abstraction that
Direction C shows the program-space cannot supply.

---

## Overall ranking by actual results

1. **B — retro-biosynthesis.** The only direction that produced a **working, sound new capability**, out of
   existing machinery, with an honestly-bounded limitation (sound positives, heuristic negatives; the BAB
   false-negative caught and diagnosed). Pursue.
2. **C — terpene grammar.** A **decisive, primary-source-grounded finding** (program-space does not fit
   terpenes). High value as a scoping/limitations result; abandon as a build.
3. **D — hybrid scale.** A **precise barrier map** (four typed gaps) and a sound minimal composition.
   Buildable incrementally (tetramic-acid release first); side experiment.
4. **A — sequence head.** **Blocked on sequence data (0/1039), endpoint 404 live; no result.** Highest
   ceiling of the four (it alone could reframe the paper), but ranked last *by results obtained* because
   there are none. Pursue only behind a dedicated data-assembly effort.

**Consequence for the paper.** A did not close the 0.57 wall this session, so the wall stays a wall and the
**triage-with-named-limits framing is still correct** — resume §3 under the current frame. If/when A's data
is unblocked and the wall partially closes, the framing shifts (and §6 gains the learned-predictor footnote
already anticipated); that is a future fork, not a current one.
