# Exploration report A: the ESM-2 sequence head (Direction A, unblocked)

*Session 2026-05-27, branch `exploration`. The ClusterCAD endpoint was reported alive; this session
assembled the sequence dataset, trained/evaluated the head, and measured the 0.57 fungal transfer.
Discipline: endpoint verified, sample sequences checked against source, the obvious confound (homology
leakage) measured, the structural limit named rather than papered over.*

**Headline (two findings, not one):**
1. **Bacterial (modular): the hypothesis is CONFIRMED.** The frozen-ESM-2 sequence head extracts
   per-module chemistry signal that gene content cannot — inactive-KR detection **0.860 balanced acc /
   0.921 AUC** and KR stereo **0.849** vs **0.50–0.53** baselines — *and these survive a close-homolog
   (70%-identity) -partitioned re-evaluation (0.850 / 0.910 / 0.855); remote-homolog generalization is
   untestable here because KR domains are a single conserved family (see Stage 3b).*
2. **Fungal (iterative): the 0.57 wall is UNMOVED, and structurally so.** A per-domain sequence head
   cannot close the fungal per-cycle wall, because a fungal iterative synthase presents **one KR sequence
   across all its cycles** — the per-cycle signal is not in the (constant) sequence. `0.57 → 0.57`.

So the paper's triage-with-named-limits framing **stands, with a sharper mechanistic reason** for the
wall. The bacterial result is real but is a *different* result (modular per-module prediction), not the
fungal-wall closure that would have reframed the paper.

---

## The original blocker was a missing header, not a dead endpoint

`GET /pks/domainLookup?domainid=...` returns Django `404 Not Found` to a plain request but **200 + JSON
to a request carrying `X-Requested-With: XMLHttpRequest`** (the view is AJAX-only; jQuery adds that header
automatically, which is why a browser worked and the scraper "permanently 404'd"). Verified live on the
same id: 404 plain / 200 with the header. Fixed in `src/lpi/model/esm_data.py`.

## Stage 1 — domain ID discovery: already done
The labelled parquet `clustercad_kr_examples.parquet` (1039 KR modules) already carries the ClusterCAD
domain ids in `kr_domainid` (parsed from cached cluster-page `data-domainid` attributes in a prior
session). No discovery needed.

## Stage 2 — sequence retrieval: complete and verified
Polite scrape (≥1 s/request, per-id cache, backoff): **1039 / 1039 fetched, 0 failures.** Wrote
`data/processed/clustercad_kr_sequences.parquet`; joined into `clustercad_kr_examples.parquet`
(**1039/1039 rows now have a sequence**). **Verified 3 ids against the live endpoint:** live re-fetch ==
cached, valid amino acids, and the `name` field matches the expected cluster (id 19773 → "Abyssomicin
subunit abyB1 module 1: KR domain" ↔ parquet cluster BGC0000001.1). Sequences are real KR domains
(~175–180 aa, Rossmann `GTVLVTGGTGALGAL…` NAD(P)-binding motif).

## Stage 3 — bacterial ESM-2 head: strong signal, with a homology caveat

`evaluate_head()` (frozen `esm2_t6_8M_UR50D`, mean-pooled, tiny linear head, leave-cluster-out):

| target | n | head | AUC | majority | domain-rule |
|---|---|---|---|---|---|
| inactive-KR detection (did the present KR fire) | 1039 | **0.860** bal-acc | **0.921** | 0.500 | 0.500 |
| KR stereochemistry (R/S) | 331 | **0.849** acc | — | 0.529 | 0.529 |

Both targets are ones the domain cartoon **provably cannot** do (baselines at chance). The head clears
them decisively — sequence carries the "did it fire" and the stereo signal that gene content discards.
The prior domain-only MLP was at chance on stereo (~0.72 ≈ its majority); the sequence head is at 0.85.

**CAVEAT (measured, load-bearing).** Leave-cluster-out controls *module* leakage but **not
sequence-homology leakage across clusters**: of 1039 rows there are **952 unique sequences; 164 rows are
exact-duplicate sequences (77 groups, 63 of them spanning >1 cluster); 16% of rows share a cross-cluster
40-residue prefix.** Homologous KR domains in related clusters can therefore land a near-copy in both
train and test, **inflating these numbers by an unmeasured amount.** The 0.86/0.85 are **upper bounds** until the homology-partitioned re-evaluation below.

## Stage 3b — homology-partitioned re-evaluation: signal survives at 70%, 30% is inapplicable

Re-ran leave-cluster-out with folds disjoint by **sequence-identity cluster** (CD-HIT-style greedy
clustering on the KR domains, BLOSUM62 global %identity; embeddings reused). Identity metric verified:
self-identity 1.00; cross-cluster KR pairs 0.35–0.59 (mean 0.49).
- **70% identity → 425 homology clusters** (removes near-duplicates and close homologs): inactive-KR
  **0.850 bal-acc / 0.910 AUC**, KR stereo **0.855** — **essentially unchanged** from the BGC-cluster
  numbers (0.860 / 0.921 / 0.849). **The signal is not near-duplicate memorization; it survives
  close-homolog control.**
- **30% identity → 1 cluster** (test inapplicable). KR domains are a single conserved Rossmann-fold
  family with a pairwise-identity **floor ~35%** (verified: all sampled cross-cluster pairs ≥0.30), so
  the whole set collapses to one cluster — there are no remote (≤30%) homologs to hold out. The strict
  remote-homolog test **cannot be constructed** on a single-domain-family dataset; it is **untestable
  here, not a collapse.**
- **Net:** the strongest defensible claim is *"the head extracts per-step chemistry signal beyond close
  homologs (survives 70%-identity-clean evaluation)."* We do **not** claim remote-homolog generalization
  (untestable on one family). The earlier upper-bound caveat is resolved at the applicable threshold.

## Stage 4 — fungal transfer: the 0.57 wall is structurally unmovable by this head

Baseline reproduced exactly: the domain-feature MLP transfers to the 30 curated fungal cycles at
**active-domains 1.000 / constant-domains 0.567** (the latter is the iteration-grammar wall).

The headline experiment (a bacteria-trained *sequence* head moving the 0.567) **cannot be run as
specified, for a structural reason that no amount of data fixes.** The 30 fungal cycles span **9
synthases** (LovB 8 cycles, 6-OH-mellein 4, tenellin 4, mellein 4, 6-MSA 3, …). A fungal iterative
synthase has **one KR domain — one sequence — reused across all its cycles.** A per-domain sequence head
therefore emits **one prediction per synthase**, which is exactly the constant-domains condition that
gives 0.567. The per-cycle programming signal **is not in the per-domain sequence** (the sequence is
constant where the label varies); it is in the chain-length/substrate context the iterative enzyme sees
each cycle. Two consequences:
- **Data:** we also have no fungal per-cycle KR sequences — but this is moot given the above.
- **Structure:** even with perfect fungal sequences, per-domain sequence → one value per synthase → cannot
  exceed the within-synthase ceiling the 0.567 measures. `0.57 → 0.57`.

The bacterial head works *because* bacterial PKS are **modular** (each module its own KR domain/sequence);
the fungal wall is hard *because* fungal PKS are **iterative** (one domain, many cycles). The two are the
same coin: sequence carries per-module signal where there is a per-module sequence, and cannot carry
per-cycle signal where there is not. That is the iteration grammar wall, restated at the sequence level.

**Regime: no movement of the fungal wall — and not for lack of signal or data, but structurally.** This
*reinforces* the paper's framing (the wall is intrinsic to the iterative architecture) and is a stronger
limitations statement than "we didn't try."

## Soundness boundary
The sequence head is a **learned predictor with no soundness guarantee** — the unsound predictive layer.
It is kept separate from the sound executor (different module, no shared verdict path). If it were used,
the sound executor would remain the verifier; the head would only rank/predict over the sound type
system. Boundary typed, not smeared.

## Assessment and next steps
- **Worth pursuing — but reframed.** Not as "close the fungal 0.57 wall" (structurally precluded for
  per-domain sequence), but as **(i)** a bacterial modular per-module predictor (inactive-domain / stereo)
  once the homology-controlled number is in — useful for ClusterCAD-style modular prediction; and **(ii)**
  a *per-synthase* fungal program classifier (does this synthase's sequence predict HR vs PR, or its
  reduction profile as a whole?), which is the question the iterative architecture actually permits and is
  not the 0.567 per-cycle metric.
- **Homology-clean re-eval: DONE (Stage 3b).** Survives at 70% identity (0.850/0.910/0.855); the 30%
  remote-homolog test is inapplicable (single conserved family, ≥35% floor). Remaining refinement if
  pushed: a proper MMseqs2/CD-HIT clustering (vs the Biopython greedy used here) — expected to agree, but
  the binary is the standard reviewers know.
- **The 0.567 per-cycle wall:** not a sequence-head target. If anything closes it, it is a model of
  per-cycle *context* (chain length / intermediate state), not per-domain sequence — a different organ.

## Commit / tag
The head **trained and produced meaningful evaluation numbers** (Stage 3), so per policy this ships and is
tagged **`lpi-v0.16-sequence-head`** — the tag marks **the unblocked sequence pipeline + the demonstrated
(homology-caveated) bacterial signal**, NOT a closed fungal wall. Committed on `exploration`; **not merged
to `grammar-generalization`.** Whether/how it changes the paper is the next conversation's call — but the
honest read is: it does not reframe the paper, it sharpens the wall.

The homology-clean re-eval (Stage 3b) survives at the applicable threshold (70%), so it ships and is
tagged **`lpi-v0.17-homology-clean`** — the bacterial signal is now homology-controlled (not
near-duplicate leakage), with remote-homolog generalization explicitly untestable on this single-family
dataset. Both tags are on `exploration`; neither is merged to `grammar-generalization`.
