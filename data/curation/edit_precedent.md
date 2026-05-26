# Edit-tier curation log — precedent for the realizability cost basis

*Companion to [`edit_tiers.csv`](edit_tiers.csv). Web-verified, 2026-05-26. **Curation closed as v0.14** —
the four expert calls below are resolved and wired into [`realizability.py`](../../src/lpi/realizability.py).*

**Method & honesty bar.** Every reference comes from a live literature search; the load-bearing
control-axis and `add_cmt` claims were confirmed by fetching the source. Citations give
title + venue + year + DOI/PMID + URL; author names appear **only where confirmed at the source**
(the Cox papers). **No citation here is invented.**

**What the curation replaced.** The realizability layer assigns each design a per-edit feasibility tier
(`DOCUMENTED` / `PLAUSIBLE` / `SPECULATIVE`) — the entire cost basis the experiment-selection planner inherits.
These were class-level constants; they are now literature-grounded and the control axis is super-additive.

---

## Status — wired, tested, re-run

The four calls are wired into `realizability.edits_from` / `cost`; full suite green (130 tests).
Re-running the 1-edit design neighbourhood, the headline ratio **softens from 93:34 to 69:49**
(frontier : engineerable; speculative 15 → 24). Still control-dominant — but a defensible margin rather
than a dramatic one. Release-reprogram designs moved into `engineerable`; `cycle_remove` and `add_cmt`
designs moved into `speculative`.

## Headline findings

1. **The moat is literature-grounded.** Control-axis cost is backed by Cox 2023 (*Nat. Prod. Rep.*):
   programming iterative HR-PKS is "a key unsolved problem," rational reprogramming "will remain
   extremely difficult," the programme is "an emergent property... highly unpredictable."
2. **Cox 2023 names machine learning** as the promising route where rational engineering fails — a
   citable endorsement of the LPI premise, for the introduction, not related work.
3. **Curation did real work, not just confirmation.** The four calls changed 3 tiers and split 1 kind:
   `add_cmt` DOCUMENTED→SPECULATIVE, `cycle_remove` PLAUSIBLE→SPECULATIVE, `release` SPECULATIVE→PLAUSIBLE,
   and `add_domain` split into `add_reductive` / `add_cmt`.
4. **The cost model now *is* the published difficulty landscape.** Control edits stack super-additively
   (quadratic), reflecting Cox 2023's "emergent" framing; a single control edit stays accessible
   (Fisch & Cox 2011).

---

## Structural axis — bacterial modular-PKS engineering precedent

### `add_reductive` — add a reductive domain (+KR/+DH/+ER) — **DOCUMENTED** (high) — *Call 1 split*
Reductive loops (KR / KR-DH / KR-DH-ER) transplant as a single structural unit; "most hybrid
multienzymes were active." Mature modular precedent. **Demonstrated: yes (modular).**
- *A polylinker approach to reductive loop swaps in modular PKS.* [PMID 18937219](https://pubmed.ncbi.nlm.nih.gov/18937219/)
- *Engineering Polyketide Stereocenters with Ketoreductase Domain Exchanges* (44 KR exchanges → all four stereoisomers). JACS 2025. [10.1021/jacs.5c06736](https://pubs.acs.org/doi/10.1021/jacs.5c06736)
- *Assessing and harnessing updated PKS modules through combinatorial engineering.* [PMC10402262](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402262/)

### `add_cmt` — de-novo C-MeT insertion — **SPECULATIVE** (high) — *Call 1 split, verified*
The model's `add_cmt` gives a methylation-free synthase (e.g. 6-MSAS) a C-MeT it never had. **The
literature has no precedent for this.** Verified at source: in the tenellin/desmethylbassianin system
*both* synthases carry the C-MeT domain; the methylation difference is *programming* ("competition
between the C-MeT and ketoreductase domains"), and the chimera "inserts the DMBS C-MeT–ΨKR region into
TENS." Precedent is reprogramming a *present* C-MeT, never de-novo insertion. **Demonstrated: no.**
- *Molecular basis of methylation and chain-length programming in a fungal iterative HR-PKS.* Chem. Sci. 2019. [PMC6839510](https://pmc.ncbi.nlm.nih.gov/articles/PMC6839510/) / [10.1039/C9SC03173A](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc03173a)

### `extender` — AT-domain / extender-unit swap (malonyl ↔ methylmalonyl) — **DOCUMENTED** (high)
The canonical combinatorial-biosynthesis move (YASH ↔ HAFH motif switch in DEBS). **Demonstrated: yes**,
caveat — hybrid modules active but chain elongation "seriously attenuated" (yield penalty, not a wall).
- *Acyltransferases as Tools for Polyketide Synthase Engineering.* Antibiotics 2018, 7(3):62. [PMC6164871](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6164871/)
- *Expanding Extender Substrate Selection... by AT Domain Exchange within a Modular PKS.* JACS. [10.1021/jacs.2c11027](https://pubs.acs.org/doi/10.1021/jacs.2c11027)
- *Mechanistic Analysis of Acyl Transferase Domain Exchange in PKS Modules.* JACS 2003. [PMID 12720450](https://pubmed.ncbi.nlm.nih.gov/12720450/)
- *The malonyl/acetyl-transferase from murine FAS is a promiscuous engineering tool.* Commun. Chem. 2024. [s42004-024-01269-1](https://www.nature.com/articles/s42004-024-01269-1)

### `starter` — loading-module / starter-unit swap (acetyl → propionyl/butyryl/hexanoyl) — **DOCUMENTED** (high)
Loading-module swaps + precursor-directed biosynthesis well established; downstream machinery "quite
tolerant to non-natural starter unit side-chains." **Demonstrated: yes**; some changes abolish activity.
- *Precursor-Directed Biosynthesis of Erythromycin Analogs by an Engineered PKS.* Science 1997, 277(5324):367. [10.1126/science.277.5324.367](https://www.science.org/doi/10.1126/science.277.5324.367)
- *Engineering broader specificity into an antibiotic-producing PKS* (avermectin loading → DEBS1). [PMID 9422686](https://pubmed.ncbi.nlm.nih.gov/9422686/)
- *Engineering specificity of starter unit selection by the erythromycin-producing PKS.* [PMID 11918808](https://pubmed.ncbi.nlm.nih.gov/11918808/)

### `release` — thioesterase / cyclase reprogram — **PLAUSIBLE** (med) — *Call 3: demoted*
Enzymatically a **TE/cyclase domain swap** — structurally bacterial-precedent territory (pikromycin/DEBS
TEs; ring-size control). Demoting from SPECULATIVE→PLAUSIBLE fixes the contradiction where the cost table
ranked a domain swap as harder than reprogramming the iteration. **Demonstrated: partial** — cyclization
*mode* change (aromatic/lactone/Claisen) is still imperfectly controlled, hence PLAUSIBLE not DOCUMENTED.
- *Structural basis for macrolactonization by the pikromycin thioesterase.* Nat. Chem. Biol. 2006. [PMID 16969372](https://www.nature.com/articles/nchembio824)
- *Crystal structure of the macrocycle-forming TE domain of the erythromycin PKS.* PNAS 2001. [10.1073/pnas.011399198](https://www.pnas.org/doi/10.1073/pnas.011399198)
- *Thioesterase Domains of Fungal Nonreducing PKS Act as Decision Gates during Combinatorial Biosynthesis.* [PMC3780601](https://pmc.ncbi.nlm.nih.gov/articles/PMC3780601/)
- *Structure and function of an iterative PKS thioesterase domain catalyzing Claisen cyclization in aflatoxin biosynthesis.* PNAS 2010. [10.1073/pnas.0913531107](https://www.pnas.org/doi/10.1073/pnas.0913531107)

---

## Control axis — the iteration-grammar frontier (super-additive cost, *Call 2*)

A single control edit near a real template is achievable; stacking them is the wall. Cost is therefore
quadratic in control-edit count: `cost = Σ(structural tiers) + Σ(control tiers)·k_control` — linear at
k=1, ~k² for k>1. This is Cox 2023's "emergent, non-separable programme" written into the cost function.

### `reduction` / `c_methyl` — per-cycle reduction state / C-MeT timing — **PLAUSIBLE single-edit** (med)
Rational domain swaps between *closely related* synthases (tenellin ↔ desmethylbassianin in *A. oryzae*)
mapped methylation-pattern and chain-length changes cleanly, even resurrecting an extinct metabolite
(Fisch & Cox 2011) — so a single edit near a real template has genuine precedent. General programming is
"emergent... highly unpredictable" (Cox 2023) → handled by super-additive stacking.
- *Rational domain swaps decipher programming in fungal HR-PKS and resurrect an extinct metabolite.* Fisch, Bakeer, Yakasai, Song, Pedrick, Wasil, Bailey, Lazarus, Simpson, **Cox**. JACS 2011, 133(41):16635–41. [10.1021/ja206914q](https://pubmed.ncbi.nlm.nih.gov/21899331/)
- *Molecular basis of methylation and chain-length programming in a fungal iterative HR-PKS.* Chem. Sci. 2019. [10.1039/C9SC03173A](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc03173a)
- *Curiouser and curiouser: progress in understanding the programming of iterative HR-PKS.* **Russell J. Cox.** Nat. Prod. Rep. 2023, 40:9–27. [10.1039/D2NP00007E](https://pubs.rsc.org/en/content/articlehtml/2023/np/d2np00007e)

### `cycle_add` — add an iteration — **PLAUSIBLE** (low)
Chain length is set partly by KS and partly by an **extrinsic partner** thiohydrolase — not a pure
intrinsic reprogramming knob. Chain-length control listed as unsolved (Cox 2023). **Demonstrated: partial.**
- *Fungal PKS Product Chain-Length Control by Partnering Thiohydrolase.* ACS Chem. Biol. 2014. [10.1021/cb500284t](https://pubs.acs.org/doi/10.1021/cb500284t)

### `cycle_remove` — remove an iteration — **SPECULATIVE** (low) — *Call 4: raised*
Changing the iteration count changes the integer the program emits as its termination condition — deep
iteration-program editing, the least-precedented edit in the table. The cost table now correctly says
"you can plausibly reprogram one cycle's chemistry; you cannot plausibly change how many cycles happen."
- *Reengineering the programming of a functional domain of an iterative HR-PKS.* RSC Adv. 2020. [10.1039/D0RA04026F](https://pubs.rsc.org/en/content/articlehtml/2020/ra/d0ra04026f)

---

## The four calls — resolved (Brage, 2026-05-26)

1. **Split `add_domain`** → `add_reductive` (DOCUMENTED) / `add_cmt` (SPECULATIVE, verified de-novo
   insertion is unprecedented). ✓ wired.
2. **Super-additive control cost** — quadratic in control-edit count; literature-grounded by Cox 2023's
   emergent/non-separable framing, not a free modeling choice. ✓ wired in `cost`.
3. **Demote `release`** SPECULATIVE→PLAUSIBLE — it is a TE/cyclase domain swap (structural), not the
   iteration frontier; resolves the thesis contradiction. ✓ wired.
4. **Raise `cycle_remove`** PLAUSIBLE→SPECULATIVE — termination-count editing is deeper than per-cycle
   reprogramming. ✓ wired.

**Effect:** worked-figure ratio 93:34 → 69:49 (control still dominant, defensibly). Tests updated +
extended (`add_cmt`, `cycle_remove`, super-additive cost); suite green.
