"""Gene budget: a typed tailoring-enzyme multiset per BGC (Phase 0b).

Each cluster's co-located tailoring enzymes bound which tailoring edits are admissible
(an edit type may fire only if its enzyme family is present, ~as many times as the gene
count + slack). This is one of the two observed budgets that keep the joint
(z_pks, z_tail) search tight (the other is the Delta-formula budget).

Source priority:
  1. MIBiG gene annotations (``genes.annotations[].product`` strings) -- available for ~64
     of the 326 fungal PKS clusters, and where present they name the families directly.
  2. (Phase 0b-full, scaffolded) HMMER vs Pfam on NCBI-fetched protein sequences -- needed
     to cover the remaining clusters; documented in the README, not run here.

We map free-text enzyme descriptions to tailoring-edit *families* by keyword. Unmatched
descriptions (PKS, transporter, transcription factor, hypothetical) contribute no budget.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from lpi.data.mibig import DEFAULT_MIBIG_DIR

# Tailoring-edit families (enzyme classes). Keys are the canonical family names used by
# the tailoring vocabulary; values are lowercase keyword fragments matched in the MIBiG
# gene 'product' text.
FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "oxygenase": ("p450", "cytochrome", "monooxygenase", "oxygenase", "hydroxylase",
                  "fad-dependent mono", "flavin-dependent mono", "flavin-dependent oxido",
                  "fad-dependent oxido", "nonheme", "non-heme", "dioxygenase"),
    "methyltransferase": ("methyltransferase", "o-methyl", "c-methyl", "n-methyl"),
    "halogenase": ("halogenase", "chlorinase", "brominase"),
    "prenyltransferase": ("prenyltransferase", "dmats", "prenyl", "aromatic prenyl"),
    "glycosyltransferase": ("glycosyltransferase", "glucosyltransferase", "glycosyl"),
    "oxidoreductase": ("short-chain dehydrogenase", "reductase", "oxidoreductase",
                       "dehydrogenase", "ketoreductase"),
    "decarboxylase": ("decarboxylase",),
    "acyltransferase": ("acyltransferase", "acetyltransferase"),
}


def family_of(product: str) -> str | None:
    """Map a gene 'product' description to a tailoring-edit family, or None."""
    if not product:
        return None
    p = product.lower()
    # exclude the synthase itself and obvious non-tailoring genes
    if "polyketide synthase" in p or "pks" in p:
        return None
    for fam, kws in FAMILY_KEYWORDS.items():
        if any(kw in p for kw in kws):
            return fam
    return None


def gene_budget_from_mibig(accession: str, mibig_dir: Path = DEFAULT_MIBIG_DIR
                           ) -> dict[str, int] | None:
    """Typed enzyme multiset for a cluster from MIBiG annotations; None if unannotated."""
    path = mibig_dir / f"{accession}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    anns = d.get("genes", {}).get("annotations", [])
    products = [a.get("product", "") for a in anns if a.get("product")]
    if not products:
        return None
    budget: collections.Counter[str] = collections.Counter()
    for prod in products:
        fam = family_of(prod)
        if fam:
            budget[fam] += 1
    return dict(budget)
