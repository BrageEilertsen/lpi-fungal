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
import re
import time
from pathlib import Path

from lpi.data.mibig import DEFAULT_MIBIG_DIR, RAW

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CDS_CACHE = RAW / "ncbi_cds_cache"
_PROTEIN_RE = re.compile(r"\[protein=([^\]]+)\]")

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


def _budget_from_products(products: list[str]) -> dict[str, int]:
    budget: collections.Counter[str] = collections.Counter()
    for prod in products:
        fam = family_of(prod)
        if fam:
            budget[fam] += 1
    return dict(budget)


def gene_budget_from_mibig(accession: str, mibig_dir: Path = DEFAULT_MIBIG_DIR
                           ) -> dict[str, int] | None:
    """Typed enzyme multiset for a cluster from MIBiG annotations; None if unannotated."""
    path = mibig_dir / f"{accession}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    products = [a.get("product", "") for a in d.get("genes", {}).get("annotations", [])
                if a.get("product")]
    return _budget_from_products(products) if products else None


def _loci_accessions(accession: str, mibig_dir: Path = DEFAULT_MIBIG_DIR) -> list[str]:
    path = mibig_dir / f"{accession}.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    return [loc["accession"] for loc in d.get("loci", []) if loc.get("accession")]


def _fetch_cds_fasta(genbank_acc: str, email: str | None = None,
                     offline: bool = False) -> str | None:
    """Fetch the CDS protein FASTA for a GenBank accession (cached). The headers carry
    `[protein=...]` descriptions we map to tailoring families."""
    CDS_CACHE.mkdir(parents=True, exist_ok=True)
    cache = CDS_CACHE / f"{genbank_acc}.fasta"
    if cache.exists():
        return cache.read_text()
    if offline:
        return None
    import requests

    params = {"db": "nuccore", "id": genbank_acc, "rettype": "fasta_cds_aa",
              "retmode": "text"}
    if email:
        params["email"] = email
    try:
        resp = requests.get(EUTILS, params=params, timeout=30)
        resp.raise_for_status()
        text = resp.text
        if text.startswith(">"):
            cache.write_text(text)
            time.sleep(0.34)  # <= 3 req/s without an API key
            return text
    except Exception:  # noqa: BLE001
        return None
    return None


def gene_budget_from_ncbi(accession: str, email: str | None = None,
                          offline: bool = False) -> dict[str, int] | None:
    """Enzyme multiset from the GenBank CDS `[protein=...]` descriptions of the cluster's
    loci. Note: when a locus accession is a whole contig (no cluster coordinates) this may
    include non-cluster genes -- a mild over-count of the budget (errs permissive)."""
    products: list[str] = []
    got = False
    for acc in _loci_accessions(accession):
        fasta = _fetch_cds_fasta(acc, email=email, offline=offline)
        if fasta is None:
            continue
        got = True
        products += _PROTEIN_RE.findall(fasta)
    if not got:
        return None
    return _budget_from_products(products)


def gene_budget(accession: str, email: str | None = None, offline: bool = False
                ) -> dict[str, int] | None:
    """Best-available gene budget: MIBiG annotations first, else GenBank CDS descriptions."""
    b = gene_budget_from_mibig(accession)
    if b is not None:
        return b
    return gene_budget_from_ncbi(accession, email=email, offline=offline)
