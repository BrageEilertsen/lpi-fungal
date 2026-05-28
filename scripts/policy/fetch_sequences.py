"""Fetch full-length protein sequences for the 6 real curated HR/PR synthases (Phase B1).

The other 3 HR/PR curated entries (3-HB, hexanoic, octanoic) are deliberate proxy
programs without a real enzyme behind them; they are skipped here and remain
substrate-only in the policy.

Sources (verified by manual cross-reference):
  6-MSA       UniProt P22367         (Penicillium patulum MSAS / AtX)
  mellein     NCBI    AIW00670.1     (P. nodorum SnPKS19, mlnS, in BGC0001244)
  6-OH-mell.  NCBI    AUW31184.1     (Cladonia uncialis type-I PKS in BGC0001489 MG777496)
  LovB        UniProt Q9Y8A5         (A. terreus lovastatin nonaketide synthase)
  LovF        NCBI    AAD34559.1     (A. terreus lovastatin diketide synthase, AF141925)
  TenS        NCBI    CAL69597.1     (B. bassiana tenellin PKS-NRPS, BGC0001049 AM409327)

NCBI fetches reuse the existing CDS cache; UniProt fetches go through a thin REST
wrapper with its own cache. Both are rate-limited politely; no manual API keys needed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "data" / "policy" / "sequences"
CDS_CACHE = ROOT / "data" / "raw" / "ncbi_cds_cache"
UNIPROT_CACHE = ROOT / "data" / "raw" / "uniprot_cache"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


@dataclass
class Target:
    slug: str          # filename slug
    name: str          # human-readable
    source: str        # "uniprot" or "ncbi_cds"
    accession: str     # UniProt or NCBI protein accession
    locus: str | None  # GenBank locus (for ncbi_cds source)
    curated_name: str  # matches curated YAML name


TARGETS = [
    Target("6_methylsalicylic_acid", "MSAS / AtX", "uniprot", "P22367", None,
           "6-methylsalicylic acid"),
    Target("mellein", "SnPKS19 / mlnS", "ncbi_cds", "AIW00670.1", "KM365454.1",
           "mellein"),
    # 6-OH-mellein: BGC0001489 has two PKS-family proteins. AUW31184 (2152 aa,
    # annotated "type I PKS") has NO KR domain by PF08659 -- looks like an NR-PKS that
    # makes a different cluster metabolite. AUW31183 (946 aa, "PKS-like") DOES carry a
    # strong KR domain (bitscore 200.3, e<1e-60). For a KR-only conformational pilot
    # this is the right protein to fold even though the literature gene-product
    # assignment is not fully resolved by MIBiG. Noted in the Phase B checkpoint.
    Target("6_hydroxymellein", "Cladonia PKS-like w/ KR", "ncbi_cds", "AUW31183.1",
           "MG777496.1", "6-hydroxymellein"),
    Target("lovb", "LovB nonaketide synthase", "uniprot", "Q9Y8A5", None,
           "LovB dihydromonacolin L nonaketide"),
    Target("lovf", "LovF diketide synthase", "ncbi_cds", "AAD34559.1", "AF141925.1",
           "2-methylbutyric acid (LovF diketide; C-MeT + full-cascade test)"),
    Target("tenellin_tens", "TenS PKS-NRPS", "ncbi_cds", "CAL69597.1", "AM409327.1",
           "tenellin polyketide backbone (TenS PKS portion)"),
]

# The 3 deliberate proxies -- substrate-only in the policy; no protein to fold.
PROXIES = [
    "3-hydroxybutyric acid (KR-only diketide; beta-hydroxyl state test)",
    "hexanoic acid (fully-reduced triketide; FAS/HR-PKS full-cascade proxy)",
    "octanoic acid (fully-reduced tetraketide; full-cascade proxy)",
]


def fetch_uniprot(acc: str) -> str:
    UNIPROT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = UNIPROT_CACHE / f"{acc}.fasta"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text()
    resp = requests.get(UNIPROT_URL.format(acc=acc), timeout=30)
    resp.raise_for_status()
    text = resp.text
    if not text.startswith(">"):
        raise RuntimeError(f"UniProt {acc} returned non-FASTA: {text[:200]}")
    cache.write_text(text)
    time.sleep(0.4)
    return text


def fetch_ncbi_cds_locus(locus: str) -> str:
    """Reuse the existing CDS FASTA cache used elsewhere in the repo."""
    CDS_CACHE.mkdir(parents=True, exist_ok=True)
    cache = CDS_CACHE / f"{locus}.fasta"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text()
    params = {"db": "nuccore", "id": locus, "rettype": "fasta_cds_aa", "retmode": "text",
              "email": "brageei@uio.no"}
    resp = requests.get(EUTILS, params=params, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if not text.startswith(">"):
        raise RuntimeError(f"NCBI {locus} returned non-FASTA")
    cache.write_text(text)
    time.sleep(0.4)
    return text


def extract_cds_entry(multi_fasta: str, accession: str) -> str | None:
    """Pull a single >header...sequence record matching ``accession`` from a multi-FASTA."""
    # CDS headers look like ">lcl|<LOCUS>_prot_<ACCESSION>_<N> [protein=...] ..."
    pat = re.compile(rf">\S*_{re.escape(accession)}_\d+\b")
    records = re.split(r"(?m)^(?=>)", multi_fasta)
    for rec in records:
        if not rec.strip():
            continue
        first = rec.split("\n", 1)[0]
        if pat.search(first) or accession in first:
            return rec
    return None


def normalize_fasta(text: str, header_id: str, description: str) -> str:
    """Strip the multi-line FASTA to a single record with a clean header for AF2."""
    lines = text.strip().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    seq = re.sub(r"[^A-Za-z]", "", seq)
    # AF2 expects the canonical 20 amino acids; X/U/B/Z/J get rejected by some downstream
    # tools. ColabFold tolerates X; flag if present.
    return f">{header_id} {description}\n{seq}\n"


def main() -> None:
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tgt in TARGETS:
        try:
            if tgt.source == "uniprot":
                raw = fetch_uniprot(tgt.accession)
                rec = raw  # UniProt returns a single record
            else:
                multi = fetch_ncbi_cds_locus(tgt.locus)  # type: ignore[arg-type]
                rec = extract_cds_entry(multi, tgt.accession)
                if rec is None:
                    raise RuntimeError(
                        f"could not find protein {tgt.accession} in locus {tgt.locus}"
                    )
            clean = normalize_fasta(rec, header_id=tgt.slug,
                                    description=f"{tgt.name} {tgt.accession}")
            seq_len = sum(1 for c in clean if c.isalpha()) - len(tgt.slug) - len(tgt.name) - len(tgt.accession)
            # length more reliably: count residues in the body
            body = clean.split("\n", 1)[1]
            seq_len = sum(1 for c in body if c.isalpha())
            out = SEQ_DIR / f"{tgt.slug}.fasta"
            out.write_text(clean)
            rows.append((tgt.slug, tgt.name, tgt.accession, seq_len, "OK"))
        except Exception as exc:  # noqa: BLE001 -- report every failure
            rows.append((tgt.slug, tgt.name, tgt.accession, 0, f"FAIL: {exc}"))

    print(f"{'slug':<22} {'protein':<28} {'acc':<14} {'len':>5}  status")
    print("-" * 80)
    for slug, name, acc, ln, status in rows:
        print(f"{slug:<22} {name:<28} {acc:<14} {ln:>5}  {status}")

    print(f"\nproxies skipped (substrate-only in the policy):")
    for p in PROXIES:
        print(f"  - {p}")

    # Coverage of the curated HR+PR cycle table.
    import pandas as pd
    df = pd.read_parquet(ROOT / "data" / "policy" / "intermediates.parquet")
    hr_pr = df[df.subclass_hr_pr]
    by_syn = hr_pr.groupby("synthase").size()
    covered_names = {tgt.curated_name for tgt in TARGETS}
    covered_cycles = int(hr_pr[hr_pr.synthase.isin(covered_names)].shape[0])
    print(f"\ncoverage of HR+PR cycle table: {covered_cycles}/{len(hr_pr)} cycles "
          f"({len(covered_names)} synthases of {hr_pr.synthase.nunique()}).")
    print("Uncovered (proxies; substrate-only arm):")
    for s, n in by_syn.items():
        if s not in covered_names:
            print(f"  {s}  ({n} cycle(s))")


if __name__ == "__main__":
    main()
