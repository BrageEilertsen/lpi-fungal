"""Phase D3a: auto-discover + fetch sequences for the 11 new chemistries.

For each new BGC, we don't hand-curate the protein accession (as we did in Phase B
for the 6 curated synthases) -- we auto-discover. The lesson from 6-OH-mellein is
that MIBiG's textual ``[protein=...]`` annotations are NOT reliable: the canonical
"type I PKS" in BGC0001489 had no KR, while the smaller "PKS-like" carried it.

Algorithm per BGC:
  1. Load MIBiG JSON, get loci accessions.
  2. Fetch each locus's CDS FASTA via genebudget._fetch_cds_fasta (cached).
  3. Parse all CDS records into Protein records.
  4. HMMER PF08659 scan over EVERY protein in the cluster.
  5. Pick the protein with the highest-scoring KR domain hit.
  6. Extract KR+ACP didomain (KR alignment +/- N30 / C150 pad).
  7. Write data/policy/sequences/{bgc_id}_kr_acp.fasta.

BGCs with non-CHO products that still have a KR (e.g. citrinin uses C-MeT branches
but the backbone is still polyketide CHO at the core) are handled the same way -- the
KR domain is what we fold + dock around. Strobilurin's benzoyl-CoA loading is a
domain we DON'T need to model in this didomain fold; the KR+ACP geometry is what
matters for downstream Vina + descriptors.

Output: data/policy/sequences/{bgc_id}_kr_acp.fasta + data/policy/sequences/_d3_kr_index.parquet
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pyhmmer
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.data.genebudget import _fetch_cds_fasta  # noqa: E402
from lpi.data.mibig import DEFAULT_MIBIG_DIR  # noqa: E402

SEQ_DIR = ROOT / "data" / "policy" / "sequences"
KR_HMM = ROOT / "data" / "raw" / "PF08659_KR.hmm"
INDEX_PARQUET = SEQ_DIR / "_d3_kr_index.parquet"
PAD_N = 30
PAD_C = 150
PKS_MIN_LENGTH = 1500  # fungal iterative PKSes are typically 1500-5000 aa; below this
                       # we treat hits as tailoring-enzyme false positives

# 11 unique chemistries (representative BGC each). Duplicates aliased downstream.
# CITRININ is the priority target for the entropy-plateau test.
NEW_BGCS = [
    ("BGC0001121", "orsellinic_acid",                ),
    ("BGC0001606", "gibepyrone_A",                   ),
    ("BGC0002191", "prolipyrone_B",                  ),
    ("BGC0001338", "citrinin",                       ),  # <-- the key target
    ("BGC0000161", "isoterrein",                     ),
    ("BGC0002852", "3_6_dialkyl_alpha_pyrone",       ),
    ("BGC0002180", "asperlin",                       ),
    ("BGC0001273", "asperlactone",                   ),
    ("BGC0001909", "strobilurin_A",                  ),  # benzoyl starter; one fold serves both BGC0001909+0002065
    ("BGC0002238", "dihydroxy_methoxypropiophenone", ),
    ("BGC0003109", "stachysalicyloid_C",             ),
]

_LOC_RE = re.compile(r"\[location=(?:complement\()?(?:join\()?<?(\d+)")
_PROT_RE = re.compile(r"_prot_([A-Z]+\d+\.\d+)_")
_PRODUCT_RE = re.compile(r"\[protein=([^\]]+)\]")


@dataclass
class Protein:
    name: str            # NCBI protein accession (e.g. CAL69597.1)
    seq: str
    start: int           # genomic start coord (for assembly-line ordering)
    description: str     # the [protein=...] text


def parse_cds_fasta(text: str) -> list[Protein]:
    proteins: list[Protein] = []
    header: str | None = None
    desc = ""
    name = ""
    start = 0
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                proteins.append(Protein(name, "".join(buf), start, desc))
            header = line[1:]
            mn = _PROT_RE.search(header)
            name = mn.group(1) if mn else header.split()[0]
            ml = _LOC_RE.search(header)
            start = int(ml.group(1)) if ml else 0
            mp = _PRODUCT_RE.search(header)
            desc = mp.group(1) if mp else ""
            buf = []
        else:
            buf.append(line.strip())
    if header is not None:
        proteins.append(Protein(name, "".join(buf), start, desc))
    return proteins


@dataclass
class BGCResult:
    bgc_id: str
    chemistry: str
    protein_accession: str
    protein_description: str
    full_length: int
    kr_aln_start: int    # 1-based, inclusive
    kr_aln_end: int
    kr_bitscore: float
    kr_evalue: float
    kr_acp_start: int
    kr_acp_end: int
    kr_acp_length: int
    notes: str


def fetch_locus(acc: str) -> str | None:
    """Use the existing genebudget._fetch_cds_fasta -- cached."""
    return _fetch_cds_fasta(acc, email="brageei@uio.no", offline=False)


def scan_cluster_for_kr(bgc_id: str, chemistry: str,
                        alphabet, hmm, pipeline) -> BGCResult | None:
    """Open BGC JSON, fetch all loci's CDS, HMMER scan every protein, pick best KR."""
    path = DEFAULT_MIBIG_DIR / f"{bgc_id}.json"
    if not path.exists():
        return BGCResult(bgc_id, chemistry, "", "", 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                         f"FAIL: no MIBiG JSON")
    j = json.loads(path.read_text())
    loci = [l.get("accession") for l in j.get("loci", []) if l.get("accession")]
    if not loci:
        return BGCResult(bgc_id, chemistry, "", "", 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                         "FAIL: no loci in MIBiG entry")
    proteins: list[Protein] = []
    for acc in loci:
        text = fetch_locus(acc)
        if text is None:
            continue
        proteins.extend(parse_cds_fasta(text))
    if not proteins:
        return BGCResult(bgc_id, chemistry, "", "", 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                         f"FAIL: could not fetch CDS for loci {loci}")
    # HMMER scan over all proteins in the cluster.
    digital = [pyhmmer.easel.TextSequence(name=p.name.encode(), sequence=p.seq).digitize(alphabet)
               for p in proteins if p.seq]
    if not digital:
        return BGCResult(bgc_id, chemistry, "", "", 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                         "FAIL: no sequences to scan")
    block = pyhmmer.easel.DigitalSequenceBlock(alphabet, digital)
    hits = list(pipeline.search_hmm(hmm, block))
    by_name = {p.name: p for p in proteins}
    # Flatten (protein, domain) pairs; sort by domain bitscore.
    domain_records = []
    for h in hits:
        if not h.included:
            continue
        hit_name = h.name.decode() if isinstance(h.name, bytes) else h.name
        p = by_name.get(hit_name)
        if p is None:
            continue
        for d in h.domains:
            domain_records.append((d.score, h, d, p))
    if not domain_records:
        return BGCResult(bgc_id, chemistry, "", "", 0, 0, 0, 0.0, 0.0, 0, 0, 0,
                         f"FAIL: no PF08659 KR hit across {len(proteins)} cluster proteins")
    # PF08659 hits short tailoring enzymes (e.g. CitE dehydrogenase at score 42 in
    # the citrinin cluster) more aggressively than its true target (the iterative
    # PKS with KR domain). Filter to fungal-PKS-sized proteins first.
    large_records = [r for r in domain_records if len(r[3].seq) >= PKS_MIN_LENGTH]
    fallback_to_full_pks = False
    if large_records:
        large_records.sort(key=lambda r: -r[0])
        score, hit, dom, p = large_records[0]
        size_filter = f"size_filter>={PKS_MIN_LENGTH}aa applied"
    else:
        # No large protein has a PF08659 KR hit. This happens for divergent PR-PKS
        # KR domains (e.g. citrinin's CitS at 2593 aa is undetected by PF08659).
        # Fall back to picking the largest "polyketide synthase" / "PKS" protein in
        # the cluster by description; we will fold this protein FULL-LENGTH and let
        # the ACP-Ser motif scan locate the anchor downstream.
        pks_by_desc = [p for p in proteins
                       if any(k in p.description.lower() for k in
                              ("polyketide synthase", "pks", "type i pks", "iterative pks"))
                       and len(p.seq) >= PKS_MIN_LENGTH]
        if pks_by_desc:
            pks_by_desc.sort(key=lambda p: -len(p.seq))
            p = pks_by_desc[0]
            # Synthetic "hit": no KR localization, just full-length fold for ACP scan.
            class _FakeHit:
                evalue = 0.0
            class _FakeAln:
                target_from = 1
                target_to = len(p.seq)
            class _FakeDom:
                alignment = _FakeAln()
                score = 0.0
            score = 0.0
            hit = _FakeHit()
            dom = _FakeDom()
            fallback_to_full_pks = True
            size_filter = (f"NO_LARGE_PKS_KR_HIT; fallback to largest 'PKS' protein "
                           f"by description ({p.description[:50]}, {len(p.seq)} aa); "
                           f"folding FULL-LENGTH -- ACP-Ser identified by motif scan downstream")
        else:
            domain_records.sort(key=lambda r: -r[0])
            score, hit, dom, p = domain_records[0]
            size_filter = "NO_LARGE_PKS in cluster; using best small-protein hit"
    aln = dom.alignment
    aln_from, aln_to = aln.target_from, aln.target_to  # 1-based inclusive
    pad_from = max(1, aln_from - PAD_N)
    pad_to = min(len(p.seq), aln_to + PAD_C)
    return BGCResult(
        bgc_id=bgc_id, chemistry=chemistry,
        protein_accession=p.name, protein_description=p.description,
        full_length=len(p.seq),
        kr_aln_start=aln_from, kr_aln_end=aln_to,
        kr_bitscore=float(score), kr_evalue=float(hit.evalue),
        kr_acp_start=pad_from, kr_acp_end=pad_to,
        kr_acp_length=pad_to - pad_from + 1,
        notes=f"picked {p.name} ({p.description[:50]}) from {len(proteins)} cluster proteins; {size_filter}",
    )


def write_didomain_fasta(result: BGCResult) -> None:
    """Write the KR+ACP didomain FASTA for AF2 input."""
    # Need to re-load the protein sequence (we have the metadata only).
    path = DEFAULT_MIBIG_DIR / f"{result.bgc_id}.json"
    j = json.loads(path.read_text())
    loci = [l.get("accession") for l in j.get("loci", []) if l.get("accession")]
    seq = ""
    for acc in loci:
        text = fetch_locus(acc)
        if text is None:
            continue
        for p in parse_cds_fasta(text):
            if p.name == result.protein_accession:
                seq = p.seq
                break
        if seq:
            break
    if not seq:
        raise RuntimeError(f"could not re-load sequence for {result.protein_accession}")
    kr_seq = seq[result.kr_acp_start - 1 : result.kr_acp_end]
    out = SEQ_DIR / f"{result.bgc_id}_kr_acp.fasta"
    header = (f">{result.bgc_id}_kr_acp {result.chemistry} {result.protein_accession} | "
              f"KR+ACP didomain (PF08659 KR aln {result.kr_aln_start}-{result.kr_aln_end}; "
              f"extracted {result.kr_acp_start}-{result.kr_acp_end} of {result.full_length})")
    out.write_text(f"{header}\n{kr_seq}\n")


def main() -> None:
    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(str(KR_HMM)) as fh:
        hmm = fh.read()
    pipeline = pyhmmer.plan7.Pipeline(alphabet)

    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    results: list[BGCResult] = []
    for bgc_id, chemistry in NEW_BGCS:
        print(f"  {bgc_id} ({chemistry})... ", end="", flush=True)
        r = scan_cluster_for_kr(bgc_id, chemistry, alphabet, hmm, pipeline)
        if r is None:
            print("SKIP")
            continue
        if r.protein_accession:
            try:
                write_didomain_fasta(r)
                print(f"OK  {r.protein_accession} bitscore={r.kr_bitscore:.1f} "
                      f"kr_acp_len={r.kr_acp_length}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL writing fasta: {exc!r}")
                r.notes = f"{r.notes}; WRITE_FAIL: {exc!r}"
        else:
            print(f"NO_KR  {r.notes}")
        results.append(r)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_parquet(INDEX_PARQUET, index=False)
    print(f"\nwrote {INDEX_PARQUET.relative_to(ROOT)}")
    print()
    print(df[["bgc_id", "chemistry", "protein_accession", "full_length",
              "kr_aln_start", "kr_aln_end", "kr_bitscore", "kr_acp_length"]].to_string(index=False))


if __name__ == "__main__":
    main()
