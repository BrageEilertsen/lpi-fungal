"""Reconstruct per-KR-domain sequences for ClusterCAD modules (Track B).

The clean source (ClusterCAD ``/pks/domainLookup``) is permanently 404. We rebuild per-KR
sequences from primary data: MIBiG gives each cluster's GenBank accession(s); we fetch the
CDS protein translations (``genebudget._fetch_cds_fasta``, cached) and locate KR domains with
the Pfam KR pHMM (PF08659) via pyhmmer.

To guarantee correct labels with NO fragile alignment, we keep only clusters where the HMM
KR-domain count equals ClusterCAD's KR-module count, then zip the two ordered lists (genomic
order). Clusters that don't match exactly are skipped (reported), trading n for label safety.

Output: fills ``kr_sequence`` in the kr_examples parquet, ready for ``esm_head.evaluate_head``.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import pyhmmer

from lpi.data.genebudget import CDS_CACHE, _loci_accessions
from lpi.data.mibig import RAW
from lpi.model.esm_data import build_kr_examples

KR_HMM = RAW / "PF08659_KR.hmm"
_LOC_RE = re.compile(r"\[location=(?:complement\()?(?:join\()?<?(\d+)")


@dataclass
class Protein:
    name: str
    seq: str
    start: int  # genomic start coordinate, for assembly-line ordering


def parse_cds_fasta(text: str) -> list[Protein]:
    """Parse an NCBI ``fasta_cds_aa`` dump into proteins with a genomic-start sort key."""
    proteins: list[Protein] = []
    name: str | None = None
    start = 0
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                proteins.append(Protein(name, "".join(buf), start))
            name = line[1:].split()[0]
            m = _LOC_RE.search(line)
            start = int(m.group(1)) if m else 0
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        proteins.append(Protein(name, "".join(buf), start))
    return proteins


def _load_hmm():
    with pyhmmer.plan7.HMMFile(str(KR_HMM)) as fh:
        return fh.read()


def kr_domain_sequences(proteins: list[Protein], hmm, alphabet, pipeline) -> list[str]:
    """KR-domain AA subsequences across a cluster's proteins, in (gene, in-protein) order."""
    by_name = {p.name: p for p in proteins if p.seq}
    block = pyhmmer.easel.DigitalSequenceBlock(
        alphabet,
        [pyhmmer.easel.TextSequence(name=p.name.encode(), sequence=p.seq).digitize(alphabet)
         for p in by_name.values()],
    )
    hits = pipeline.search_hmm(hmm, block)
    found: list[tuple[int, int, str]] = []
    for hit in hits:
        if not hit.included:
            continue
        hit_name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
        p = by_name.get(hit_name)
        if p is None:
            continue
        for dom in hit.domains:
            aln = dom.alignment
            tf, tt = aln.target_from, aln.target_to  # 1-based inclusive
            found.append((p.start, tf, p.seq[tf - 1:tt]))
    found.sort(key=lambda x: (x[0], x[1]))
    return [s for _a, _b, s in found]


def reconstruct() -> tuple[dict[str, str], dict[str, int]]:
    """Return (kr_domainid -> sequence) for count-matched clusters, plus a stats dict."""
    examples = build_kr_examples()
    by_cluster: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_cluster[ex.cluster].append(ex)  # document order == module order

    alphabet = pyhmmer.easel.Alphabet.amino()
    hmm = _load_hmm()
    pipeline = pyhmmer.plan7.Pipeline(alphabet)

    seqs: dict[str, str] = {}
    stats = defaultdict(int)
    for cluster, exs in by_cluster.items():
        proteins: list[Protein] = []
        for acc in _loci_accessions(cluster.split(".")[0]):
            f = CDS_CACHE / f"{acc}.fasta"
            if f.exists():
                proteins += parse_cds_fasta(f.read_text())
        if not proteins:
            stats["no_proteins"] += 1
            continue
        kr_seqs = kr_domain_sequences(proteins, hmm, alphabet, pipeline)
        if kr_seqs and len(kr_seqs) == len(exs):
            for ex, s in zip(exs, kr_seqs):
                seqs[ex.kr_domainid] = s
            stats["clusters_matched"] += 1
            stats["kr_sequences"] += len(exs)
        else:
            stats["count_mismatch"] += 1
            stats[f"mm_{len(kr_seqs)}v{len(exs)}"] += 0  # placeholder; detail printed by caller
    return seqs, dict(stats)
