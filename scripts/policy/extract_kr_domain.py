"""KR-domain extraction via pyhmmer (Phase B2).

For each full-length synthase FASTA in ``data/policy/sequences/{slug}.fasta``, scan with
the cached Pfam KR pHMM (``data/raw/PF08659_KR.hmm``) and emit the KR-domain region (with
a 30 aa pad on each side -- enough to keep the Rossmann fold's flanking helices intact
without dragging in adjacent KS/AT domain).

Fungal iterative synthases have exactly one KR per polypeptide (reused across cycles);
we report the highest-scoring KR hit and flag any synthase where multiple non-overlapping
KR hits appear (none expected for our 6).

Output: ``data/policy/sequences/{slug}_kr.fasta`` and a summary index parquet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pyhmmer

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "data" / "policy" / "sequences"
KR_HMM = ROOT / "data" / "raw" / "PF08659_KR.hmm"
PAD_N = 30   # N-terminal flank around the KR HMM hit (keeps Rossmann's first helix)
PAD_C = 150  # C-terminal flank -- enough to capture the immediately-downstream ACP in
             # the canonical fungal HR-PKS domain order (...KR-ACP-TE), giving Vina the
             # ACP serine OG anchor for covalent docking (per Brage's decision 1).
INDEX_PARQUET = SEQ_DIR / "_kr_index.parquet"


@dataclass
class KRHit:
    slug: str
    full_length: int
    n_kr_hits: int
    kr_start: int  # 1-based inclusive, after padding
    kr_end: int    # 1-based inclusive, after padding
    kr_aln_start: int  # 1-based inclusive, HMM alignment region (pre-pad)
    kr_aln_end: int
    bitscore: float
    evalue: float
    kr_length: int
    notes: str


def load_fasta(path: Path) -> tuple[str, str]:
    text = path.read_text().strip()
    header, *body = text.splitlines()
    seq = re.sub(r"[^A-Za-z]", "", "".join(body))
    return header.lstrip(">"), seq


def scan_kr(slug: str, header: str, seq: str, alphabet, hmm, pipeline) -> KRHit:
    digital = pyhmmer.easel.TextSequence(name=slug.encode(), sequence=seq).digitize(alphabet)
    block = pyhmmer.easel.DigitalSequenceBlock(alphabet, [digital])
    hits = list(pipeline.search_hmm(hmm, block))
    included = [h for h in hits if h.included]
    if not included:
        raise RuntimeError(f"no KR domain found in {slug}")
    # Sort the (hit, domain) pairs by per-domain bitscore; keep the top.
    domain_records = []
    for h in included:
        for d in h.domains:
            domain_records.append((d.score, h, d))
    domain_records.sort(key=lambda r: -r[0])
    score, top_hit, top_dom = domain_records[0]
    aln = top_dom.alignment
    aln_from, aln_to = aln.target_from, aln.target_to  # 1-based inclusive
    pad_from = max(1, aln_from - PAD_N)
    pad_to = min(len(seq), aln_to + PAD_C)
    return KRHit(
        slug=slug,
        full_length=len(seq),
        n_kr_hits=sum(len(h.domains) for h in included),
        kr_start=pad_from,
        kr_end=pad_to,
        kr_aln_start=aln_from,
        kr_aln_end=aln_to,
        bitscore=float(score),
        evalue=float(top_hit.evalue),
        kr_length=pad_to - pad_from + 1,
        notes=f"top hit bitscore={score:.1f}, evalue={top_hit.evalue:.1e}; "
              f"KR align {aln_from}-{aln_to} (+N{PAD_N}/+C{PAD_C} for ACP didomain)",
    )


def main() -> None:
    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(str(KR_HMM)) as fh:
        hmm = fh.read()
    pipeline = pyhmmer.plan7.Pipeline(alphabet)

    rows = []
    for fasta in sorted(SEQ_DIR.glob("*.fasta")):
        if fasta.name.endswith("_kr.fasta"):
            continue  # skip our own outputs if re-running
        slug = fasta.stem
        header, seq = load_fasta(fasta)
        try:
            hit = scan_kr(slug, header, seq, alphabet, hmm, pipeline)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(slug=slug, full_length=len(seq), kr_length=0,
                             bitscore=0.0, evalue=0.0, kr_start=0, kr_end=0,
                             kr_aln_start=0, kr_aln_end=0, n_kr_hits=0,
                             notes=f"FAIL: {exc}"))
            continue
        # Write KR+ACP didomain FASTA (the AF2 input; gives Vina the ACP-Ser:OG anchor).
        kr_seq = seq[hit.kr_start - 1 : hit.kr_end]
        out = SEQ_DIR / f"{slug}_kr_acp.fasta"
        out.write_text(
            f">{slug}_kr_acp {header} | KR+ACP didomain "
            f"(PF08659 KR aln {hit.kr_aln_start}-{hit.kr_aln_end}; "
            f"extracted {hit.kr_start}-{hit.kr_end} of {hit.full_length})\n{kr_seq}\n"
        )
        rows.append(asdict(hit))

    df = pd.DataFrame(rows)
    df.to_parquet(INDEX_PARQUET, index=False)
    print(f"{'slug':<22} {'full':>5} {'kr_len':>6}  {'aln':<10}  {'bitscore':>8}  {'evalue':>9}  notes")
    print("-" * 100)
    for _, r in df.iterrows():
        aln = f"{r.kr_aln_start}-{r.kr_aln_end}"
        print(f"{r.slug:<22} {r.full_length:>5} {r.kr_length:>6}  {aln:<10}  "
              f"{r.bitscore:>8.1f}  {r.evalue:>9.1e}  {r.notes[:50]}")
    print(f"\nwrote KR-only FASTAs + {INDEX_PARQUET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
