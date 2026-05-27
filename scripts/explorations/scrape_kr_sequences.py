"""EXPLORATORY (Direction A, Stage 2): polite scrape of per-KR-domain AA sequences from ClusterCAD.

Stage 1 is already done: the labelled parquet (clustercad_kr_examples.parquet, 1039 rows) carries the
ClusterCAD domain ids in `kr_domainid`. This fetches each id's AA sequence via the (now-fixed, AJAX-only)
domainLookup endpoint, writes the sequence dataset, and fills the sequence column the ESM-2 head needs.

Politeness (non-negotiable): >=1 s between network requests; per-id cache so re-runs hit the network only
for missing ids; backoff on 429/503 (in esm_data.fetch_domain_sequence); real project User-Agent + email.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from lpi.data.mibig import PROCESSED
from lpi.model.esm_data import DOMSEQ_CACHE, _session, fetch_domain_sequence

_DELAY = 1.1  # seconds between network requests (>= 1/s)
_KR = PROCESSED / "clustercad_kr_examples.parquet"
_SEQS = PROCESSED / "clustercad_kr_sequences.parquet"


def main() -> None:
    df = pd.read_parquet(_KR)
    ids = list(dict.fromkeys(str(d) for d in df["kr_domainid"].dropna()))
    print(f"KR domain ids to fetch: {len(ids)} (cache: {DOMSEQ_CACHE})", flush=True)

    sess = _session()
    rows, fetched, failed = [], 0, []
    for i, did in enumerate(ids, 1):
        cached = (DOMSEQ_CACHE / f"{did}.json").exists()
        seq = fetch_domain_sequence(did, session=sess)
        if not cached:
            time.sleep(_DELAY)               # pace only real network calls
        cache_file = DOMSEQ_CACHE / f"{did}.json"
        meta = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        if seq:
            fetched += 1
            rows.append({"kr_domainid": did, "name": meta.get("name", ""),
                         "start": meta.get("start", ""), "stop": meta.get("stop", ""),
                         "annotations": meta.get("annotations", ""), "aa_sequence": seq})
        else:
            failed.append(did)
        if i % 50 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}  fetched={fetched} failed={len(failed)}", flush=True)

    seqdf = pd.DataFrame(rows)
    seqdf.to_parquet(_SEQS, index=False)
    print(f"\nwrote {_SEQS} ({len(seqdf)} sequences)")

    # fill kr_sequence in the labelled parquet (what evaluate_head reads)
    seqmap = dict(zip(seqdf["kr_domainid"], seqdf["aa_sequence"]))
    df["kr_sequence"] = df["kr_domainid"].astype(str).map(seqmap)
    joined = df["kr_sequence"].notna().sum()
    df.to_parquet(_KR, index=False)
    print(f"joined sequences into {_KR.name}: {joined}/{len(df)} rows now have a sequence")
    if failed:
        print(f"FAILED ids ({len(failed)}): {failed[:20]}{' ...' if len(failed) > 20 else ''}")
    print("\nfirst 3 sequences (sanity -- should look like proteins):")
    for _, r in seqdf.head(3).iterrows():
        print(f"  {r['kr_domainid']} {r['name'][:50]}  len={len(r['aa_sequence'])}  {r['aa_sequence'][:60]}")


if __name__ == "__main__":
    main()
