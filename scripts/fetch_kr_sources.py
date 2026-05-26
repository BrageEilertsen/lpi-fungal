"""Track B background fetch: cache bacterial-cluster CDS protein FASTAs + the Pfam KR HMM.

Pure caching / IO, no fragile mapping logic -- runs in the background while Track A proceeds.
The sequence reconstruction (pyhmmer KR search + module-order mapping + ESM-2 head) is built
once these are local. ClusterCAD cluster ids are 'BGCxxxxxxx.N'; the MIBiG json is 'BGCxxxxxxx'.
"""
from __future__ import annotations

import gzip
import urllib.request

import pandas as pd

from lpi.data.genebudget import _fetch_cds_fasta, _loci_accessions
from lpi.data.mibig import RAW

EMAIL = "brageei@uio.no"


def main():
    df = pd.read_parquet("data/processed/clustercad_kr_examples.parquet")
    clusters = sorted(df["cluster"].dropna().unique())
    print(f"clusters: {len(clusters)}", flush=True)
    ok, accs_done = 0, set()
    for i, c in enumerate(clusters, 1):
        for acc in _loci_accessions(str(c).split(".")[0]):
            if acc in accs_done:
                continue
            accs_done.add(acc)
            if _fetch_cds_fasta(acc, email=EMAIL):
                ok += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(clusters)}] unique accessions ok={ok}", flush=True)
    print(f"CDS fetch done: {ok}/{len(accs_done)} accessions cached", flush=True)

    hmm = RAW / "PF08659_KR.hmm"
    if not hmm.exists():
        url = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/PF08659?annotation=hmm"
        try:
            raw = urllib.request.urlopen(url, timeout=60).read()
            data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            hmm.write_bytes(data)
            print(f"KR HMM cached: {hmm} ({len(data)} bytes)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"KR HMM fetch FAILED ({e}) -- will fix URL later", flush=True)
    else:
        print(f"KR HMM already present: {hmm}", flush=True)


if __name__ == "__main__":
    main()
