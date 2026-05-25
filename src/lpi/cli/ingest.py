"""CLI: build the fungal PKS pairs parquet from MIBiG and report the usable count.

    python -m lpi.cli.ingest [--offline] [--email you@example.com]
"""

from __future__ import annotations

import argparse
import collections
import sys

from lpi.data import mibig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build fungal PKS pairs from MIBiG")
    ap.add_argument("--mibig-dir", default=str(mibig.DEFAULT_MIBIG_DIR))
    ap.add_argument("--email", default=None, help="contact email for NCBI E-utilities")
    ap.add_argument("--offline", action="store_true",
                    help="skip NCBI lineage lookup (genus-heuristic fungal detection)")
    args = ap.parse_args(argv)

    from pathlib import Path

    rows = mibig.build_pairs(Path(args.mibig_dir), email=args.email, offline=args.offline)
    out = mibig.write_parquet(rows)

    n = len(rows)
    parseable = sum(r["smiles_parse_ok"] for r in rows)
    by_subclass = collections.Counter(r["pks_subclass"] for r in rows)
    by_evidence = collections.Counter(r["fungal_evidence"] for r in rows)
    with_seqs = sum(r["n_embedded_sequences"] > 0 for r in rows)

    print(f"wrote {out}")
    print(f"usable fungal PKS pairs (PKS + fungal + structure): {n}")
    print(f"  with RDKit-parseable canonical SMILES           : {parseable}")
    print(f"  subclass breakdown    : {dict(by_subclass)}")
    print(f"  fungal evidence       : {dict(by_evidence)}")
    print(f"  with embedded protein seqs in MIBiG             : {with_seqs}/{n}")
    print("  (subclass 'unknown' = MIBiG lacks module/domain annotation; needs HMMER, Phase 0b)")
    print("  (entries lacking embedded sequences need NCBI fetch from loci accession, Phase 0b)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
