"""MIBiG ingestion: extract fungal PKS BGC -> product-structure pairs.

Pipeline (Phase 0, sub-task 0.1):
  1. Parse every MIBiG BGC JSON.
  2. Keep entries whose biosynthetic class includes PKS and that have a compound with
     a structure (SMILES).
  3. Classify the producing organism as fungal via NCBI taxonomy *lineage* (cached),
     not a genus guess.
  4. Infer PKS subclass (HR / NR / PR) from reductive-domain content *when MIBiG
     annotates modules*. Fungal iterative PKS are frequently NOT module-annotated in
     MIBiG (e.g. the lovastatin entry has zero modules), so subclass is 'unknown' for
     those pending HMMER-based domain detection (Phase 0b).
  5. Canonicalise product SMILES with RDKit.
  6. Emit a parquet of usable fungal PKS pairs and report the count.

Honesty note: the count reported is the real, restricted count (PKS + fungal +
resolvable structure), which the paper anticipates is well below the 314 figure.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from lpi.chem import mol as M

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MIBIG_DIR = RAW / "mibig_json_4.0"
LINEAGE_CACHE = RAW / "ncbi_lineage_cache.json"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# Reductive-cassette domain types as they appear in MIBiG module domain fields.
_REDUCTIVE_DOMAIN_KEYS = ("kr_domain", "dh_domain", "er_domain")


# --------------------------------------------------------------------------- parsing
def iter_bgc_files(mibig_dir: Path = DEFAULT_MIBIG_DIR):
    yield from sorted(mibig_dir.glob("BGC*.json"))


def _classes(d: dict) -> list[str]:
    return [c.get("class") for c in d.get("biosynthesis", {}).get("classes", [])]


def _first_structure_compound(d: dict) -> dict | None:
    for c in d.get("compounds", []):
        if c.get("structure"):
            return c
    return None


def _embedded_sequences(d: dict) -> dict[str, str]:
    seqs: dict[str, str] = {}
    genes = d.get("genes", {})
    for g in genes.get("to_add", []):
        if g.get("translation"):
            seqs[g.get("id", f"gene{len(seqs)}")] = g["translation"]
    return seqs


def infer_subclass(d: dict) -> str:
    """HR/NR/PR from module reductive-domain content; 'unknown' if not annotated."""
    modules = d.get("biosynthesis", {}).get("modules", [])
    pks_modules = [m for m in modules if str(m.get("type", "")).startswith("pks")]
    if not pks_modules:
        return "unknown"
    reductive = 0
    for m in pks_modules:
        if any(m.get(k) for k in _REDUCTIVE_DOMAIN_KEYS):
            reductive += 1
    # presence of a product-template (PT) cyclase => non-reducing aromatic
    cyclases = []
    for c in d.get("biosynthesis", {}).get("classes", []):
        cyclases += c.get("cyclases", []) or []
    if reductive == 0:
        return "NR"
    if reductive == len(pks_modules):
        return "HR"
    return "PR"


# ------------------------------------------------------------------ NCBI lineage
def _load_cache() -> dict[str, str]:
    if LINEAGE_CACHE.exists():
        return json.loads(LINEAGE_CACHE.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    LINEAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LINEAGE_CACHE.write_text(json.dumps(cache, indent=0))


def resolve_lineages(taxids: list[str], email: str | None = None,
                     batch_size: int = 200, offline: bool = False) -> dict[str, str]:
    """Resolve NCBI taxIds -> lineage string, cached on disk.

    If ``offline`` (or the network is unreachable), unresolved taxIds are left out and
    callers fall back to the genus heuristic.
    """
    cache = _load_cache()
    todo = [t for t in {str(t) for t in taxids} if t and t not in cache]
    if todo and not offline:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            params = {"db": "taxonomy", "id": ",".join(batch), "retmode": "xml"}
            if email:
                params["email"] = email
            try:
                resp = requests.get(EUTILS, params=params, timeout=30)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                for taxon in root.findall("Taxon"):
                    tid = taxon.findtext("TaxId")
                    lineage = taxon.findtext("Lineage") or ""
                    if tid:
                        cache[tid] = lineage
            except Exception as exc:  # noqa: BLE001
                logger.warning("NCBI lineage batch failed (%s); falling back", exc)
                break
            time.sleep(0.34)  # <= 3 req/s without an API key
        _save_cache(cache)
    return cache


# common filamentous-fungal genera, used only as a fallback when lineage is missing
_FUNGAL_GENERA = {
    "Aspergillus", "Penicillium", "Fusarium", "Beauveria", "Monascus", "Trichoderma",
    "Alternaria", "Cochliobolus", "Bipolaris", "Colletotrichum", "Magnaporthe",
    "Chaetomium", "Acremonium", "Talaromyces", "Emericella", "Hypocrea", "Stachybotrys",
    "Pestalotiopsis", "Aschersonia", "Metarhizium", "Claviceps", "Epichloe", "Daldinia",
    "Xylaria", "Sordaria", "Neurospora", "Botrytis", "Sclerotinia", "Phoma",
    "Cladosporium", "Ustilago", "Cercospora", "Pyricularia", "Parastagonospora",
    "Zymoseptoria", "Pochonia", "Tolypocladium", "Cordyceps", "Hypomyces",
}


def is_fungal(taxid: str | int | None, organism: str, lineage_cache: dict[str, str]) -> tuple[bool, str]:
    lineage = lineage_cache.get(str(taxid), "")
    if lineage:
        return ("Fungi" in lineage.split("; ")), "lineage"
    genus = organism.split()[0] if organism else ""
    return (genus in _FUNGAL_GENERA), "genus-fallback"


# --------------------------------------------------------------------- main extract
def build_pairs(mibig_dir: Path = DEFAULT_MIBIG_DIR, email: str | None = None,
                offline: bool = False) -> list[dict]:
    files = list(iter_bgc_files(mibig_dir))
    if not files:
        raise FileNotFoundError(
            f"No MIBiG JSON found in {mibig_dir}. See README for download steps."
        )

    # First pass: collect PKS-with-structure entries and their taxIds.
    candidates: list[tuple[Path, dict]] = []
    taxids: list[str] = []
    for f in files:
        d = json.loads(f.read_text())
        if "PKS" not in _classes(d):
            continue
        if _first_structure_compound(d) is None:
            continue
        candidates.append((f, d))
        tid = d.get("taxonomy", {}).get("ncbiTaxId")
        if tid:
            taxids.append(str(tid))

    lineage_cache = resolve_lineages(taxids, email=email, offline=offline)

    rows: list[dict] = []
    for f, d in candidates:
        org = d.get("taxonomy", {}).get("name", "")
        tid = d.get("taxonomy", {}).get("ncbiTaxId")
        fungal, how = is_fungal(tid, org, lineage_cache)
        if not fungal:
            continue
        comp = _first_structure_compound(d)
        try:
            canon = M.canonical_smiles(M.mol_from_smiles(comp["structure"]))
        except ValueError:
            canon = None  # keep the row but flag the unparseable structure
        seqs = _embedded_sequences(d)
        rows.append(
            {
                "bgc_id": d["accession"],
                "organism": org,
                "ncbi_taxid": tid,
                "fungal_evidence": how,
                "pks_subclass": infer_subclass(d),
                "n_modules": len(d.get("biosynthesis", {}).get("modules", [])),
                "compound_name": comp.get("name"),
                "product_smiles_raw": comp.get("structure"),
                "product_smiles_canonical": canon,
                "smiles_parse_ok": canon is not None,
                "database_ids": ";".join(comp.get("databaseIds", []) or []),
                "formula": comp.get("formula"),
                "mass": comp.get("mass"),
                "n_embedded_sequences": len(seqs),
                "completeness": d.get("completeness"),
                "quality": d.get("quality"),
            }
        )
    return rows


def write_parquet(rows: list[dict], out: Path | None = None) -> Path:
    import pandas as pd

    out = out or (PROCESSED / "fungal_pks_pairs.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out
