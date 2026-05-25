"""ClusterCAD 2.0 scraper/parser -> bacterial modular-PKS module dataset.

ClusterCAD has no JSON API, but each cluster page (/pks/<BGC>.N/) embeds, per module, the
running intermediate SMILES (with stereochemistry, thioester written as C(=O)[S]) and the
ordered catalytic domains. We scrape + cache the cluster pages and parse them into modules.

This is the abundant, clean, *colinear* bacterial corpus for the differential PoC: a module
is (domains, intermediate). Consecutive intermediates give the per-step structural change
(Δstructure); the domain set is the ΔBGC feature.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import requests

from lpi.data.mibig import PROCESSED, RAW

BASE = "https://clustercad.jbei.org"
CACHE = RAW / "clustercad_cache"
_UA = {"User-Agent": "Mozilla/5.0 (LPI-fungal research; brageei@uio.no)"}

# domains that matter for the per-step action; others (KS/ACP/docking/...) are structural
_DOMAIN_VOCAB = ("KS", "AT", "ACP", "KR", "DH", "ER", "cMT", "oMT", "nMT", "TE",
                 "CAL", "A", "C", "E", "R", "Red", "Ox", "MOX", "F", "X")


@dataclass
class Module:
    cluster: str
    index: int
    domains: tuple[str, ...]
    smiles: str  # the intermediate after this module (thioester C(=O)[S] or final product)


@dataclass
class ClusterRecord:
    accession: str
    modules: list[Module] = field(default_factory=list)


def cluster_list(timeout: int = 40) -> list[str]:
    """Accessions from the /pks/ browse page (e.g. 'BGC0000001.1')."""
    resp = requests.get(f"{BASE}/pks/", headers=_UA, timeout=timeout)
    resp.raise_for_status()
    return sorted(set(re.findall(r'href="/pks/(BGC\d+\.\d+)"', resp.text)))


def _fetch_cluster_html(accession: str, timeout: int = 40, offline: bool = False) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{accession}.html"
    if cache.exists():
        return cache.read_text()
    if offline:
        return None
    try:
        resp = requests.get(f"{BASE}/pks/{accession}/", headers=_UA, timeout=timeout)
        resp.raise_for_status()
        if resp.text and len(resp.text) > 1000:
            cache.write_text(resp.text)
            time.sleep(0.5)  # polite
            return resp.text
    except Exception:  # noqa: BLE001
        return None
    return None


# Each module block on the page carries a `smiles="..."` and a run of `domain="..."`
# attributes. We split the page on the smiles anchors and read the domains that follow each.
_SMILES_RE = re.compile(r'smiles="([^"]+)"')
_DOMAIN_RE = re.compile(r'(?<![a-z])domain="([^"]+)"')


def parse_cluster(accession: str, page: str) -> ClusterRecord:
    """Parse a cluster page into ordered modules (domains + intermediate SMILES).

    The first ``smiles`` on the page is the final product; subsequent ones are module
    intermediates in order. Domains appearing between intermediate i and i+1 belong to the
    module producing intermediate i+1.
    """
    rec = ClusterRecord(accession=accession)
    # positions of every smiles and domain occurrence, in document order
    events: list[tuple[int, str, str]] = []
    for m in _SMILES_RE.finditer(page):
        events.append((m.start(), "smiles", html.unescape(m.group(1))))
    for m in _DOMAIN_RE.finditer(page):
        d = m.group(1)
        if d in _DOMAIN_VOCAB:
            events.append((m.start(), "domain", d))
    events.sort()

    # Walk: collect domains since the last intermediate; the module's intermediate is the
    # next thioester smiles. The very first smiles (final product, not a thioester) is skipped.
    pending_domains: list[str] = []
    idx = 0
    for _pos, kind, val in events:
        if kind == "domain":
            pending_domains.append(val)
        else:  # smiles (an intermediate or the final product)
            is_thioester = val.endswith("C(=O)[S]") or "[S]" in val
            if is_thioester:
                rec.modules.append(Module(accession, idx, tuple(pending_domains), val))
                idx += 1
                pending_domains = []
            else:
                pending_domains = []  # reset at the (final-product) anchor
    return rec


def scrape(accessions: list[str], offline: bool = False) -> list[ClusterRecord]:
    out: list[ClusterRecord] = []
    for acc in accessions:
        page = _fetch_cluster_html(acc, offline=offline)
        if page:
            rec = parse_cluster(acc, page)
            if rec.modules:
                out.append(rec)
    return out


def write_parquet(records: list[ClusterRecord], out: Path | None = None) -> Path:
    import pandas as pd

    out = out or (PROCESSED / "clustercad_modules.parquet")
    rows = []
    for rec in records:
        prev = None
        for mod in rec.modules:
            rows.append({
                "cluster": rec.accession,
                "module_idx": mod.index,
                "domains": ";".join(mod.domains),
                "intermediate_smiles": mod.smiles,
                "prev_intermediate_smiles": prev,
            })
            prev = mod.smiles
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main() -> int:
    """Scrape all ClusterCAD clusters (cached) and write the module parquet."""
    recs = scrape(cluster_list())
    out = write_parquet(recs)
    n_mods = sum(len(r.modules) for r in recs)
    print(f"clusters parsed: {len(recs)}  modules: {n_mods}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
