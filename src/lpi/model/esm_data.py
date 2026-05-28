"""ESM-2 head data: per-module KR-domain sequences + labels, for the 'beyond the rule' test.

Re-parses cached ClusterCAD cluster pages to pair each module's domains with their
ClusterCAD domainids, fetches the KR-domain amino-acid sequence via the /pks/domainLookup
endpoint (cached), and assembles two BACTERIAL prediction targets the domain cartoon
provably cannot do:
  * KR stereochemistry (R/S) -- domains carry no stereo signal (MLP 0.72 ~ majority 0.69);
  * inactive-domain detection -- did the present KR actually fire (reduction != KETO)? --
    the ~19% domain-present-but-inactive phenomenon, which gene content cannot determine.

The question: does sequence carry chemistry-relevant signal beyond gene content?
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from rdkit import Chem, RDLogger

from lpi.data.clustercad import CACHE
from lpi.data.mibig import PROCESSED, RAW
from lpi.model.clustercad_labels import reduction_state

RDLogger.DisableLog("rdApp.*")
DOMSEQ_CACHE = RAW / "clustercad_domainseq"
_PAIR_RE = re.compile(r'data-domainid="(\d+)"\s*data-domain="([A-Za-z]+)"')
_SMILES_RE = re.compile(r'smiles="([^"]+)"')


@dataclass
class KRExample:
    cluster: str
    kr_domainid: str
    reduction: str   # KETO / KR / DH / ER (structure-derived)
    stereo: str      # none / R / S
    kr_active: int   # 1 if the KR fired (reduction != KETO), else 0


def _modules_with_ids(page: str):
    """Yield (domain_id_by_name, intermediate_smiles) per module from a cached page."""
    events = []
    for m in _SMILES_RE.finditer(page):
        events.append((m.start(), "smiles", m.group(1)))
    for m in _PAIR_RE.finditer(page):
        events.append((m.start(), "domain", (m.group(2), m.group(1))))
    events.sort()
    pend: list[tuple[str, str]] = []
    prev_c = None
    for _pos, kind, val in events:
        if kind == "domain":
            pend.append(val)  # (name, id)
        else:
            smi = val
            if smi.endswith("C(=O)[S]") or "[S]" in smi:
                m = Chem.MolFromSmiles(smi)
                nc = sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()) if m else -1
                yield dict(pend), smi, prev_c, nc
                prev_c = nc
                pend = []
            else:
                pend = []


def build_kr_examples(cache_dir: Path = CACHE) -> list[KRExample]:
    out: list[KRExample] = []
    for html_file in sorted(cache_dir.glob("BGC*.html")):
        cluster = html_file.stem
        page = html_file.read_text()
        for dom_ids, smi, prev_c, nc in _modules_with_ids(page):
            if "KS" not in dom_ids or "KR" not in dom_ids:
                continue  # extension module that carries a KR domain
            if prev_c is None or nc - prev_c not in (2, 3):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            red, stereo = reduction_state(mol)
            out.append(KRExample(cluster, dom_ids["KR"], red, stereo,
                                 int(red != "KETO")))
    return out


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (LPI research; brageei@uio.no)"
    s.get("https://clustercad.jbei.org/pks/", timeout=30)  # set csrftoken cookie
    return s


_DOMLOOKUP = "https://clustercad.jbei.org/pks/domainLookup"
# The endpoint is AJAX-ONLY: it returns 404 unless the request carries the X-Requested-With header
# jQuery adds automatically. THAT missing header (not a dead endpoint) was the original "permanent
# 404" blocker -- verified 2026-05-27: same id returns 404 plain / 200 with the header.
_XHR_HEADERS = {"X-Requested-With": "XMLHttpRequest", "Referer": "https://clustercad.jbei.org/pks/"}


def fetch_domain_sequence(domainid: str, session=None, offline: bool = False) -> str | None:
    """Fetch a domain's amino-acid sequence via the ClusterCAD AJAX endpoint. Caches per id; backs off
    on 429/503; 404/other is not retried. The caller paces requests politely (>= 1 s apart)."""
    DOMSEQ_CACHE.mkdir(parents=True, exist_ok=True)
    cache = DOMSEQ_CACHE / f"{domainid}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("AAsequence") or None
    if offline:
        return None
    s = session or _session()
    backoff = 2.0
    for _ in range(4):
        try:
            r = s.get(_DOMLOOKUP, params={"domainid": domainid}, headers=_XHR_HEADERS, timeout=30)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                j = r.json()
                cache.write_text(json.dumps(j))
                return j.get("AAsequence") or None
            if r.status_code in (429, 503):       # server stressed -> exponential backoff
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            return None                            # 404 / other -> not retryable
        except Exception:  # noqa: BLE001
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return None


def fetch_all_sequences(examples: list[KRExample], offline: bool = False) -> dict[str, str]:
    s = None if offline else _session()
    seqs: dict[str, str] = {}
    for ex in examples:
        seq = fetch_domain_sequence(ex.kr_domainid, session=s, offline=offline)
        if seq:
            seqs[ex.kr_domainid] = seq
    return seqs


def write_examples(examples: list[KRExample], seqs: dict[str, str]) -> Path:
    import pandas as pd

    out = PROCESSED / "clustercad_kr_examples.parquet"
    rows = [{"cluster": e.cluster, "kr_domainid": e.kr_domainid, "reduction": e.reduction,
             "stereo": e.stereo, "kr_active": e.kr_active, "kr_sequence": seqs.get(e.kr_domainid)}
            for e in examples]
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out
