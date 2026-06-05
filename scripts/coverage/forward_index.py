"""Completion 1 -- Forward Sound Indexing: complete the inverse operator without weakening it.

The campaign's inverse BEAM is sound but INCOMPLETE: zearalenone (BGC0001057) is in-grammar under Theta_0
(kappa>=1, locked by test_zearalenone_program_within_scan_grammar) yet the beam reports kappa_beta=0 through
beta=200000 -- its carbon-closeness ranking lets C-MeT partials crowd out the long all-keto prefix at
depth 7. Widening beta changed 0/205 classifications, so WIDTH is not the fix.

The fix is to stop INVERTING. Run the executor FORWARD (it is sound): enumerate Theta_0 programs up to a
size bound (N cycles, Cmax carbons), compute each released formula analytically (pure integer arithmetic,
no RDKit), keep only programs whose formula can match a CORPUS target, execute those through the real
executor, and bucket by executed structure. Forward enumeration has no ranking, so it cannot starve: it is
COMPLETE BY CONSTRUCTION up to (Theta_0, N, Cmax). Two things the beam could never deliver follow:
  (1) it surfaces the beam-starved in-grammar cores (zearalenone), and
  (2) an EMPTY bucket up to the bound is a PROVABLE structural-gap certificate, not a heuristic one.

Re-orchestration, not new chemistry: reuses generate's exact primitives -- linear_acid_formula,
_formula_consistent, _formula_cho, core.exec -- the same forward+prefilter path `make duality` proves equals
the Theorem-1 dual set D(y;O) (460/460). Changes vs generate(): (a) gate ONE sweep against the SET of corpus
formulas (build-once / serve-all); (b) parallelize across cores (the sweep is embarrassingly parallel: the
combo space is partitioned by (starter, first cycle), no overlap, union = the whole space -- validated by
--selftest); (c) a per-exec SIGALRM timeout so a pathological core.exec cannot stall the run -- and, to keep
the gap certificate SOUND, any corpus formula a timed-out program was gated for is marked INCOMPLETE, so its
verdict degrades to inconclusive (we did not finish enumerating it), never to a false structural gap.

Verdicts (each RELATIVE to (Theta_0, N, Cmax) -- never "biologically impossible"):
  recovered              some enumerated program renders the deposited structure (beam-starved if the beam
                         had it unreachable). SOUND.
  sound-structural-gap   formula-feasible AND exhaustively enumerated (no timeouts at its formula), yet NO
                         program renders the deposited structure up to the bound. PROVABLE absence rel. to
                         (Theta_0, N) -- the certificate the starving beam categorically cannot give.
  formula-infeasible     no program's product formula can equal the target (analytic). SOUND. (rung-3.)
  inconclusive-budget    formula-feasible but the target needs cycles>N or carbons>Cmax, OR its formula had
                         a timed-out exec -- NOT enumerated to depth. Honest "don't know", never a gap.

Soundness guardrails (ground rules; not traded): executor and ~= (canonical-isomorphism via flat canonical
SMILES) UNCHANGED; no relaxed matching / no slack; the formula gate is computed WITHOUT RDKit and only
formula-matching programs are executed; every verdict prints N and Cmax; timeouts degrade to inconclusive,
never to a gap.

Single command: `make forward-index`  (PYTHONHASHSEED pinned).  Self-check: `... forward_index.py --selftest`.
"""
from __future__ import annotations

import csv
import dataclasses
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, Release
from lpi.data.mibig import PROCESSED
from lpi.executor import core as _core
from lpi.search.beam import _formula_cho, formula_feasible, linear_acid_formula
from lpi.search.reachability import CARBON_CAP, CARBON_MIN, _scan_spec

RDLogger.DisableLog("rdApp.*")

PARQUET = PROCESSED / "fungal_pks_pairs.parquet"
GENERABILITY_CSV = Path("results/generability_scan.csv")

# ---- Theta level: the parameterized build-loop --------------------------------------------------------
# THETA=0 is the cited Theta_0 baseline (== _scan_spec(); reproduces 117/11). THETA=n adds the
# witness-validated rung-2 build-loop operators registered through level n -- each is kept OUT of
# beam._ALL_RELEASES so the Theta_0 baseline stays reproducible, and opted in here. This keeps EVERY level
# reproducible from one command at HEAD (`make forward-index THETA=n`) and makes R_term-monotonicity a
# RUNNABLE check (`--monotone`: across levels, gaps only decrease, recovered only grows, zero demotions).
_THETA_EXTENSIONS = {
    1: (Release.PT_NAPHTHALENE,),   # T4HN proof-of-loop; witness-locked by tests/test_pt_naphthalene.py
}


def theta(level: int):
    """Theta_level = Theta_0 (_scan_spec) + the build-loop releases registered through `level`."""
    base = _scan_spec()
    extra = tuple(r for lv in range(1, level + 1) for r in _THETA_EXTENSIONS.get(lv, ()))
    return dataclasses.replace(base, releases=base.releases + extra) if extra else base


THETA_LEVEL = int(os.environ.get("THETA", "0"))
THETA = theta(THETA_LEVEL)

N_MAX = int(os.environ.get("NMAX", "9"))             # cycle bound: 9 covers C<=20 fully (acetyl + 9*2 = 20)
CMAX = int(os.environ.get("CMAX", str(CARBON_CAP)))  # carbon bound: default 20 (campaign cap); raise for the C>20 push

# Default config (Theta_0, N=9, C<=20) keeps the cited baseline filename untouched; any other
# config is tagged into its own file so a C>20 / Theta sweep can never clobber the cited result.
_suffix = (f"_theta{THETA_LEVEL}" if THETA_LEVEL else "") + \
          (f"_cmax{CMAX}_n{N_MAX}" if (CMAX, N_MAX) != (CARBON_CAP, 9) else "")
OUT_CSV = Path(f"results/forward_index{_suffix}.csv")
EXEC_TIMEOUT_S = 30        # per-exec wall limit: only TRUE pathological hangs hit it (normal exec <50ms)
CHO_STATUSES = ("reachable", "unreachable", "too_large", "budget")
_STARTER_C = {"acetyl": 2, "propionyl": 3, "butyryl": 4, "hexanoyl": 6}
_Target = namedtuple("_Target", "bgc_id name smiles status")

CSV_HEADER = ["bgc_id", "name", "C", "H", "O", "base_status", "generability_class",
              "forward_verdict", "formula_feasible", "in_range", "bucket_size_approx",
              "recovered_program"]

# worker globals (set by _worker_init under multiprocessing 'spawn')
_FSET: frozenset = frozenset()
_TIDX: dict = {}


def _flat(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def _cycle_opts() -> list:
    me_opts = (False, True) if THETA.allow_c_methyl else (False,)
    return [Cycle(reduction=r, c_methyl=me) for r in THETA.reductions for me in me_opts]


def _gate(lin: tuple, release: Release, fset: frozenset) -> bool:
    """Could a chain with linear-acid formula `lin` + this release reach ANY corpus formula? Set version
    of beam._formula_consistent: HYDROLYSIS loses no water; a cyclizing release loses k>=0 whole waters."""
    cl, hl, ol = lin
    if release == Release.HYDROLYSIS:
        return (cl, hl, ol) in fset
    k = 0
    while ol - k >= 0 and hl - 2 * k >= 0:
        if (cl, hl - 2 * k, ol - k) in fset:
            return True
        k += 1
    return False


def _gated_targets(lin: tuple, release: Release, fset: frozenset) -> list:
    """The corpus formulas (lin, release) is consistent with -- used only on a timeout, to mark those
    formulas INCOMPLETE (sound: a program we couldn't finish executing may have produced one of them)."""
    cl, hl, ol = lin
    if release == Release.HYDROLYSIS:
        return [(cl, hl, ol)] if (cl, hl, ol) in fset else []
    out = []
    k = 0
    while ol - k >= 0 and hl - 2 * k >= 0:
        t = (cl, hl - 2 * k, ol - k)
        if t in fset:
            out.append(t)
        k += 1
    return out


def _enumerate_from(starter_c: int, first: Cycle):
    """All combos (len 1..N_MAX, program carbons <= CMAX) whose FIRST cycle is `first`. Union over all
    first-options (per starter) = the whole combo space, with no overlap (every combo has a unique first
    cycle) -- the parallel partition. DFS carbon-pruned: adding a cycle only adds carbons."""
    opts = _cycle_opts()
    opt_c = [3 if o.c_methyl else 2 for o in opts]
    fc = 3 if first.c_methyl else 2
    if starter_c + fc > CMAX:
        return
    combo = [first]
    yield tuple(combo)

    def rec(carbons: int):
        for o, oc in zip(opts, opt_c):
            nc = carbons + oc
            if nc > CMAX:
                continue
            combo.append(o)
            yield tuple(combo)
            if len(combo) < N_MAX:
                yield from rec(nc)
            combo.pop()

    if len(combo) < N_MAX:
        yield from rec(starter_c + fc)


def _enumerate_combos(starter_c: int):
    """Whole (unpartitioned) combo space for a starter -- for --selftest equality vs the partition."""
    for first in _cycle_opts():
        yield from _enumerate_from(starter_c, first)


def _on_alarm(signum, frame):
    raise TimeoutError


def _worker_init(fset: frozenset, tidx: dict) -> None:
    global _FSET, _TIDX
    _FSET, _TIDX = fset, tidx
    signal.signal(signal.SIGALRM, _on_alarm)


def _exec_chunk(chunk: tuple) -> tuple:
    """One parallel unit: (starter, first-cycle index). Enumerate its sub-space, gate, execute (with a
    per-exec timeout), and bucket. Returns (recovered{bgc:witness}, incomplete{cho}, bucket_counts{cho:n},
    n_exec, n_skip, n_gated). recovered/incomplete are exact across the union of chunks."""
    starter, first_idx = chunk
    opts = _cycle_opts()
    first = opts[first_idx]
    sc = _STARTER_C.get(starter, 2)
    recovered: dict = {}
    incomplete: set = set()
    bucket: dict = defaultdict(set)
    n_exec = n_skip = n_gated = 0
    for combo in _enumerate_from(sc, first):
        lin = linear_acid_formula(Program(starter, combo))
        for rel in THETA.releases:
            if not _gate(lin, rel, _FSET):
                continue
            n_gated += 1
            prog = Program(starter, combo, rel)
            signal.setitimer(signal.ITIMER_REAL, EXEC_TIMEOUT_S)
            try:
                mol = _core.exec(prog)
            except TimeoutError:
                n_skip += 1
                for t in _gated_targets(lin, rel, _FSET):
                    incomplete.add(t)          # sound: un-finished -> cannot certify this formula's gap
                continue
            except Exception:  # noqa: BLE001  (non-rendering program contributes nothing -- sound)
                continue
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            n_exec += 1
            cho = _formula_cho(mol)
            if cho not in _FSET:
                continue                       # exact actual-formula gate (pins real waters lost)
            flat = Chem.MolToSmiles(mol, isomericSmiles=False)
            bucket[cho].add(flat)
            bgc = _TIDX.get((cho, flat))
            if bgc is not None and bgc not in recovered:
                recovered[bgc] = repr(prog)
    return recovered, incomplete, {c: len(s) for c, s in bucket.items()}, n_exec, n_skip, n_gated


def load_targets() -> list:
    """The 214 CHO in-scope targets, loaded DIRECTLY from the parquet (no beam): exactly scan_target's
    in-scope filter -- parse OK + single fragment + elements in {H,C,O} + C>=CARBON_MIN -- which is the
    reachable+unreachable+too_large set. The forward index does its OWN reachability, so the beam scan
    (slow, single-threaded, opaque) is pure overhead here. base_status is joined from the committed
    generability scan (its 205 == the in-scope cores); the remaining 9 CHO cores are the reachable ones."""
    print("[1/3] load 214 CHO targets directly from the parquet (no beam scan) ...", flush=True)
    gen_status = {}
    if GENERABILITY_CSV.exists():
        with GENERABILITY_CSV.open() as f:
            for row in csv.DictReader(f):
                gen_status[row["bgc_id"]] = row["base_status"]
    df = pd.read_parquet(PARQUET)
    targets = []
    for _, r in df.iterrows():
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        if not smi:
            continue
        try:
            mol = M.mol_from_smiles(smi)
        except Exception:  # noqa: BLE001
            continue
        if mol is None or len(Chem.GetMolFrags(mol)) > 1:
            continue
        if not all(a.GetAtomicNum() in (1, 6, 8) for a in mol.GetAtoms()):
            continue
        c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if c < CARBON_MIN:
            continue
        bgc = r["bgc_id"]
        status = gen_status.get(bgc) or ("too_large" if c > CARBON_CAP else "reachable")
        targets.append(_Target(bgc, str(r.get("compound_name")), smi, status))
    n_reach = sum(1 for t in targets if t.status == "reachable")
    n_tl = sum(1 for t in targets if t.status == "too_large")
    print(f"  CHO targets: {len(targets)} (reachable={n_reach}, "
          f"unreachable={len(targets) - n_reach - n_tl}, too_large={n_tl})", flush=True)
    return targets


def _generability_classes() -> dict:
    if not GENERABILITY_CSV.exists():
        return {}
    with GENERABILITY_CSV.open() as f:
        return {row["bgc_id"]: row["class"] for row in csv.DictReader(f)}


def _selftest() -> None:
    """The parallelization-specific correctness: the (starter, first-cycle) partition reconstructs the
    serial combo space exactly, with no overlap. (exec/gate/recovery logic is unchanged from the
    duality-validated generate primitives.)"""
    global N_MAX, CMAX
    N_MAX, CMAX = 4, 12
    opts = _cycle_opts()
    for starter in THETA.starters:
        sc = _STARTER_C[starter]
        serial = set(tuple((c.reduction, c.c_methyl) for c in cb) for cb in _enumerate_combos(sc))
        par: set = set()
        for i in range(len(opts)):
            for cb in _enumerate_from(sc, opts[i]):
                key = tuple((c.reduction, c.c_methyl) for c in cb)
                assert key not in par, f"OVERLAP in {starter}: {key}"
                par.add(key)
        assert par == serial, f"{starter}: partition {len(par)} != serial {len(serial)}"
        print(f"  {starter:9s}: partition == serial, {len(serial)} combos, no overlap  OK")
    print("SELFTEST PASS: chunked enumeration is complete + non-overlapping (parallel == serial).")


def _monotone() -> None:
    """Runnable R_term-monotonicity check across Theta levels (Prop 16.4c as a test): read the per-level
    forward_index CSVs and assert the build-loop only ADDS reach -- gaps non-increasing, recovered
    non-decreasing, and ZERO demotions (no core recovered at a lower level becomes a gap at a higher one).
    Run the levels first (`make forward-index THETA=0`, `... THETA=1`, ...), then `... --monotone`."""
    levels: dict = {}
    paths = [Path("results/forward_index.csv"), *sorted(Path("results").glob("forward_index_theta*.csv"))]
    for p in paths:
        if not p.exists():
            continue
        lvl = 0 if p.name == "forward_index.csv" else int(p.stem.split("theta")[1])
        rows = list(csv.DictReader(p.open()))
        rec = {r["bgc_id"] for r in rows if r["forward_verdict"] == "recovered"}
        gaps = sum(1 for r in rows if r["forward_verdict"] in ("sound-structural-gap", "formula-infeasible"))
        incon = sum(1 for r in rows if r["forward_verdict"] == "inconclusive-budget")
        levels[lvl] = (rec, gaps, incon, len(rows))
    print(f"R_term-MONOTONICITY across Theta levels {sorted(levels)} (gaps = sound-structural-gap + formula-infeasible):")
    ok, prev = True, None
    for lvl in sorted(levels):
        rec, gaps, incon, n = levels[lvl]
        line = f"  Theta_{lvl}: recovered={len(rec):>3d}  certified-gaps={gaps:>3d}  inconclusive={incon:>3d}  (of {n})"
        if prev is not None:
            prec, pgaps, pincon, _ = prev
            demoted = prec - rec
            grew, shrank, clean = len(rec) >= len(prec), gaps <= pgaps, not demoted
            ok = ok and grew and shrank and clean
            line += (f"  | recovered {len(rec) - len(prec):+d} {'OK' if grew else 'FAIL'}, "
                     f"gaps {gaps - pgaps:+d} {'OK' if shrank else 'FAIL'}, "
                     f"demotions {len(demoted)} {'OK' if clean else 'FAIL'}")
            if demoted:
                line += f"  DEMOTED {sorted(demoted)} -- check _THETA_EXTENSIONS is base UNION {{op}}, not base-minus"
        print(line)
        prev = (rec, gaps, incon, n)
    print(f"\n  R_term non-decreasing in Theta (no silent prune): {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


def _drycount() -> None:
    """Tractability probe for the C>20 push. Brute forward enumeration grows ~(branching)^N under the
    carbon cap; resolving C>20 cores needs many more cycles than the C<=20 sweep (a C=2k core needs ~k-1
    cycles), so the space can explode super-exponentially. This counts RAW combos at the current
    (CMAX, N_MAX) WITHOUT gating/executing -- the enumeration wall is the binding cost -- and verdicts
    whether a full run is overnight-feasible BEFORE any daemon is committed.
    Run: CMAX=<c> NMAX=<n> python scripts/coverage/forward_index.py --drycount"""
    import time as _t
    CAP = 60_000_000  # raw-combo abort: enumeration alone beyond this is not an overnight job
    print(f"  drycount: CMAX={CMAX}, N_MAX={N_MAX}, starters={list(THETA.starters)}, "
          f"cycle options/step={len(_cycle_opts())}", flush=True)
    raw = 0
    t0 = _t.time()
    aborted = False
    for starter in THETA.starters:
        sc = _STARTER_C[starter]
        for first in _cycle_opts():
            for _ in _enumerate_from(sc, first):
                raw += 1
                if raw >= CAP:
                    aborted = True
                    break
            if aborted:
                break
        if aborted:
            break
    dt = max(_t.time() - t0, 1e-9)
    print(f"  raw combos: {raw:,}{'  (HIT CAP -- not exhaustive)' if aborted else ''}  "
          f"in {dt:.1f}s  ({raw / dt:,.0f}/s)", flush=True)
    if aborted:
        print(f"  VERDICT: INTRACTABLE at CMAX={CMAX}, N={N_MAX} -- raw enumeration alone exceeds "
              f"{CAP:,} combos. A brute forward sweep here is not overnight-feasible.", flush=True)
    else:
        print(f"  VERDICT: ENUMERABLE -- {raw:,} raw combos. Full run (gate+exec) is feasible; "
              f"gated execs are a small fraction (Theta_0: 7.26M gated).", flush=True)


def main() -> None:
    if "--selftest" in sys.argv[1:]:
        _selftest()
        return
    if "--monotone" in sys.argv[1:]:
        _monotone()
        return
    if "--drycount" in sys.argv[1:]:
        _drycount()
        return
    t0 = time.time()
    ext = [r.name for lv in range(1, THETA_LEVEL + 1) for r in _THETA_EXTENSIONS.get(lv, ())]
    print(f"Completion 1 -- forward sound indexing (parallel; THETA level {THETA_LEVEL})\n", flush=True)
    print(f"Theta_{THETA_LEVEL}: starters={THETA.starters}, releases={len(THETA.releases)} modes"
          f"{' +' + '+'.join(ext) if ext else ' (base)'}, c_methyl={THETA.allow_c_methyl}; "
          f"N={N_MAX}, Cmax={CMAX}; exec_timeout={EXEC_TIMEOUT_S}s -> {OUT_CSV.name}\n", flush=True)

    targets = load_targets()
    gen_class = _generability_classes()

    fmap, tidx, fset_build = {}, {}, set()
    for r in targets:
        m = M.mol_from_smiles(r.smiles)
        cho = _formula_cho(m)
        flat = _flat(r.smiles)
        fmap[r.bgc_id] = (cho, flat, r.name, r.status)
        tidx[(cho, flat)] = r.bgc_id
        fset_build.add(cho)
    fset = frozenset(c for c in fset_build if formula_feasible(c, THETA, max_cycles=c[0] // 2 + 1))
    print(f"  corpus formula gate: {len(fset)} feasible of {len(fset_build)} distinct (C,H,O)\n", flush=True)

    chunks = [(s, i) for s in THETA.starters for i in range(len(_cycle_opts()))]
    k = max(2, (os.cpu_count() or 4) - 2)
    print(f"[2/3] parallel forward sweep: {len(chunks)} chunks over {k} workers ...", flush=True)
    recovered: dict = {}
    incomplete: set = set()
    bucket_counts: dict = defaultdict(int)
    n_exec = n_skip = n_gated = done = 0
    pool = mp.Pool(k, initializer=_worker_init, initargs=(fset, tidx))
    try:
        for rec, inc, bc, ne, ns, ng in pool.imap_unordered(_exec_chunk, chunks):
            recovered.update(rec)
            incomplete |= inc
            for c, n in bc.items():
                bucket_counts[c] += n
            n_exec += ne
            n_skip += ns
            n_gated += ng
            done += 1
            print(f"    chunk {done}/{len(chunks)} | exec={n_exec:,} gated={n_gated:,} skip={n_skip} "
                  f"recovered={len(recovered)}/{len(targets)} incomplete_formulas={len(incomplete)} "
                  f"t={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        pool.close()      # graceful: no more tasks, let workers drain + exit (avoids terminate()'s BrokenPipe)
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    sweep_s = time.time() - t0
    print(f"  swept: executed={n_exec:,} gated={n_gated:,} timed_out={n_skip} "
          f"recovered={len(recovered)}/{len(targets)}  ({sweep_s:.0f}s)\n", flush=True)

    print("[3/3] classify + write ...", flush=True)
    # a deposited structure can be shared by >1 BGC (duplicate metabolites -> tidx dedups (cho,flat));
    # recovering that structure via ANY one BGC recovers them all. Expand recovered over identical (cho,flat).
    rec_structs = {fmap[b][:2]: prog for b, prog in recovered.items()}  # (cho,flat) -> witness program
    rows = []
    for bgc, (cho, flat, name, status) in fmap.items():
        C, H, O = cho
        feasible = formula_feasible(cho, THETA, max_cycles=C // 2 + 1)
        in_range = (C <= CMAX) and (((C - 2 + 1) // 2) <= N_MAX)
        is_rec = (cho, flat) in rec_structs
        if not feasible:
            verdict = "formula-infeasible"
        elif is_rec:
            verdict = "recovered"
        elif (not in_range) or (cho in incomplete):
            verdict = "inconclusive-budget"
        else:
            verdict = "sound-structural-gap"
        rows.append([bgc, str(name), C, H, O, status, gen_class.get(bgc, ""), verdict,
                     feasible, in_range, bucket_counts.get(cho, 0), rec_structs.get((cho, flat), "")])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    _summary(rows, n_exec, n_gated, n_skip, len(incomplete), sweep_s)
    print(f"\nwrote {OUT_CSV}  ({time.time() - t0:.0f}s total)", flush=True)


def _summary(rows: list, n_exec: int, n_gated: int, n_skip: int, n_incomplete: int, sweep_s: float) -> None:
    v = Counter(r[7] for r in rows)
    print(f"\n{'=' * 72}")
    print(f"COMPLETION 1 -- FORWARD SOUND INDEX of {len(rows)} CHO targets "
          f"(Theta_0, N={N_MAX}, Cmax={CMAX}):\n")
    for key in ("recovered", "sound-structural-gap", "formula-infeasible", "inconclusive-budget"):
        print(f"  {key:24s}: {v.get(key, 0):>4d}")
    print(f"  {'-' * 50}")
    reach = [r for r in rows if r[5] == "reachable"]
    reach_ok = sum(1 for r in reach if r[7] == "recovered")
    print(f"  A0 parity   : {reach_ok}/{len(reach)} reachable cores recovered "
          f"({'PASS' if reach_ok == len(reach) else 'FAIL -- bug, stop'})")
    zea = [r for r in rows if r[0] == "BGC0001057"]
    if zea:
        print(f"  A1 headline : zearalenone (BGC0001057) -> {zea[0][7]} "
              f"({'PASS' if zea[0][7] == 'recovered' else 'FAIL -- investigate'})")
    struct169 = [r for r in rows if r[6] == "coverage-limited-structural"]
    if struct169:
        s = Counter(r[7] for r in struct169)
        print(f"\n  A2 -- re-partition of the beam's {len(struct169)} `coverage-limited-structural`:")
        print(f"        recovered (beam-starved, in-grammar) : {s.get('recovered', 0)}")
        print(f"        sound-structural-gap (genuine)       : {s.get('sound-structural-gap', 0)}")
        print(f"        inconclusive-budget (C>{CMAX} / N>{N_MAX}) : {s.get('inconclusive-budget', 0)}")
        print(f"        formula-infeasible (reclassified)    : {s.get('formula-infeasible', 0)}")
    gen_formula = [r for r in rows if r[6] == "coverage-limited-formula"]
    gf_ok = sum(1 for r in gen_formula if r[7] == "formula-infeasible")
    if gen_formula:
        print(f"\n  consistency : {gf_ok}/{len(gen_formula)} of generability's coverage-limited-formula "
              f"re-confirm formula-infeasible ({'PASS' if gf_ok == len(gen_formula) else 'MISMATCH'})")
    print(f"\n  A4 efficiency: one parallel sweep served all {len(rows)} targets in {sweep_s:.0f}s, "
          f"{n_exec:,} executor calls ({n_gated:,} formula-gated).")
    print(f"  exec timeouts: {n_skip} (-> {n_incomplete} formulas marked incomplete -> their gaps degrade "
          f"to inconclusive, never false gaps)")
    print(f"{'=' * 72}", flush=True)


if __name__ == "__main__":
    main()
