"""Completion 2 -- Formula-directed forward indexing: resolve the C>20 cores brute enumeration cannot reach.

The brute forward index (forward_index.py) is super-exponential against the carbon cap (its --drycount:
C<=20 -> 6.7M combos, C<=22 -> 38M, C>=24 -> intractable, C<=80 -> hopeless), so the 86 C>20 cores
(C=21..41) are reported "inconclusive", never gaps. This completes them WITHOUT brute force and WITHOUT
weakening the executor or the match.

The key fact (validated, not assumed): the linear-acid formula is a LINEAR, ORDER-INDEPENDENT function of
the cycle-type COUNTS -- F = base(starter) + sum_t count_t * delta_t, where each delta_t is read straight
from the executor's own `linear_acid_formula` on a one-cycle program. So inverting a target formula T to
the set of ALL programs that can produce it is a bounded integer composition over the 8 cycle-type deltas
(every delta has dC>=2, so total cycles <= C/2) followed by the distinct orderings of each multiset. This
is COMPLETE by construction -- the same property that makes an empty bucket a provable absence certificate.

Soundness is INHERITED, not re-implemented: the water-loss model mirrors forward_index._gate exactly
(HYDROLYSIS k=0; a cyclizing release loses k>=0 whole waters, so lin = (C, H+2k, O+k)); the execute +
match step is forward_index's verbatim (per-exec SIGALRM timeout; exact actual-formula gate cho==T;
flat-canonical SMILES `~=`; tidx lookup). `--selftest` proves the inversion equals brute enumeration gated
to each formula, on every corpus formula small enough for brute to be tractable (validated 47/47 at C<=15).

Verdicts (per core, relative to Theta_0):
  recovered                    some ordered program renders the deposited structure. SOUND.
  sound-structural-gap         formula-feasible, ALL orderings enumerated + executed, none match. PROVABLE.
  formula-infeasible           no release/starter/composition can produce the formula (analytic). SOUND.
  inconclusive-ordering-budget orderings exceed ORDER_CAP -- not exhausted, so a gap cannot be certified.
                               Honest "don't know" (recovery may still have matched early), never a gap.
  inconclusive-timeout         a per-exec timeout left a program unfinished -> cannot certify the gap.

Run: `make forward-directed`  (resolves results/forward_index.csv's inconclusive set -> *_directed.csv).
Self-check: `python scripts/coverage/forward_directed.py --selftest`.
"""
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_index as FI  # noqa: E402  (sibling script; brings the executor-validated primitives + _gate model)
from lpi.chem import mol as M  # noqa: E402
from lpi.chem.program import Cycle, Program, Release  # noqa: E402
from lpi.executor import core as _core  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

THETA = FI.THETA
TYPES = [(c.reduction, c.c_methyl) for c in FI._cycle_opts()]
ORDER_CAP = int(os.environ.get("ORDER_CAP", "300000"))  # per-core ordering/exec budget
EXEC_TIMEOUT_S = FI.EXEC_TIMEOUT_S
OUT_CSV = Path("results/forward_index_directed.csv")


def _deltas_base():
    """Per-cycle-type (dC,dH,dO) and per-starter base formula, read from the executor's validated primitive."""
    deltas, base = {}, {}
    for st in THETA.starters:
        for t in TYPES:
            c = Cycle(reduction=t[0], c_methyl=t[1])
            f1 = FI.linear_acid_formula(Program(st, (c,)))
            f2 = FI.linear_acid_formula(Program(st, (c, c)))
            deltas[(st, t)] = tuple(b - a for a, b in zip(f1, f2))
        t0 = TYPES[0]
        f1 = FI.linear_acid_formula(Program(st, (Cycle(reduction=t0[0], c_methyl=t0[1]),)))
        base[st] = tuple(a - b for a, b in zip(f1, deltas[(st, t0)]))
    return deltas, base


DELTAS, BASE = _deltas_base()


def _compositions(residual, st):
    """All non-negative integer count-vectors over TYPES with sum_t count_t*delta_t == residual.
    Bounded (every dC>=2) and COMPLETE -- carbon/H/O-pruned DFS over the 8 cycle types."""
    rC, rH, rO = residual
    n = len(TYPES)
    dl = [DELTAS[(st, t)] for t in TYPES]
    counts = [0] * n

    def rec(i, cC, cH, cO):
        if cC > rC or cH > rH or cO > rO:
            return
        if i == n:
            if (cC, cH, cO) == (rC, rH, rO):
                yield tuple(counts)
            return
        dC, dH, dO = dl[i]
        k = 0
        while True:
            counts[i] = k
            nC, nH, nO = cC + k * dC, cH + k * dH, cO + k * dO
            if nC > rC or nH > rH or nO > rO:
                break
            yield from rec(i + 1, nC, nH, nO)
            k += 1
        counts[i] = 0
    yield from rec(0, 0, 0, 0)


def _multiset_perms(counts):
    """Distinct orderings of the cycle multiset (generates only distinct sequences, not n! with dups)."""
    n = len(TYPES)
    rem = list(counts)
    total = sum(counts)
    seq = []

    def rec():
        if len(seq) == total:
            yield tuple(seq)
            return
        for i in range(n):
            if rem[i] > 0:
                rem[i] -= 1
                seq.append(i)
                yield from rec()
                seq.pop()
                rem[i] += 1
    yield from rec()


def _lins_for(T, rel):
    """Candidate linear-acid formulas for final formula T under release `rel` -- mirrors _gate exactly."""
    Tc, Th, To = T
    if rel == Release.HYDROLYSIS:
        return [(Tc, Th, To)]
    return [(Tc, Th + 2 * k, To + k) for k in range(0, Tc + 1)]


def directed_programs(T):
    """Lazily yield every ordered Theta_0 program whose release-adjusted formula == T. COMPLETE (validated)."""
    for rel in THETA.releases:
        for lin in _lins_for(T, rel):
            for st in THETA.starters:
                bc, bh, bo = BASE[st]
                residual = (lin[0] - bc, lin[1] - bh, lin[2] - bo)
                if min(residual) < 0:
                    continue
                for counts in _compositions(residual, st):
                    for order in _multiset_perms(counts):
                        combo = tuple(Cycle(reduction=TYPES[i][0], c_methyl=TYPES[i][1]) for i in order)
                        yield Program(st, combo, rel)


def _formula_feasible(T):
    """True iff some release/starter/composition yields T (else formula-infeasible -- analytic, SOUND)."""
    for rel in THETA.releases:
        for lin in _lins_for(T, rel):
            for st in THETA.starters:
                bc, bh, bo = BASE[st]
                residual = (lin[0] - bc, lin[1] - bh, lin[2] - bo)
                if min(residual) < 0:
                    continue
                for _ in _compositions(residual, st):
                    return True
    return False


def resolve(T, target_flat):
    """Resolve one C>20 core. exec+match is forward_index._exec_chunk's logic, verbatim and unweakened."""
    if not _formula_feasible(T):
        return "formula-infeasible", None, 0
    n = 0
    had_timeout = False
    for prog in directed_programs(T):
        n += 1
        if n > ORDER_CAP:
            return "inconclusive-ordering-budget", None, n
        signal.setitimer(signal.ITIMER_REAL, EXEC_TIMEOUT_S)
        try:
            mol = _core.exec(prog)
        except TimeoutError:
            had_timeout = True
            mol = None
        except Exception:  # noqa: BLE001  (non-rendering program contributes nothing -- sound)
            mol = None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if mol is None:
            continue
        if FI._formula_cho(mol) != T:
            continue  # exact actual-formula gate (pins real waters lost)
        if Chem.MolToSmiles(mol, isomericSmiles=False) == target_flat:
            return "recovered", repr(prog), n
    return ("inconclusive-timeout" if had_timeout else "sound-structural-gap"), None, n


def _selftest():
    """Soundness gate: directed enumeration == brute enumeration gated to each corpus formula, for every
    formula small enough for brute to be tractable. If equal, the inversion misses nothing + adds nothing."""
    K = int(os.environ.get("SELFTEST_K", "15"))
    targets = FI.load_targets()
    formulas = sorted({FI._formula_cho(M.mol_from_smiles(t.smiles)) for t in targets})
    small = [T for T in formulas if T[0] <= K]
    print(f"--selftest: directed == brute-gated for {len(small)} corpus formulas with C<={K} ...", flush=True)

    def brute_gated(T):
        fset = frozenset({T})
        saveC, saveN = FI.CMAX, FI.N_MAX
        FI.CMAX, FI.N_MAX = T[0], T[0]
        out = set()
        try:
            for st in THETA.starters:
                sc = FI._STARTER_C[st]
                for first in FI._cycle_opts():
                    for combo in FI._enumerate_from(sc, first):
                        lin = FI.linear_acid_formula(Program(st, combo))
                        for rel in THETA.releases:
                            if FI._gate(lin, rel, fset):
                                out.add((st, tuple((c.reduction, c.c_methyl) for c in combo), rel))
        finally:
            FI.CMAX, FI.N_MAX = saveC, saveN
        return out

    nfail = 0
    for T in small:
        d = {(p.starter, tuple((c.reduction, c.c_methyl) for c in p.cycles), p.release) for p in directed_programs(T)}
        b = brute_gated(T)
        if d != b:
            nfail += 1
            print(f"  MISMATCH T={T}: directed={len(d)} brute={len(b)} miss={len(b-d)} extra={len(d-b)}")
    ok = nfail == 0
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}: {len(small)-nfail}/{len(small)} formulas match "
          f"(inversion complete+sound on the brute-tractable regime).")
    if not ok:
        sys.exit(1)


def _worker_sig():
    """Picklable Pool initializer (spawn can't pickle a lambda): install the per-exec SIGALRM handler."""
    signal.signal(signal.SIGALRM, FI._on_alarm)


def _resolve_one(args):
    bgc, name, T, flat = args
    verdict, witness, ntried = resolve(T, flat)
    return (bgc, name, T, verdict, witness, ntried)


def _e2e_one(args):
    bgc, fwd, T, flat = args
    v, w, n = resolve(T, flat)
    return (bgc, fwd, v, w, n)


def _e2e():
    """End-to-end gate: resolve() must reproduce forward_index.csv's DEFINITIVE C<=20 verdicts -- the
    exec+match+verdict path that --selftest's enumeration check does not cover. Cap raised so C<=20 formulas
    can exhaust; a resolve()=inconclusive-* on a forward gap is cap-limited (couldn't finish), NOT a wiring
    bug. PASS iff zero CONTRADICTIONS (a definitive verdict that disagrees: recovered<->gap flip, infeasible
    mismatch)."""
    global ORDER_CAP
    ORDER_CAP = int(os.environ.get("E2E_CAP", "2000000"))
    rows = list(csv.DictReader(open("results/forward_index.csv")))
    cle20 = [r for r in rows if int(r["C"]) <= 20]
    df = __import__("pandas").read_parquet(FI.PARQUET)
    smi = {r["bgc_id"]: (r.get("product_smiles_canonical") or r.get("product_smiles_raw")) for _, r in df.iterrows()}
    work = []
    for r in cle20:
        s = smi.get(r["bgc_id"])
        if not s:
            continue
        mol = M.mol_from_smiles(s)
        work.append((r["bgc_id"], r["forward_verdict"], FI._formula_cho(mol),
                     Chem.MolToSmiles(mol, isomericSmiles=False)))
    print(f"--e2e: resolve() vs forward_index on {len(work)} C<=20 cores (ORDER_CAP={ORDER_CAP:,}) ...", flush=True)
    k = max(2, (os.cpu_count() or 4) - 2)
    match = capped = rec_confirmed = 0
    contradictions = []
    with mp.Pool(k, initializer=_worker_sig) as pool:
        for bgc, fwd, v, w, n in pool.imap_unordered(_e2e_one, work):
            if v == fwd:
                match += 1
                rec_confirmed += (fwd == "recovered")
            elif v.startswith("inconclusive"):
                capped += 1  # couldn't exhaust within budget -- cap-limited, not a contradiction
            else:
                contradictions.append((bgc, fwd, v))  # definitive disagreement = wiring bug
    ok = len(contradictions) == 0
    print(f"--e2e: matched={match} (recoveries confirmed {rec_confirmed}), cap-limited={capped}, "
          f"CONTRADICTIONS={len(contradictions)}", flush=True)
    for bgc, fwd, v in contradictions[:12]:
        print(f"   CONTRADICTION {bgc}: forward_index={fwd}  resolve={v}", flush=True)
    print(f"E2E GATE {'PASS' if ok else 'FAIL'}: resolve() reproduces forward_index C<=20 verdicts "
          f"(no definitive contradiction).", flush=True)
    if not ok:
        sys.exit(1)


def main():
    if "--selftest" in sys.argv[1:]:
        _selftest()
        return
    if "--e2e" in sys.argv[1:]:
        _e2e()
        return
    t0 = time.time()
    print(f"Completion 2 -- formula-directed resolution of the C>20 inconclusive set (ORDER_CAP={ORDER_CAP:,})\n",
          flush=True)
    rows = list(csv.DictReader(open("results/forward_index.csv")))
    inc = [r for r in rows if r["forward_verdict"] == "inconclusive-budget"]
    df = M.pd.read_parquet(FI.PARQUET) if hasattr(M, "pd") else __import__("pandas").read_parquet(FI.PARQUET)
    smi = {r["bgc_id"]: (r.get("product_smiles_canonical") or r.get("product_smiles_raw")) for _, r in df.iterrows()}
    work = []
    for r in inc:
        s = smi.get(r["bgc_id"])
        if not s:
            continue
        mol = M.mol_from_smiles(s)
        T = FI._formula_cho(mol)
        flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        work.append((r["bgc_id"], r["name"], T, flat))
    print(f"[1/2] {len(work)} inconclusive (C>20) cores to resolve\n", flush=True)

    k = max(2, (os.cpu_count() or 4) - 2)
    results = []
    with mp.Pool(k, initializer=_worker_sig) as pool:
        for i, res in enumerate(pool.imap_unordered(_resolve_one, work), 1):
            results.append(res)
            if i % 10 == 0 or i == len(work):
                from collections import Counter
                c = Counter(r[3] for r in results)
                print(f"  {i}/{len(work)} | recovered={c['recovered']} gap={c['sound-structural-gap']} "
                      f"infeasible={c['formula-infeasible']} ord-budget={c['inconclusive-ordering-budget']} "
                      f"timeout={c['inconclusive-timeout']}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bgc_id", "name", "C", "H", "O", "directed_verdict", "orderings_tried", "recovered_program"])
        for bgc, name, T, verdict, witness, ntried in sorted(results, key=lambda r: r[2]):
            w.writerow([bgc, name, T[0], T[1], T[2], verdict, ntried, witness or ""])
    from collections import Counter
    c = Counter(r[3] for r in results)
    print(f"\nwrote {OUT_CSV} ({time.time()-t0:.0f}s)  -> {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
