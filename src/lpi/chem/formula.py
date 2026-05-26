"""Monoisotopic masses and bounded molecular-formula enumeration within a ppm window.

Supports the tolerant mass observable (Track 2): given an observed neutral monoisotopic mass and a
ppm tolerance, enumerate the integer C/H/(N/)O formulas whose mass falls in the window. A grammar
then runs its *sound* exact-formula enumeration once per candidate formula, so the ppm-tolerant
candidate set Z*_eps is built without an exact-equality shortcut and without the combinatorial
blow-up of enumerating an entire highly-reducing program space (the mass prunes before execution).
"""
from __future__ import annotations

# Monoisotopic atomic masses (u). Electron mass is handled in the adduct shifts (lpi.observe).
MONO: dict[str, float] = {
    "C": 12.0,
    "H": 1.0078250319,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "S": 31.9720707,
    "Cl": 34.9688527,
}


def formula_mass(counts: dict[str, int]) -> float:
    """Monoisotopic mass of an element-count mapping, e.g. ``{'C':8,'H':8,'O':4}``."""
    return sum(MONO[e] * n for e, n in counts.items())


def enumerate_chno(neutral_mass: float, ppm: float, with_n: bool = False,
                   max_c: int = 40) -> list[tuple[int, ...]]:
    """All integer C/H/O (or C/H/N/O) formulas with monoisotopic mass within ``ppm`` of
    ``neutral_mass``, subject to a valence ceiling ``H <= 2C + 2 + N`` (DBE >= 0).

    Returns ``(C, H, O)`` tuples when ``with_n`` is false, else ``(C, H, N, O)`` -- matching the
    target-formula key arity each grammar's enumerator expects.
    """
    if neutral_mass <= 0:
        return []
    lo = neutral_mass * (1.0 - ppm * 1e-6)
    hi = neutral_mass * (1.0 + ppm * 1e-6)
    C, H, O, N = MONO["C"], MONO["H"], MONO["O"], MONO["N"]
    out: list[tuple[int, ...]] = []
    for c in range(1, max_c + 1):
        if C * c > hi:
            break
        for o in range(0, c + 3):
            base_co = C * c + O * o
            if base_co > hi:
                break
            for n in (range(0, c + 1) if with_n else range(0, 1)):
                base = base_co + N * n
                if base > hi:
                    break
                # solve the H window directly rather than scanning all H
                h_min = max(0, int((lo - base) / H))
                h_max = int((hi - base) / H) + 1
                ceil = 2 * c + 2 + n
                for h in range(h_min, h_max + 1):
                    if h > ceil:
                        continue
                    m = base + H * h
                    if lo <= m <= hi:
                        out.append((c, h, n, o) if with_n else (c, h, o))
    return out
