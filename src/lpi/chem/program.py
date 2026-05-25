"""Discrete latent biosynthetic program representation.

Implements the program ``z = {(c_t, a_t)}_{t=1}^N`` of the paper (Eq. 1): an ordered
list of catalytic *elongation* cycles, each carrying an action over the reductive
cascade and tailoring chemistry, plus a starter unit and a terminal release.

Conventions
-----------
* A cycle here is an *elongation* cycle (a Claisen extension). The starter unit is
  loaded separately and is not a cycle. So an N-ketide product has ``N - 1`` cycles
  (e.g. the lovastatin nonaketide = 8 cycles + acetyl starter = 9 ketide units),
  matching "LovB runs eight cycles" in the paper.
* The reductive cascade is strictly ordered: ER presupposes DH presupposes KR
  (Section 3). ``ReductionState`` encodes the four reachable states.
* C-MeT is a genuine SAM-dependent alpha-methylation. It is NOT used to install
  terminal backbone methyls that originate from the starter acetate (e.g. the ring
  methyl of 6-MSA / orsellinic acid) -- those programs carry no C-MeT action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReductionState(Enum):
    """The four reachable beta-position oxidation states (ordered cascade)."""

    KETO = "keto"  # beta-ketothioester, no reductive processing
    KR = "kr"  # beta-hydroxyl (ketoreduction)
    DH = "dh"  # enoyl, alpha,beta-unsaturated (KR + dehydratase)
    ER = "er"  # methylene, fully reduced (KR + DH + enoylreductase)

    @property
    def rank(self) -> int:
        """Position in the reductive cascade (0 = keto ... 3 = methylene)."""
        return {"keto": 0, "kr": 1, "dh": 2, "er": 3}[self.value]


class Extender(Enum):
    """Extender unit loaded at the AT for chain elongation."""

    MALONYL = "malonyl"  # +C2 ketide, universal fungal extender
    METHYLMALONYL = "methylmalonyl"  # +C2 ketide w/ alpha-methyl; bacterial AT feature


class Release(Enum):
    """Terminal chain-release / cyclization mode applied to the tethered chain."""

    HYDROLYSIS = "hydrolysis"  # thioesterase hydrolysis -> carboxylic acid (linear)
    LACTONIZATION = "lactonization"  # O->C(=O) lactone (e.g. triacetic acid lactone)
    ALDOL_AROMATIC = "aldol_aromatic"  # single-mode aldol/Claisen + aromatization
    DIHYDROISOCOUMARIN = "dihydroisocoumarin"  # composite: lactone + aromatic aldol (mellein)
    NONE = "none"  # leave tethered (debug / inspection only)


@dataclass(frozen=True)
class Cycle:
    """One elongation cycle: extension + optional tailoring on the new beta-keto."""

    reduction: ReductionState = ReductionState.KETO
    c_methyl: bool = False  # C-MeT alpha-methylation this cycle
    extender: Extender = Extender.MALONYL


@dataclass(frozen=True)
class Program:
    """A complete biosynthetic program: starter + ordered cycles + release."""

    starter: str = "acetyl"
    cycles: tuple[Cycle, ...] = field(default_factory=tuple)
    release: Release = Release.HYDROLYSIS

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycles", tuple(self.cycles))

    @property
    def n_cycles(self) -> int:
        """Number of elongation cycles."""
        return len(self.cycles)

    @property
    def n_ketide_units(self) -> int:
        """Total ketide units = starter (1) + one per elongation cycle."""
        return 1 + len(self.cycles)
