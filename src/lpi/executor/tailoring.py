"""Tailoring-edit vocabulary: deterministic, parameterised post-PKS graph edits.

Each edit is fully deterministic once its type + position parameters are fixed (soundness
preserved); the *search* is over which edits and where. Each edit type is tied to an
enzyme family (the gene budget gates which types may fire) and carries an exact molecular-
formula delta (the Delta-formula budget pins the edit-type multiset).

Formula deltas are net atoms added to the whole molecule, as a Counter over element
symbols (negative = removed). Example: O-methylation OH -> OCH3 is +C +2H = {C:1, H:2}.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EditType:
    name: str
    family: str  # enzyme family (must be in the cluster's gene budget to fire)
    delta: dict[str, int] = field(default_factory=dict)  # net formula change
    note: str = ""


# ~12 edit types spanning the common fungal post-PKS tailoring chemistry.
EDIT_TYPES: tuple[EditType, ...] = (
    EditType("hydroxylation", "oxygenase", {"O": 1}, "C-H -> C-OH"),
    EditType("epoxidation", "oxygenase", {"O": 1}, "C=C -> epoxide"),
    EditType("ketone_oxidation", "oxygenase", {"O": 1, "H": -2}, "CH2 -> C=O"),
    EditType("o_methylation", "methyltransferase", {"C": 1, "H": 2}, "O-H -> O-CH3"),
    EditType("c_methylation", "methyltransferase", {"C": 1, "H": 2}, "C-H -> C-CH3"),
    EditType("halogenation_cl", "halogenase", {"Cl": 1, "H": -1}, "C-H -> C-Cl"),
    EditType("halogenation_br", "halogenase", {"Br": 1, "H": -1}, "C-H -> C-Br"),
    EditType("prenylation", "prenyltransferase", {"C": 5, "H": 8}, "C-H -> C-C5H8 (DMA)"),
    EditType("glycosylation_hex", "glycosyltransferase", {"C": 6, "H": 10, "O": 5},
             "O-H -> O-hexosyl"),
    EditType("oxidative_cyclization", "oxygenase", {"H": -2},
             "new C-C / C-O bond between existing atoms (ring closure)"),
    EditType("oxidative_decarboxylation", "decarboxylase", {"C": -1, "O": -2},
             "COOH -> H (loss of CO2)"),
    EditType("reduction", "oxidoreductase", {"H": 2}, "C=O -> C-OH"),
    EditType("desaturation", "oxidoreductase", {"H": -2}, "CH-CH -> C=C"),
)

BY_NAME: dict[str, EditType] = {e.name: e for e in EDIT_TYPES}
BY_FAMILY: dict[str, list[EditType]] = {}
for _e in EDIT_TYPES:
    BY_FAMILY.setdefault(_e.family, []).append(_e)


def families_present(gene_budget: dict[str, int]) -> set[str]:
    return {fam for fam, n in gene_budget.items() if n > 0}


def admissible_edits(gene_budget: dict[str, int]) -> list[EditType]:
    """Edit types whose enzyme family is present in the cluster's gene budget."""
    present = families_present(gene_budget)
    return [e for e in EDIT_TYPES if e.family in present]
