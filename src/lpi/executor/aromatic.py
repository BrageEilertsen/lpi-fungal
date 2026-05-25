"""NR-PKS product-template (PT) domain aromatic cyclization -- STUB (Phase 0b).

The large non-reducing PKS (norsolorinic acid / aflatoxin class) fold a long
poly-beta-keto chain into fused aromatics under a PT domain that selects among
*multiple competing aldol regiochemistries* (Crawford et al., 2009 describe the
regioselectivity classes). This is fundamentally different from the single,
unambiguous cyclization of small aromatic PKS (handled in :mod:`lpi.executor.cyclize`)
and is intentionally NOT part of the Phase 0 gate.

Phase 0b will implement the regioselectivity classes here. Until then, callers should
treat PT-dependent systems as out of scope (tier them in the curated set).
"""

from __future__ import annotations

from rdkit import Chem


class PTDomainNotImplemented(NotImplementedError):
    pass


# Regioselectivity classes to implement in Phase 0b (Crawford et al., 2009).
PT_REGIOSELECTIVITY_CLASSES = ("C2-C7", "C4-C9", "C6-C11", "C2-C7/C4-C9", "other")


def cyclize_pt(linear_polyketo: Chem.Mol, regio_class: str) -> Chem.Mol:  # noqa: ARG001
    raise PTDomainNotImplemented(
        "PT-domain multi-regiochemistry cyclization is deferred to Phase 0b; "
        f"requested class={regio_class!r}"
    )
