"""Witness test for the lasiodiplodin demonstrator (the first genome -> position-sound decorated product).

Locks: (1) the executor-rendered resorcylic-macrolactone core + the curated, position-sound O-methylation
round-trips DEPOSITED lasiodiplodin (BGC0001245) by flat-canonical ~=; (2) the curated register pins
EXACTLY ONE of the core's two aromatic OHs (ambiguity is refused, not guessed); (3) the position-sound edit
produces exactly the o_methylation formula delta. Together: position-sound, not exact-by-budget."""
import pandas as pd
import pytest
from rdkit import Chem

from lpi.chem.program import Cycle, Program, Release, ReductionState as K
from lpi.data.mibig import PROCESSED
from lpi.executor import core as _core
from lpi.executor.tailoring_positional import LASIODIPLODIN_OME, _site_atom, apply_curated_edit

CORE = Program(starter="acetyl", cycles=(
    Cycle(reduction=K.KR), Cycle(reduction=K.ER), Cycle(reduction=K.ER), Cycle(reduction=K.ER),
    Cycle(reduction=K.KETO), Cycle(reduction=K.KETO), Cycle(reduction=K.KETO),
), release=Release.RESORCYLIC_MACROLACTONE)


def _flat(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, isomericSmiles=False)


def _deposited() -> Chem.Mol:
    df = pd.read_parquet(PROCESSED / "fungal_pks_pairs.parquet")
    smi = df[df.bgc_id == "BGC0001245"].iloc[0]["product_smiles_canonical"]
    return Chem.MolFromSmiles(smi)


def test_lasiodiplodin_roundtrips_deposited():
    core = _core.exec(CORE)
    product = apply_curated_edit(core, LASIODIPLODIN_OME)
    assert _flat(product) == _flat(_deposited())


def test_register_uniquely_pins_site():
    core = _core.exec(CORE)
    n_oh = len({m[0] for m in core.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1]-c"))})
    assert n_oh == 2  # the resorcinol has two free OHs in the core
    site = _site_atom(core, LASIODIPLODIN_OME.register_smarts)  # raises on 0 or >1
    assert core.GetAtomWithIdx(site).GetSymbol() == "O"


def test_formula_delta_cross_check():
    # apply_curated_edit verifies the applied delta == declared o_methylation delta; a bad edit raises
    core = _core.exec(CORE)
    apply_curated_edit(core, LASIODIPLODIN_OME)


def test_ambiguous_register_is_refused():
    # a register that matches both phenols (any aromatic OH) must be REFUSED, not silently guessed
    core = _core.exec(CORE)
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        _site_atom(core, "[OX2H1]-c")
