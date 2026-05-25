"""ESM-2 head logic tests (no network / no model download): metrics + page parsing."""

from __future__ import annotations

import numpy as np

from lpi.model.esm_head import _balanced_acc, _roc_auc


def test_roc_auc_known():
    # perfect separation -> 1.0; reversed -> 0.0; tie -> 0.5
    assert _roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert _roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert abs(_roc_auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9


def test_balanced_accuracy():
    y = np.array([0, 0, 0, 1])
    # predict all-0: class0 acc 1.0, class1 acc 0.0 -> balanced 0.5 (not the 0.75 raw acc)
    assert abs(_balanced_acc(y, np.zeros(4, dtype=int)) - 0.5) < 1e-9


def test_module_id_parse():
    from lpi.model.esm_data import _modules_with_ids

    page = ('smiles="CC(=O)[S]" data-domainid="1" data-domain="KS" '
            'data-domainid="2" data-domain="AT" data-domainid="3" data-domain="ACP" '
            'smiles="CC(=O)CC(=O)[S]" data-domainid="4" data-domain="KS" '
            'data-domainid="5" data-domain="KR" smiles="CC(O)CC(=O)[S]"')
    mods = list(_modules_with_ids(page))
    # last module carries a KR domain id
    assert any("KR" in dom_ids for dom_ids, *_ in mods)
