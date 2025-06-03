from __future__ import annotations

import pytest

from deformation_models import SLS, AndradeModel, Burgers, MaxwellModel

# just testing instantiation here

_combos = [
    (SLS, (60, 1e4, 80)),
    (AndradeModel, (60, 1e5)),
    (MaxwellModel, (60, 1e5)),
    (Burgers, (60, 1e5, 80, 1e4)),
]


@pytest.mark.parametrize(("model_class", "model_args"), _combos)
def test_model_instantiation(model_class, model_args):
    model = model_class(*model_args)
    assert model.J_t(10) > 0
