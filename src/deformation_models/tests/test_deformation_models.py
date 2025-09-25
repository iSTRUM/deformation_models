from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from deformation_models import (
    SLS,
    AndradeModel,
    Burgers,
    DeformationExperiment,
    MaxwellModel,
)

# just testing instantiation here

_combos = [
    (SLS, (60.0, 1e4, 80.0)),
    (AndradeModel, (60.0, 1e5)),
    (MaxwellModel, (60.0, 1e5)),
    (Burgers, (60.0, 1e5, 80.0, 1e4)),
]


@pytest.mark.parametrize(("model_class", "model_args"), _combos)
def test_model_instantiation(model_class, model_args):
    model = model_class(*model_args)
    assert model.J_t(10.0) > 0.0
    assert isinstance(model.J_t(100.0), float)
    assert isinstance(model.J_t(np.array([10.0, 100.0])), np.ndarray)


@pytest.mark.parametrize(("model_class", "model_args"), _combos)
def test_deformation_experiment(model_class, model_args):
    model = model_class(*model_args)

    def stress_func(t: npt.NDarray) -> npt.NDarray:
        return np.full(t.shape, 100)

    def_exp = DeformationExperiment(
        stress_func=stress_func,
        solid=model,
        t_max=10,
        dt_max=0.1,
        n_t=50,
        delayed_solve=False,
    )

    assert len(def_exp.t_actual) == def_exp.n_t
    assert len(def_exp.stress_vals) == def_exp.n_t
    assert len(def_exp.strain_vals) == def_exp.n_t

    assert len(np.unique(def_exp.stress_vals)) == 1
    assert np.all(np.isreal(def_exp.strain_vals))
    assert np.all(np.isreal(def_exp.stress_vals))
