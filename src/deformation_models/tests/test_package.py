from __future__ import annotations

import importlib.metadata

import deformation_models as m


def test_version():
    assert importlib.metadata.version("deformation_models") == m.__version__
