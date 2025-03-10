"""
Copyright (c) 2024 Chris Havlin. All rights reserved.

deformation_models: Phenomenological deformation models
"""

from __future__ import annotations

from ._version import version as __version__
from .main import SLS, AndradeModel, Burgers, MaxwellModel

__all__ = ["__version__", "MaxwellModel", "AndradeModel", "SLS", "Burgers"]
