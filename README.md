# deformation_models

The `deformation_models` package is a tool for learning about phenomenological
models of transient deformation (so called spring-dashpot models). It is meant
to be used alongside a forthcoming Jupyter book tutorial, but at present there
are a couple of notebooks in this repository (see below).

## installation

To install from the `pypi` repository with dependencies required for the
examples:

```shell
pip install "deformation-models[examples]"
```

## usage

A number of deformation models are available at present:

```python
from deformation_models import SLS, AndradeModel, Burgers, MaxwellModel
```

Each of the models represents a phenomenological model (SLS above is standard
linear solid). To use them, you generally supply values for the unrelaxed
compliance and maxwell time scales for the various components making up each
model, check the help for each for what is required.

## examples

In addition to the forthcoming jupyter book, this repository has a few examples:

- [Calculating stress-strain curves for a Standard Linear Solid](https://github.com/iSTRUM/deformation_models/blob/main/docs/examples/ex_01_stress_strain_curves.ipynb)
- [Calculating attenuation with `deformation_models` and `pyleoclim`](https://github.com/iSTRUM/deformation_models/blob/main/docs/examples/ex_02_attenuation_calculation.ipynb)

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/iSTRUM/deformation_models/workflows/CI/badge.svg
[actions-link]:             https://github.com/iSTRUM/deformation_models/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/deformation_models
[conda-link]:               https://github.com/conda-forge/deformation_models-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/iSTRUM/deformation_models/discussions
[pypi-link]:                https://pypi.org/project/deformation_models/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/deformation_models
[pypi-version]:             https://img.shields.io/pypi/v/deformation_models
[rtd-badge]:                https://readthedocs.org/projects/deformation_models/badge/?version=latest
[rtd-link]:                 https://deformation_models.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
