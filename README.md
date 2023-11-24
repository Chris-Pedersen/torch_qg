# torch_qg
Modelling of a 2 layer [quasi-geostrophic](https://en.wikipedia.org/wiki/Quasi-geostrophic_equations) system in PyTorch. We use [PyTorch 2.0.0](https://pytorch.org/get-started/pytorch-2.0/). This repo is based on [pyqg](https://pyqg.readthedocs.io/en/latest/), with a few changes. We implement all components of the numerical scheme in PyTorch, such that the simulation is end-to-end differentiable. Additionally, whilst pyqg uses a pseudo-spectral method to evolve the system forward in time, we include a real-space time-stepper, with an [Arakawa advection scheme](https://www.sciencedirect.com/science/article/pii/0021999166900155).

Core dependencies can be found in `tests/requirements.txt`. After git cloning the repo, install by running

`pip install .`

or to run in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), run

`pip install -e .`
