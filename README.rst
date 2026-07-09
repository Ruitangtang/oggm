=============================================================================
SERMeQ_RT — Frontal Ablation Extension (SERMeQ) for OGGM
=============================================================================

.. image:: https://img.shields.io/badge/📄-GMD_Preprint_2026-blue
   :target: https://doi.org/10.5194/egusphere-2026-1081
   :alt: Paper

.. image:: https://img.shields.io/badge/DOI-10.5281/zenodo.18761729-blue.svg
   :target: https://doi.org/10.5281/zenodo.18761729
   :alt: Zenodo DOI

.. image:: https://img.shields.io/github/v/release/Ruitangtang/oggm?label=stable&color=blue
   :target: https://github.com/Ruitangtang/oggm/releases/tag/v1.0.0-zenodo-frontal-ablation
   :alt: Stable Release

This branch contains the OGGM core modifications used for the frontal ablation study:

*"Joint Bayesian Calibration of Frontal Ablation and Surface Mass Balance in Global Glacier Models"* (GMD Preprint, 2026)

**Key modifications:**

- Frontal ablation module (SERMeQ) added to ice dynamics
- Frontal ablation, terminus change hooks coupled with climatic mass balance, ice thickness, width, and ice velocity

**Full research archive:** `Zenodo concept DOI 10.5281/zenodo.18761729 <https://doi.org/10.5281/zenodo.18761729>`_

**Stable release:** `v1.0.0-zenodo-frontal-ablation <https://github.com/Ruitangtang/oggm/releases/tag/v1.0.0-zenodo-frontal-ablation>`_

🚀 Lightweight showcase repo: `frontal-ablation-glacier-demo <https://github.com/Ruitangtang/frontal-ablation-glacier-demo>`_ — YAML configs, dry-run quickstart commands, tests, method/results visuals, and reproducibility links.

**Citation:**
> Yang et al.  *Joint Bayesian Calibration of Frontal Ablation and Surface Mass Balance in Global Glacier Models*. GMD Preprint, 2026. DOI: `10.5194/egusphere-2026-1081 <https://doi.org/10.5194/egusphere-2026-1081>`_


-------------------------------------------------------------------------------------------------------------------------------------------

.. image:: docs/_static/logo.png

|


**OGGM is a modular open source model for glacier dynamics**

OGGM is able to simulate past and
future mass balance, volume and geometry of (almost) any glacier in the world,
in a fully automated and extensible workflow.

The model accounts for glacier geometry (including contributory branches) and
includes an explicit ice dynamics module. We rely exclusively on publicly
available data for calibration and validation. **OGGM is modular and
supports novel modelling workflows**: it LOVES to be remixed and reused!

.. image:: docs/_static/ex_tasman.jpg


Installation, documentation
---------------------------

The documentation is hosted on ReadTheDocs: http://docs.oggm.org


Get in touch
------------

- View the source code `on GitHub`_.
- Report bugs or share your ideas on the `issue tracker`_.
- Improve the model by submitting a `pull request`_.
- Follow us on `Twitter`_.
- Or you can always send us an `e-mail`_ the good old way.

.. _e-mail: info@oggm.org
.. _on GitHub: https://github.com/OGGM/oggm
.. _issue tracker: https://github.com/OGGM/oggm/issues
.. _pull request: https://github.com/OGGM/oggm/pulls
.. _Twitter: https://twitter.com/OGGM1


About
-----

:Version:
    .. image:: https://img.shields.io/pypi/v/oggm.svg
        :target: https://pypi.python.org/pypi/oggm
        :alt: Pypi version
        
    .. image:: https://img.shields.io/pypi/pyversions/oggm.svg
        :target: https://pypi.python.org/pypi/oggm
        :alt: Supported python versions

:Citation:
    .. image:: https://img.shields.io/badge/Citation-GMD%20paper-orange.svg
        :target: https://www.geosci-model-dev.net/12/909/2019/
        :alt: GMD Paper

    .. image:: https://zenodo.org/badge/43965645.svg
        :target: https://zenodo.org/badge/latestdoi/43965645
        :alt: Zenodo

:Tests:       
    .. image:: https://coveralls.io/repos/github/OGGM/oggm/badge.svg?branch=master
        :target: https://coveralls.io/github/OGGM/oggm?branch=master
        :alt: Code coverage

    .. image:: https://github.com/OGGM/oggm/actions/workflows/run-tests.yml/badge.svg?branch=master
        :target: https://github.com/OGGM/oggm/actions/workflows/run-tests.yml
        :alt: Linux build status

    .. image:: https://img.shields.io/badge/Cross-validation-blue.svg
        :target: https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/crossval.html
        :alt: Mass balance cross validation

    .. image:: https://readthedocs.org/projects/oggm/badge/?version=latest
        :target: http://docs.oggm.org/en/latest/
        :alt: Documentation status

    .. image:: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
        :target: https://cluster.klima.uni-bremen.de/~github/asv/
        :alt: Benchmark status

:License:
    .. image:: https://img.shields.io/pypi/l/oggm.svg
        :target: https://github.com/OGGM/oggm/blob/master/LICENSE.txt
        :alt: BSD-3-Clause License

:Authors:

    See the `version history`_ for a list of all contributors.

    .. _version history: http://docs.oggm.org/en/stable/whats-new.html
