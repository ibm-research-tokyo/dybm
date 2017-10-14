.. pydybm documentation master file, created by
   sphinx-quickstart on Sat Oct  7 03:47:16 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pydybm is a Python implementation for learning time-series with DyBMs.

Packages and Modules
==================================

.. toctree::
   :maxdepth: 5

   pydybm




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`







Getting started with pydybm
---------------------------

Prerequisites for pydybm
~~~~~~~~~~~~~~~~~~~~~~~~

The modules in the following assume latest versions (as of July, 2017).
They might work in other versions, unless required versions are
explicitely stated. To update a module into the latest version, run
``pip install [module] --upgrade``.

**Essentials**

pydybm runs either on **Python 2.x** or on **Python 3.x** (it has been
most heavily tested with **Python 2.7** and **Python 3.5** on Ubuntu and
Mac). pydybm relies heavily on **numpy**, **six**, **scipy**,
**sklearn**, and **gym**. These are the minimal prerequisites for
pydybm.

``pip install numpy six scipy sklearn gym``

To get benefit of GPUs, **cupy** and **chainer** is also required. Note
that **CUDA 8.0** is needed for **pydybm**, while **cupy** can be
installed with **CUDA 7.5**.

``pip install cupy chainer``

Install and test pydybm
~~~~~~~~~~~~~~~~~~~~~~~

The following commands should be executed at the DyBM root directory
(this Readme is at [Root]/src/pydybm/).

| Install pydybm by:
| ``python setup.py develop``

| Run unit test by:
| ``python setup.py test``

If the test is successful, it completes with a message like this:

::

    ----------------------------------------------------------------------
    Ran 108 tests in 89.631s

    OK

DyBMs that can be readily used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pydybm provides the following classes of DyBMs that can be readily used
for general purposes of time-series learning:

-  ``pydybm.time_series.dybm.LinearDyBM``
-  ``pydybm.time_series.rnn_gaussian_dybm.RNNGaussianDyBM``
-  ``pydybm.time_series.functional_dybm.FunctionalDyBM``

Choose FunctionalDyBM if a one-step pattern in your time-series is in a
feature space. Choose LinearDyBM or RNNGaussianDyBM otherwise (or if you
do not understand what it means for a one-step pattern to be in a
feature space).

How to train a DyBM and give prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we provide a minimal example of how to train a DyBM and how to give
prediction with that DyBM. We train the simplest DyBM,
``pydybm.time_series.dybm.LinearDyBM``, with a one-dimensional
time-series of noisy sine wave. As we train the DyBM in an online
manner, we let the DyBM predict the next value of the time-series. In
the end, we plot the time-series used for training and the corresponding
predictions.

::

    import matplotlib.pyplot as plt
    from pydybm.time_series.dybm import LinearDyBM
    from pydybm.base.generator import NoisySin

    # Prepare a generator of time-series
    # In this example, we generate a noisy sine wave
    length = 300 # length of the time-series
    period = 60  # period of the sine wave
    std = 0.1   # standard deviation of the noise
    dim = 1      # dimension of the time-series
    data = NoisySin(length,period,std,dim)

    # Create a DyBM
    # In this example, we use the simplest Linear DyBM
    dybm = LinearDyBM(dim)

    # Learn and predict the time-series in an online manner
    result = dybm.learn(data)

    # Plot the time-series and prediction
    plt.plot(result["actual"],label="target")
    plt.plot(result["prediction"],label="prediction")
    plt.legend()

Other DyBMs can be used in an analogous manner, but pydybm allows more
sophisticated use of DyBMs with more complex time-series data. See
examples by running ``jupyter notebook`` at ``../../examples/``.

pydybm for reinforcement learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pydybm.reinforce.dysarsa`` is the base class for using a DyBM network
factored into observations nodes and action nodes, in order to perform
SARSA temporal difference reinforcement learning. This uses the linear
energy of a binary DyBM as the Q-action-value function for SARSA update.
The parameters of DySARSA network is updated using a temporal difference
error.

See further details in: S. Dasgupta & T. Osogami, Spike Timing Dependent
Reinforcement learning with Dynamic Boltzmann Machine (2016).

How to create your own DyBMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can create your own DyBMs by the use of the building
blocks provided in ``pydybm.time_series.dybm``. An example of such
created DyBM is ``pydybm.time_series.dybm.GaussianBernoulliDyBM``.

Other how-to on pydybm
----------------------


How to generate documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate documentation, you need **Sphinx** (version 1.4 or above),
**numpydoc**, **jupyter**, and **pandoc**. Generating the mathematical symbols for included formulas requires **LaTeX**  and **dvipng** (and possibly the package **texlive-latex-extra** as well).

| Generate html documentation with the command (inside the ``src/pydybm/docs/`` directory):
| 	``make html``
| You will find html documentation under ``docs/_build/html``

How to uninstall
~~~~~~~~~~~~~~~~

The following command should be executed inside the DyBM Root directory (this
Readme is at ``[Root]/src/pydybm/``).

| Uninstall pydybm by:
| ``python setup.py develop --uninstall``
