(C) Copyright IBM Corp. 2016

This library contains multiple implementations of dynamic Boltzmann machines (DyBMs) and relevant tools.  The core of this library is __pydybm__, a Python implementation for learning time-series with DyBMs (see [src/pydybm/Readme.md](src/pydybm/Readme.md)), and __jdybm__, a Java implementation used in the first publication of the DyBM in  [www.nature.com/articles/srep14149](http://www.nature.com/articles/srep14149) (see [src/jdybm/Readme.md](src/jdybm/Readme.md)).

## What is DyBM?

The DyBM is an IBM’s artificial neural network, proposed in [www.nature.com/articles/srep14149](http://www.nature.com/articles/srep14149), that is trained via biologically plausible spike-timing dependent plasticity (STDP) in an online and distributed manner for prediction, anomaly detection, classification, reinforcement learning, and other tasks with time-series.  DyBM’s learning time per step is independent of the length of (the dependency in) the time-series under consideration (i.e., local in time), whereas existing recurrent neural networks including long short term memory (LSTM) perform, at each step, backpropagation through time whose computational complexity grows linearly with respect to that length.  DyBM’s computation for learning, prediction, sampling, and other operations can all be performed in a distributed manner, and its computational complexity of the operation at each unit is independent of the size of the network (i.e., local in space). 

DyBM stands for Dynamic Boltzmann Machine.  It is abbreviated as DyBM instead of DBM, because DBM is reserved for Deep Boltzmann Machine in the community.

## Directory structure

Here we provide descriptions of some of the important directories in this library.

- `src/`: You find source codes here.
 - `src/pydybm/`: You find __pydybm__ here.  See [src/pydybm/Readme.md](src/pydybm/Readme.md).
 - `src/jdybm/`: You find a Java implementation, which is used for the experiments in [www.nature.com/articles/srep14149](http://www.nature.com/articles/srep14149).  See [Readme.md for jdybm](src/jdybm/Readme.md).
- `examples/`: You find examples of using __pydybm__ here.  Run `jupyter notebook` at this directory to see the examples.  See also [src/pydybm/Readme.md](src/pydybm/Readme.md).
- `data/`: You will store datasets here.

