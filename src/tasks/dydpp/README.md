(C) Copyright IBM Corp. 2019

# Dynamic determinantal point processes

DyDPP.py contains core methods of learning and inference with dynamic determinantal point processes.  This directory also contains scripts that can be used to reproduce the experimental results and figures reported in the following paper:

T. Osogami, R. Raymond, A. Goel, T. Shirai, T. Maehara, "Dynamic Determinantal Point Processes," AAAI-18

# Preparation

Place the pickle file of data as follows:
```
data/XXXX.pickle
```
We assume that the pickle file is formatted in the same way as the datasets published at http://www-etud.iro.umontreal.ca/~boulanni/icml2012.  Note that these datasets are provided by a third party under the terms and conditions specified by the third party.

The experiments run with Python 3.  Install the dependencies by
```
pip install requirements.txt
```

# Run experiments

```
python experiment.py data/XXXX.pickle
```

# Plot figures

```
python plot.py XXX
```
where "XXX" is the first three letters of the pickle file (XXXX.pickle).

Note that the generated figures are similar but not completely identical to those in the paper.  We have modified the code to run with Python 3, while the original experiments were run with Python 2.
