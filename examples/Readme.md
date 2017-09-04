# Examples

Here we provide examples of time-series learning and reinfrocement learning with DyBMs.

## Prerequisites

Some of the esamples rely on `__gym__` and `__matplotlib__`.

```
pip install gym gym[atari] matplotlib
```

## Time-series learning

Examples are provided under `time-series`

## Reinforcement learning

Examples are provided under `reinforce`

`DySARSA_discreteAgent_Demo.py` demonstrates an example for using a model based on `pyDyBM.reinforce.DySARSA` for learning to play Atari games directly from screen pixels, using the Arcade Learning Environment. This example uses the atari games environment as provided by OpenAI Gym (https://gym.openai.com/).

