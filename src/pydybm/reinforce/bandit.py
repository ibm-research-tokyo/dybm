import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist):

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        self._seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        done = False

        if np.random.randn(1) > self.p_dist[action]:

            reward = 1  # self.r_dist[0]

        else:
            reward = -1  # self.r_dist[1]

        return 0.0, reward, done, {}

    def reset(self):
        return 0

    def render(self, mode='human', close=False):
        pass


class FourArmedBandit(BanditEnv):
    """Stochastic version of four-armed bandit where bandit four pays out with highest reward"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.2, 0.0, -0.2, -5], r_dist=[1, -1])
