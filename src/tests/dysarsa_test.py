# (C) Copyright IBM Corp. 2017
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from pydybm.reinforce.dysarsa import DYSARSA
from pydybm.reinforce.discrete_agent import DySARSAAgent
from pydybm.base.sgd import RMSProp, ADAM
from tests.bandit import FourArmedBandit
import pydybm.arraymath as amath
import unittest
from tests.arraymath import NumpyTestMixin, CupyTestMixin


"""
__Author__: Sakyasingha Dasgupta
"""

class DySARSATestCase(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test(self):
        env = FourArmedBandit()
        observation = env.reset()
        action_dim = env.action_space.n
        state_dim = env.observation_space.n

        print('\n\nUnit testing DySARSA RL Agent on four armed bandit problem\n')
        print('action dim is :', action_dim)
        print('state_dim is :', state_dim)

        print('Number of bandits: ', env.n_bandits)

        """
        DySARSA model parameters

        """
        delay = 2
        decay = [0.2]
        discount = 0.99
        temperature = 0.1

        SGD = ADAM()
        learning_rate = 0.01

        init_epsilon = 0.3  # initial exploration term (epsilon-greedy)
        final_epsilon = 0.1  # final exploration term (when using annealing)

        steps_per_episode = 10
        train_steps = 1000
        num_bandits = env.n_bandits
        total_reward = amath.zeros(num_bandits)

        amath.random.seed(7)

        """Create DySARSA function approximator model"""
        DySARSA_model = DYSARSA(state_dim, action_dim, delay, decay, discount, SGD,
                                learning_rate, temperature, insert_to_etrace="w_delay", L1=0.00)


        print('\nTraining new agent with DySARSA RL agent and Boltzmann policy')

        agent = DySARSAAgent(env=env, model=DySARSA_model, steps_per_episode=steps_per_episode, train_steps=train_steps,
                             exploration="Boltzmann", init_epsilon=temperature, final_epsilon=final_epsilon, UseQLearning=False,
                             frame_skip=False, suppress_print=True)

        agent.fit(test_every=2, test_num_eps=5, break_reward=100, render=False)

        print('\nTesting DySARSA RL agent with Boltzmann Policy')

        agent = DySARSAAgent(env=env, model=DySARSA_model, suppress_print=True)

        for i in range(5):
            reward = agent.predict(render=True)
            total_reward[agent.action] += reward

        print("\nAverage reward predicted for each arm is: ", total_reward/5)

        print("The agent thinks bandit " + str(agent.action+1) + " is the most rewarding....")
        if agent.action == num_bandits-1:
            print("and the prediction is correct!")
        else:
            print("and the prediction is incorrect!")

        print("\n************************************")

        self.assertEqual(agent.action, num_bandits-1)

        total_reward = amath.zeros(num_bandits)

        print('\nTraining new agent with DySARSA RL agent and epsilon-greedy policy')

        agent = DySARSAAgent(env=env, model=DySARSA_model, steps_per_episode=steps_per_episode, train_steps=train_steps,
                             exploration="greedy", init_epsilon=init_epsilon, final_epsilon=final_epsilon, UseQLearning=False,
                             frame_skip=False, suppress_print=True)

        agent.fit(test_every=2, test_num_eps=5, break_reward=100, render=False)

        print('\nTesting DySARSA RL agent with epsilon-greedy Policy')

        agent = DySARSAAgent(env=env, model=DySARSA_model, suppress_print=True)

        for i in range(5):
            reward = agent.predict(render=True)
            total_reward[agent.action] += reward

        print("\nAverage reward predicted for each arm is: ", total_reward/5)

        print("The agent thinks bandit " + str(agent.action+1) + " is the most rewarding....")
        if agent.action == num_bandits-1:
            print("and the prediction is correct!")
        else:
            print("and the prediction is incorrect!")

        self.assertEqual(agent.action, num_bandits-1)


class DyBMTestCaseNumpy(NumpyTestMixin, DySARSATestCase, unittest.TestCase):
    pass


# TODO: Not working with cupy now
# class DyBMTestCaseCupy(CupyTestMixin, DySARSATestCase, unittest.TestCase):
#     pass


if __name__ == "__main__":

    unittest.main()
                

