# (C) Copyright IBM Corp. 2016
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

from sklearn.preprocessing import MinMaxScaler
from .. import arraymath as amath
from .agent import Agent
import numpy as np


"""
Using DySARSA Agent for reinforcement learning in a given environment

In this example we use the OpenAI Gym environment but the user an suitably modify the environment to any valid MDP

OpenAI Gym can be installed from: https://gym.openai.com/

In case of Arcade Learning enviornment:
seealso: Bellemare, Marc G., et al. "The arcade learning environment: An evaluation platform for general agents." JAIR(2015).
"""


__author__ = "Sakyasingha Dasgupta"


class DySARSAAgent(Agent):

    def __init__(self, env, model, steps_per_episode=300, train_steps=50000,
                 exploration="Boltzmann", init_epsilon=1.0, final_epsilon=0.01,
                 UseQLearning=False, frame_skip=False, threshold=0, suppress_print=False):

        self.env = env
        self.time_steps = 0
        self.episode_number = 0
        self.steps_per_episode = steps_per_episode
        self.exploration = exploration
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = init_epsilon
        self.train_steps = train_steps
        self.UseQLearning = UseQLearning
        self.frame_skip = frame_skip
        self.suppress_print = suppress_print

        """Initialize DySARSA RL function approximator model"""
        self.model = model
        self.threshold = threshold

        self.action = None
        self.action_dim = self.model.out_dim
        self.state_dim = self.model.in_dim - self.model.out_dim

    def prepro(self, Image):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        Image = Image[35:195]  # crop
        Image = Image[::2, ::2, 0]  # downsample by factor of 2

        Image = amath.asarray(Image, dtype='float64')

        # normalize data to [0,1]
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        dataset = scaler.fit_transform(Image.reshape(-1, 1))

        return dataset

    def normalize(self, Image):

        Image = amath.asarray(Image, dtype='float64')
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        dataset = scaler.fit_transform(Image.reshape(-1, 1))

        return dataset

    def boltzmann_action(self, state, annealing=True):

        if annealing:
            self.epsilon -= (self.init_epsilon - self.final_epsilon)/self.train_steps
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon
        else:
            self.epsilon = self.init_epsilon

        """ Calculate probability of actions """
        aprob = self.model.prob_action(state, self.epsilon)

        """ Purely Boltzmann exploration policy"""
        action = amath.argmax(aprob[self.state_dim:])

        if np.asscalar(aprob[self.state_dim+action]) > 0.95:
            # print "action is:", action
            return action
        else:
            action = amath.random.randint(self.action_dim)
            # print "random action is:", action
            return action

    def egreedy_action(self, state, annealing=True):

        """ Calculate probability of actions """
        aprob = self.model.Q_next(state)

        if annealing:
            self.epsilon -= (self.init_epsilon - self.final_epsilon)/self.train_steps
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon

        if amath.random.random() <= self.epsilon:
            return amath.random.randint(self.action_dim)  # roll the dice!

        else:
            return amath.argmax(aprob[self.state_dim:])

    def perceive(self, state, reward, next_state):

        """ Calculate the Q action-value function with previous observations
                        and actions and new observations and actions
                        """
        prevQ = self.model.Q_next(state)

        futureQ = self.model.Q_next(next_state)

        """Calculate error"""
        if self.UseQLearning:
            error = self.model.Q_error(reward, futureQ, prevQ)
        else:
            error = self.model.TD_error(reward, futureQ, prevQ)
        if self.episode_number >= self.threshold:
            self.model.learn_one_step(next_state, error)

        self.time_steps += 1

    def _fit_episode(self):
        state = self.env.reset()
        state = self.normalize(state)
        total_reward = 0
        self.action = None

        for steps in range(self.steps_per_episode):

            if self.action is None:
                state.resize((self.state_dim + self.action_dim, 1))
                state[self.state_dim:].fill(amath.random.uniform(0, 0.01))

            if self.exploration == "Boltzmann":
                self.action = self.boltzmann_action(state, annealing=False)
            else:
                self.action = self.egreedy_action(state)

            if self.frame_skip:
                for i in range(np.random.randint(2, 5)):
                    # step the environment and get new measurements
                    next_state, reward, done, _ = self.env.step(self.action)
                    total_reward += reward
                    if done:
                        break
            else:
                next_state, reward, done, _ = self.env.step(self.action)
            next_state = self.normalize(next_state)
            next_state.resize((self.state_dim + self.action_dim, 1))
            next_state[self.state_dim:].fill(amath.random.uniform(0, 0.01))
            next_state[self.state_dim + self.action] = 0.99
            self.perceive(state, reward, next_state)

            state = next_state
            if done:
                break
        if not self.suppress_print:
            print('Current epsilon: ', self.epsilon)

        return total_reward

    def _fit(self, test_every=None, test_num_eps=10, break_reward=500, render=False):

        ave_reward = 0.0
        Curr_reward_total = 0.0

        while self.time_steps < self.train_steps:
            if not self.suppress_print:
                print("current time step is:", self.time_steps)
            # initialize task
                print('Episode : ', self.episode_number)
            train_reward = self.fit_episode()

            # Test every test_every episodes
            if test_every:
                total_reward = 0

                if (self.episode_number + 1) % test_every == 0:
                    for i in range(test_num_eps):
                        reward_i = self.predict(render=render)
                        total_reward += reward_i
                        if reward_i > ave_reward:
                            ave_reward = reward_i
                    if not self.suppress_print:
                        print('episode: ', self.episode_number)
                        print('Evaluation Max Reward:', ave_reward)

                """
                The value of the break_reward needs to be according to the task. It can also
                be removed. In that case algorithm needs a different stopping condition.
                """
                if ave_reward >= break_reward:
                    break

            Curr_reward_total += ave_reward
            running_avg = Curr_reward_total / (self.episode_number + 1)
            if not self.suppress_print:
                print('Running average reward: ', running_avg)
            self.episode_number += 1

    """
     Decision making - generating new action sequences based on learning
     """

    def _predict(self, render=True):
        total_reward = 0
        state = self.env.reset()
        state = self.normalize(state)
        self.action = None

        for j in range(self.steps_per_episode):

            if self.action is None:
                state.resize((self.state_dim + self.action_dim, 1))
                state[self.state_dim:].fill(amath.random.uniform(0, 0.01))

            self.action = self.take_action(state)  # direct action for test
            next_state, reward, done, _ = self.env.step(self.action)
            next_state = self.normalize(next_state)
            next_state.resize((self.state_dim + self.action_dim, 1))
            next_state[self.state_dim:].fill(amath.random.uniform(0, 0.01))
            next_state[self.state_dim + self.action] = 0.99
            total_reward += reward

            state = next_state
            if render:
                self.env.render()
            if done:
                break

        # print('Test Episodic reward:', total_reward)
        return total_reward

    def take_action(self, state):

        if self.exploration == "Boltzmann":
            """ Calculate probability of actions """
            aprob = self.model.prob_action(state)

            """ Purely Boltzmann exploration policy"""
            return amath.argmax(aprob[self.state_dim:self.state_dim + self.action_dim])

        else:
            """ Calculate probability of actions """
            aprob = self.model.Q_next(state)

            return np.argmax(aprob[self.state_dim:self.state_dim + self.action_dim])


if __name__ == "__main__":

    pass
