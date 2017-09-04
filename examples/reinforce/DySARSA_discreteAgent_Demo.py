import gym
from pydybm.reinforce.dysarsa import DYSARSA
from pydybm.reinforce.discrete_agent import DySARSAAgent
from pydybm.base.sgd import RMSProp, AdaGrad, NoisyRMSProp, ADAM
import pydybm.arraymath as amath

"""
__Author__: Sakyasingha Dasgupta
__copyright__ = "(C) Copyright IBM Corp. 2016"
"""


def atari_environment(learn_from_ram=True):

    if learn_from_ram:
        environment = "SpaceInvaders-ram-v0"
    else:

        environment = "SpaceInvaders-v0"

    return environment


def classical_environment():
    return "CartPole-v0"

environment = atari_environment()  # when using atari environment set frame_skip=True else False
env = gym.make(environment)  # Change to any other environments
observation = env.reset()

action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

"""
DySARSA model parameters

"""
temperature = 0.1
delay = 2
decay = [0.5]
discount = 0.99

SGD = ADAM()
learning_rate = 0.001

init_epsilon = 1.0  # initial exploration term (epsilon-greedy)
final_epsilon = 0.1  # final exploration term (when using annealing)

exploration = "Boltzmann"  # if not set to Boltzmann standard epsilon-greedy policy with annealing will be used

# If UseQLearning is false, agent will learn using default SARSA procedure
UseQLearning = True

steps_per_episode = 18000
train_steps = 10000

print('******************************')
print('DySARSA RL for , environment')
print('******************************')
print('RL Parameter initialization')
print('---------------------------')
print('action dim ', action_dim)
print('state dim ', state_dim)
print('exploration is ', exploration)
print('temperature = ', temperature)
print('initial epsilon = ', init_epsilon)
print('final epsilon = ', final_epsilon)
print('discount factor = ', discount)
print('delay = ', delay)
print('decay rate = ', decay)
if UseQLearning:
    print('Use DyBM Q-learning is , UseQLearning')
else:
    print('Use DyBM SARSA-learning is, True')
print('steps per episode = ', steps_per_episode)
print('total training steps = ', train_steps)
print('learning rate = ', learning_rate)

amath.random.seed(7)

"""Create DySARSA function approximator model"""
DySARSA_model = DYSARSA(state_dim, action_dim, delay, decay, discount, SGD,
                        learning_rate, temperature, insert_to_etrace="w_delay", L1=0.01)

is_train = True

if is_train:

    print('\n\nTraining new agent with DySARSA RL agent')

    agent = DySARSAAgent(env=env, model=DySARSA_model, steps_per_episode=steps_per_episode, train_steps=train_steps, exploration=exploration,
                         init_epsilon=temperature, final_epsilon=final_epsilon, UseQLearning=False, frame_skip=True,
                         threshold=4)

    agent.fit(test_every=5, test_num_eps=5, break_reward=500, render=False)

else:

    print('Testing DySARSA RL agent')

    agent = DySARSAAgent(env=env, model=DySARSA_model)
    for i in range(5):
        agent.predict(render=True)
