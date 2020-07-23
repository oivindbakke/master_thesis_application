import gym
import numpy as np
import math
from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
learning_rate = 0.1  # alpha
gamma = 1  # discount factor

n_episodes = 1000

epsilon = 1  # Exploration "rate"
start_decay = 1
end_decay = n_episodes // 2
decay_value = epsilon / (end_decay - start_decay)

'''
num_bins = [6, 12]
# Bounding the infinite angular velocity of the pole ( is +- inf )
min_max_ang_vel = math.radians(200)
bin_step_size = [(env.observation_space.high[2] - env.observation_space.low[2]) / num_bins[0], (min_max_ang_vel*2) / num_bins[1]]

# Init Q table all zeros
Q_table = np.zeros(num_bins + [env.action_space.n, ])

def discrete_state(cont_state):
    discrete_state = [int((cont_state[2] - env.observation_space.low[2]) / bin_step_size[0]), int((cont_state[3] + min_max_ang_vel) / bin_step_size[1])]
    return tuple(discrete_state)
'''

num_bins = (6, 12)
l_bounds = [env.observation_space.low[2], -math.radians(50)]
u_bounds = [env.observation_space.high[2], math.radians(50)]


def discrete_state(_, __, pole_angle, pole_vel) -> Tuple[int, ...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    est.fit([l_bounds, u_bounds])
    return tuple(map(int, est.transform([[pole_angle, pole_vel]])[0]))


Q_table = np.zeros(num_bins + (env.action_space.n,))


def get_action(state):
    return np.argmax(Q_table[state])


def calc_new_Q_value(reward, new_state, cur_state):
    future_max_value = np.max(Q_table[new_state])
    current_q_value = Q_table[cur_state + (action,)]
    # Q_new = Q_old + Alpha(reward + discount*max_fut_value - Q_old)
    new_value = (1 - learning_rate) * current_q_value + learning_rate * (reward + gamma * future_max_value)
    return new_value


reward_matrix = []
for episode in range(n_episodes):
    tot_reward = 0
    # Discretize state into buckets and set done to false
    current_disc_state, done = discrete_state(*env.reset()), False
    while not done:
        # policy action
        action = get_action(current_disc_state)
        # insert random action
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore
        # increment environment
        obs, reward, done, _ = env.step(action)

        if done and episode < 200:
            reward = -3
        # print(reward)
        tot_reward += reward
        new_disc_state = discrete_state(*obs)
        # Calculate new Q value and update table
        Q_table[current_disc_state + (action,)] = calc_new_Q_value(reward, new_disc_state, current_disc_state)
        current_disc_state = new_disc_state
        # env.render()
    # print(tot_reward)
    reward_matrix.append(tot_reward)

    if end_decay >= episode >= start_decay:
        epsilon -= decay_value
plt.plot(reward_matrix, '.')
plt.show()
env.close()
# print(Q_table)

# policy = lambda _, __, ___, tip_velocity: int(tip_velocity > 0)
# # #policy = lambda obs: 1
# for _ in range(1):
#     state = env.reset()
#     for _ in range(30):
#         actions = policy(*state)
#         state, reward, done, info = env.step(actions)
#         env.render()
#         time.sleep(0.05)
# env.close()

'''


def learning_rate(n : int , min_rate=0.01 ) -> float :
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float:
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
'''