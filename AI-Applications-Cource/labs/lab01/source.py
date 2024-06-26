import gym
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from IPython.display import HTML
import glob
from tiles3 import tiles, IHT


# make an OpenAI Gym Environment
def make_env(env_name):
    return gym.make(env_name)


def accumulating_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] += 1
    return trace


def replacing_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] = 1
    return trace


class QEstimator:

    def __init__(self, env, step_size, num_of_tilings=8, tiles_per_dim=8, 
                 max_size=2048, epsilon=0.0, trace_fn=replacing_trace, 
                 lambd=0, gamma=1.0):
        self.env = env
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.tiles_per_dim = tiles_per_dim
        self.epsilon = epsilon
        self.lambd = lambd
        self.gamma = gamma
        
        self.step_size = step_size / num_of_tilings
        self.trace_fn = trace_fn
        
        self.table = IHT(max_size)
        
        self.w = np.zeros(max_size)
        self.trace = np.zeros(max_size)
        
        self.pos_scale = self.tiles_per_dim / (self.env.observation_space.high[0] \
                                                  - self.env.observation_space.low[0])
        self.vel_scale = self.tiles_per_dim / (self.env.observation_space.high[1] \
                                                  - self.env.observation_space.low[1])
        
    def get_active_features(self, state, action):
        pos, vel = state
        active_features = tiles(self.table, self.num_of_tilings,
                            [self.pos_scale * (pos - self.env.observation_space.low[0]), 
                             self.vel_scale * (vel- self.env.observation_space.low[1])],
                            [action])
        return active_features
        
    def q_predict(self, state, action):
        pos, vel = state
        if pos == self.env.observation_space.high[0]:  # reached goal
            return 0.0
        else:
            active_features = self.get_active_features(state, action)
            return np.sum(self.w[active_features])
        
    
    # learn with given state, action and target
    def q_update(self, state, action, reward, next_state, next_action):

        active_features = self.get_active_features(state, action)

        q_s_a = self.q_predict(state, action)
        target = reward + self.gamma * self.q_predict(next_state, next_action)
        delta = (target - q_s_a)

        if self.trace_fn == accumulating_trace or self.trace_fn == replacing_trace:
            self.trace = self.trace_fn(self.trace, active_features, self.gamma, self.lambd)
        else:
            self.trace = self.trace_fn(self.trace, active_features, self.gamma, 0)
                
        self.w += self.step_size * delta * self.trace        
        #self.w += self.step_size * delta * self.trace        
        
    def get_eps_greedy_action(self, state):
        pos, vel = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            qvals = np.array([self.q_predict(state, action) for action in range(self.env.action_space.n)])
            return np.argmax(qvals)


def sarsa_lambda(env, qhat, episode_cnt = 10000, max_size=2048, gamma=1.0):
    episode_rewards = []
    for i in range(episode_cnt):
        state = env.reset()
        action = qhat.get_eps_greedy_action(state)
        qhat.trace = np.zeros(max_size)
        episode_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = qhat.get_eps_greedy_action(next_state)
            episode_reward += reward
            qhat.q_update(state, action, reward, next_state, next_action)
            if done:
                episode_rewards.append(episode_reward)
                break
            state = next_state   
            action = next_action
    return np.array(episode_rewards)


# plot rewards
def plot_rewards(env_name, rewards, label):
    plt.title("env={}, Mean reward = {:.1f}".format(env_name,np.mean(rewards[-20:])))
    plt.plot(rewards, label=label)
    plt.grid()
    plt.legend()
    plt.ylim(-500, 0)
    plt.show()


def generate_animation(env, estimator, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda id: True, force=True, mode='evaluation')
    except gym.error.Error as e:
        print(e.what())

    state = env.reset()
    t = 0
    while True:
        time.sleep(0.01)
        action = estimator.get_eps_greedy_action(state)
        state, _, done, _ = env.step(action)
        env.render()
        t += 1
        if done:
            print('Solved in {} steps'.format(t))
            break


def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))