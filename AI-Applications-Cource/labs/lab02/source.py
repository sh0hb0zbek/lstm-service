import random
import tensorflow as tf
from tensorflow.keras import Model
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
import os
import io
import base64
import time
import glob
from IPython.display import HTML


def make_env(env_name, seed=None):
    # remove time limit wrapper from environment
    env = gym.make(env_name).unwrapped
    if seed is not None:
        env.seed(seed)
    return env


class DQNAgent:
    def __init__(self, state_shape, n_actions, network, epsilon=0):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.model = network
    
    def __call__(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.model(state_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        qvalues = self.model(states)
        return qvalues.numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues(np.array([s]))
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break
        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size #max number of items in buffer
        self.buffer =[] #array to holde buffer
        self.next_id = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
           self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size
        
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta=0.4):
        self.size = size #max number of items in buffer
        self.buffer =[] #array to holde buffer
        self.next_id = 0
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.ones(size)
        self.epsilon = 1e-5
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        max_priority = self.priorities.max()
        if len(self.buffer) < self.size:
           self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.priorities[self.next_id] = max_priority
        self.next_id = (self.next_id + 1) % self.size
        
    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        N = len(self.buffer)
        weights = (N * probabilities) ** (-self.beta)
        weights /= weights.max()
        idxs = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        weights = weights[idxs]
        return  (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(done_flags), np.array(weights), np.array(idxs))
        
    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities+self.epsilon


def play_and_record(start_state, agent, env, exp_replay, n_steps=1):
    s = start_state
    sum_rewards = 0
    # Play the game for n_steps and record transitions in buffer
    for _ in range(n_steps):
        qvalues = agent.get_qvalues(np.array([s]))
        a = agent.sample_actions(qvalues)[0]        
        next_s, r, done, _ = env.step(a)
        sum_rewards += r
        exp_replay.add(s, a, r, next_s, done)
        if done:
            s = env.reset()
        else:
            s = next_s
    return sum_rewards, s


def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99):
    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)
    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)
    # select q-values for chosen actions
    row_indices= tf.range(len(actions))
    indices = tf.transpose([row_indices, actions])
    predicted_qvalues_for_actions = tf.gather_nd(predicted_qvalues, indices)
    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)
    # compute "target q-values" 
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)
    # mean squared error loss to minimize
    loss = tf.keras.losses.MSE(target_qvalues_for_actions, predicted_qvalues_for_actions)
    return loss


def compute_td_loss_priority_replay(agent, target_network, replay_buffer, states,
                                    actions, rewards, next_states, done_flags,
                                    weights, buffer_idxs, gamma=0.99):
    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)
    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)
    # select q-values for chosen actions
    row_indices= tf.range(len(actions))
    indices = tf.transpose([row_indices, actions])
    predicted_qvalues_for_actions = tf.gather_nd(predicted_qvalues, indices)
    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)
    # compute "target q-values" 
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)
    # weighted mean squared error loss to minimize
    loss = (target_qvalues_for_actions - predicted_qvalues_for_actions) ** 2 * weights
    loss = tf.reduce_mean(loss)
    # calculate new priorities and update buffer
    new_priorities = (target_qvalues_for_actions - predicted_qvalues_for_actions).numpy().copy()
    new_priorities = np.absolute(new_priorities)
    replay_buffer.update_priorities(buffer_idxs, new_priorities)
    return loss


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step

def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')


def generate_animation(env, agent, save_dir):
    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda id: True, force=True, mode='evaluation')
    except gym.error.Error as e:
        print(e)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state = env.reset()
    reward = 0
    while True:
        qvalues = agent.get_qvalues(np.array([state]))
        action = qvalues.argmax(axis=-1)[0]
        state, r, done, _ = env.step(action)
        reward += r
        if done:
            print('Got reward: {}'.format(reward))
            break
            
def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
