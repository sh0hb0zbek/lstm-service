import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import Model, Input
from copy import deepcopy
import gym
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

class ReplayBuffer:
    def __init__(self, size=1e6):
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
        
    def sample(self, batch_size=32):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)



def compute_q_loss_sac(agent, target_network, states, actions, rewards, next_states, done_flags,
                       gamma, alpha, act_limit, tape):
    # convert numpy array to proper data types
    states = states.astype('float32')
    actions = actions.astype('float32')
    rewards = rewards.astype('float32')
    next_states = next_states.astype('float32')
    done_flags = done_flags.astype('float32')
    # get q-values for all actions in current states
    # use agent network
    q1 = agent.q1(states, actions)
    q2 = agent.q2(states, actions)
    # Bellman backup for Q function
    with tape.stop_recording():
        mu, action_target, action_target_logp = target_network.policy(next_states)
        # Target Q            
        q1_target = target_network.q1(next_states, action_target)
        q2_target = target_network.q2(next_states, action_target)
        q_target = tf.minimum(q1_target, q2_target) 
        target = rewards + gamma * (1 - done_flags) * (q_target-alpha*action_target_logp)
    # MSE loss against Bellman backup
    loss_q1 = tf.reduce_mean((q1 - target)**2)
    loss_q2 = tf.reduce_mean((q2 - target)**2)
    loss_q = loss_q1 + loss_q2
    return loss_q

def compute_policy_loss_sac(agent, states, alpha, tape):
    # convert numpy array to proper data type
    states = states.astype('float32')
    mu, actions, actions_logp = agent.policy(states)
    q1_values = agent.q1(states, actions)
    q2_values = agent.q2(states, actions)
    q_values = tf.minimum(q1_values, q2_values)
    # Entropy regularised
    loss_policy = - tf.reduce_mean(q_values - alpha*actions_logp)
    return loss_policy

def one_step_update_sac(agent, target_network, q_params, q_optimizer, policy_optimizer, 
                        states, actions, rewards, next_states, done_flags,
                        gamma, polyak, alpha, act_limit):
    #one step gradient for q-values
    with tf.GradientTape() as tape:
        loss_q = compute_q_loss_sac(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma, alpha, act_limit, tape)
        gradients = tape.gradient(loss_q, q_params)
        q_optimizer.apply_gradients(zip(gradients, q_params))
    #Freeze Q-network
    agent.q1.trainable=False 
    agent.q2.trainable=False
    #one setep gradient for policy network
    with tf.GradientTape() as tape:
        loss_policy = compute_policy_loss_sac(agent, states, alpha, tape)
        gradients = tape.gradient(loss_policy, agent.policy.trainable_variables)
        policy_optimizer.apply_gradients(zip(gradients, agent.policy.trainable_variables))
    #UnFreeze Q-network
    agent.q1.trainable=True 
    agent.q2.trainable=True
    # update target networks with polyak averaging
    updated_model_weights = []
    for weights, weights_target in zip(agent.get_weights(), target_network.get_weights()):
        new_weights = polyak*weights_target+(1-polyak)*weights
        updated_model_weights.append(new_weights)
    target_network.set_weights(updated_model_weights)

def test_agent(env, agent, num_test_episodes, max_ep_len):
    ep_rets, ep_lens = [], []
    for j in range(num_test_episodes):
        state, done, ep_ret, ep_len = env.reset(), False, 0, 0
        while not(done or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            state, reward, done, _ = env.step(agent.get_action(state, True))
            ep_ret += reward
            ep_len += 1
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
    return np.mean(ep_rets), np.mean(ep_lens)

def sac(env, test_env, agent, steps_per_epoch=1, epochs=1, replay_size=1e6, batch_size=64,
        start_steps=1, update_every=1, seed=0, gamma=0.99, polyak=0.995, policy_lr=1e-3,
        q_lr=1e-3, alpha=0.2, update_after=1000, num_test_episodes=10, max_ep_len=1000):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    ep_rets, ep_lens = [], []
    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    # force a build of model to initialize the model parameters
    s = env.reset()
    a = env.action_space.sample()    
    agent.policy(np.array([s], dtype=np.float32))
    agent.q1(np.array([s],dtype=np.float32),np.array([a],dtype=np.float32))
    agent.q2(np.array([s],dtype=np.float32),np.array([a],dtype=np.float32))
    # make target network as a deep copy
    target_network = deepcopy(agent)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    target_network.policy.trainable=False
    target_network.q1.trainable = False
    target_network.q2.trainable = False
    # Experience buffer
    replay_buffer = ReplayBuffer(replay_size)
    # List of parameters for both Q-networks 
    q_params = agent.q1.trainable_variables+agent.q2.trainable_variables
    #optimizers
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)
    total_steps = steps_per_epoch*epochs
    state, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(total_steps):
        if t > start_steps:
            action = agent.get_action(state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        ep_ret += reward
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len==max_ep_len else done
        # Store experience to replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            state, ep_ret, ep_len = env.reset(), 0, 0
        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                states, actions, rewards, next_states, done_flags = replay_buffer.sample(batch_size)
                one_step_update_sac(
                        agent, target_network, q_params, q_optimizer, policy_optimizer, 
                        states, actions, rewards, next_states, done_flags,
                        gamma, polyak, alpha, act_limit
                )
        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            avg_ret, avg_len = test_agent(test_env, agent, num_test_episodes, max_ep_len)
            print("End of epoch: {:.0f}, Training Average Reward: {:.0f}, Training Average Length: {:.0f}".format(epoch, np.mean(ep_rets), np.mean(ep_lens)))
            print("End of epoch: {:.0f}, Test Average Reward: {:.0f}, Test Average Length: {:.0f}".format(epoch, avg_ret, avg_len))
            ep_rets, ep_lens = [], []
    return agent


def generate_animation(env, agent, save_dir):
    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda id: True, force=True, mode='evaluation')
    except gym.error.Error as e:
        print(e)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state, done, ep_ret, ep_len = env.reset(), False, 0, 0
    while not done:
        # Take deterministic actions at test time (noise_scale=0)
        state, reward, done, _ = env.step(agent.get_action(state, 0))
        ep_ret += reward
        ep_len += 1
    print('Reward: {}'.format(ep_ret))
    env.close()
            
def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))

