import gym
import tensorflow as tf
from scipy.signal import convolve, gaussian
import os
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import tqdm
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from pathlib import Path


# make an OpenAI Gym Environment
def make_env(env_name, seed=None):
    # remove time limit wrapper from environment
    env = gym.make(env_name).unwrapped
    if seed is not None:
        env.seed(seed)
    return env


def accumulating_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] += 1
    return trace


def replacing_trace(trace, active_features, gamma, lambd):
    trace *= gamma * lambd
    trace[active_features] = 1
    return trace


def sarsa_lambda(env, qhat, episode_cnt=10000, max_size=2048, gamma=1.0):
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
    plt.title("env={}, Mean reward = {:.1f}".format(env_name, np.mean(rewards[-20:])))
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
        print(e)

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


def generate_animation_q_network(env, agent, save_dir):
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


def show_videos(video_path='', prefix=''):
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                       loop controls styles="height: 400px;">
                       <source src="data:video/mp4;base64,{}" type="video/mp4" />
                       </video>'''.format(mp4, video_64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(env_name, model, video_length=500, prefix='', video_folder='videos'):
    eval_env = DummyVecEnv([lambda: make_env(env_name)])
    # start the video at step=0 and record [video_length] steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step==0,
                                video_length=video_length, name_prefix=prefix)
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)
    # close the video recorder environment
    eval_env.close()


def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))


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
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
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
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
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
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(done_flags), np.array(weights), np.array(idxs))

    def update_priorities(self, idxs, new_priorities):
        self.priorities[idxs] = new_priorities + self.epsilon


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


# predict action probabilities
def sample_action(state, model, n_actions):
    """
    params: states: [batch, state_dim]
    returns: probs: [batch, n_actions]
    """
    logits,_ = model(state)
    action_probs = tf.nn.softmax(logits, -1).numpy()[0]
    action = np.random.choice(n_actions, p=action_probs)
    return action


# play game and generate trajectory
def generate_trajectory(env, model, n_steps=1000):
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    states, actions, rewards = [], [], []
    # initialize the environment
    s = env.reset()
    # generate n_steps of trajectory:
    for t in range(n_steps):
        # sample action based on action_probs
        a = sample_action(np.array([s], dtype=np.float32), model, env.action_space.n)
        next_state, r, done, _ = env.step(a)
        # update arrays
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = next_state
        if done:
            break
    return states, actions, rewards


# calculate rewards to go
def get_rewards_to_go(rewards, gamma=0.99):
    T = len(rewards) # total number of individual rewards
    # empty array to return the rewards to go
    rewards_to_go = [0]*T
    rewards_to_go[T-1] = rewards[T-1]
    for i in range(T-2, -1, -1): #go from T-2 to 0
        rewards_to_go[i] = gamma * rewards_to_go[i+1] + rewards[i]
    return rewards_to_go


# training one episode
def train_one_episode_actor_critic_base(states, actions, rewards, model, optimizer, gamma=0.99, entropy_coef=0.01):
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)
    # convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int)
    rewards_to_go = np.array(rewards_to_go, dtype=np.float32)
    with tf.GradientTape() as tape:
        # get action probabilities from states
        logits, state_values = model(states)
        probs = tf.nn.softmax(logits, -1)
        log_probs = tf.nn.log_softmax(logits, -1)
        row_indices= tf.range(len(actions))
        indices = tf.transpose([row_indices, actions])
        log_probs_for_actions = tf.gather_nd(log_probs, indices)
        advantage = rewards_to_go - state_values
        # Compute loss to be minimized
        J = tf.reduce_mean(log_probs_for_actions*advantage)
        H = -tf.reduce_mean(tf.reduce_sum(probs*log_probs, -1))
        loss = -(J+entropy_coef*H)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return np.sum(rewards) #to show progress on training


def compute_q_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                   gamma, tape):
    # convert numpy array to proper data types
    states = states.astype('float32')
    actions = actions.astype('float32')
    rewards = rewards.astype('float32')
    next_states = next_states.astype('float32')
    done_flags = done_flags.astype('float32')
    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent.q(states, actions)
    # Bellman backup for Q function
    with tape.stop_recording():
        q__next_state_values = target_network.q(next_states, target_network.policy(next_states))
        target = rewards + gamma * (1 - done_flags) * q__next_state_values
    # MSE loss against Bellman backup
    loss_q = tf.reduce_mean((predicted_qvalues - target)**2)

    return loss_q


def compute_policy_loss(agent, states, tape):
    # convert numpy array to proper data type
    states = states.astype('float32')
    predicted_qvalues = agent.q(states, agent.policy(states))
    loss_policy = - tf.reduce_mean(predicted_qvalues)
    return loss_policy


def one_step_update(agent, target_network, q_optimizer, policy_optimizer,
                    states, actions, rewards, next_states, done_flags,
                    gamma=0.99, polyak=0.995):
    #one step gradient for q-values
    with tf.GradientTape() as tape:
        loss_q = compute_q_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma, tape)
        gradients = tape.gradient(loss_q, agent.q.trainable_variables)
        q_optimizer.apply_gradients(zip(gradients, agent.q.trainable_variables))
    #Freeze Q-network
    agent.q.trainable=False
    #one setep gradient for policy network
    with tf.GradientTape() as tape:
        loss_policy = compute_policy_loss(agent, states, tape)
        gradients = tape.gradient(loss_policy, agent.policy.trainable_variables)
        policy_optimizer.apply_gradients(zip(gradients, agent.policy.trainable_variables))
    #UnFreeze Q-network
    agent.q.trainable=True
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
            # Take deterministic actions at test time (noise_scale=0)
            state, reward, done, _ = env.step(agent.get_action(state, 0))
            ep_ret += reward
            ep_len += 1
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
    return np.mean(ep_rets), np.mean(ep_lens)


def compute_q_loss_td3(agent, target_network, states, actions, rewards, next_states, done_flags,
                       gamma, target_noise, noise_clip, act_limit, tape):
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
        action_target = target_network.policy(next_states)
        # Target policy smoothing
        epsilon = tf.random.normal(action_target.shape) * target_noise
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        action_target = action_target + epsilon
        action_target = tf.clip_by_value(action_target, -act_limit, act_limit)
        q1_target = target_network.q1(next_states, action_target)
        q2_target = target_network.q2(next_states, action_target)
        q_target = tf.minimum(q1_target, q2_target)
        target = rewards + gamma * (1 - done_flags) * q_target
    # MSE loss against Bellman backup
    loss_q1 = tf.reduce_mean((q1 - target)**2)
    loss_q2 = tf.reduce_mean((q2 - target)**2)
    loss_q = loss_q1 + loss_q2
    return loss_q


def compute_policy_loss_td3(agent, states, tape):
    # convert numpy array to proper data type
    states = states.astype('float32')
    q1_values = agent.q1(states, agent.policy(states))
    loss_policy = - tf.reduce_mean(q1_values)
    return loss_policy


def one_step_update_td3(agent, target_network, q_params, q_optimizer, policy_optimizer,
                        states, actions, rewards, next_states, done_flags,
                        gamma, polyak, target_noise, noise_clip, act_limit,
                        policy_delay, timer):
    #one step gradient for q-values
    with tf.GradientTape() as tape:
        loss_q = compute_q_loss_td3(agent, target_network, states, actions, rewards, next_states, done_flags,
                        gamma, target_noise, noise_clip, act_limit, tape)
        gradients = tape.gradient(loss_q, q_params)
        q_optimizer.apply_gradients(zip(gradients, q_params))
    # Update policy and target networks after policy_delay updates of Q-networks
    if timer % policy_delay == 0:
        #Freeze Q-network
        agent.q1.trainable=False
        agent.q2.trainable=False
        #one setep gradient for policy network
        with tf.GradientTape() as tape:
            loss_policy = compute_policy_loss_td3(agent, states, tape)
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


def ddpg(env, test_env, agent, steps_per_epoch=1, epochs=1, replay_size=1, batch_size=128, start_steps=1,
         update_every=1, seed=0, gamma=0.99, polyak=0.995, policy_lr=1e-3, q_lr=1e-3, act_noise=0.1,
         update_after=1000, num_test_episodes=10, max_ep_len=1000):
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
    agent.q(np.array([s], dtype=np.float32), np.array([a], dtype=np.float32))
    # copy target network
    target_network = deepcopy(agent)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    target_network.policy.trainable = False
    target_network.q.trainable = False
    # Experience buffer
    replay_buffer = ReplayBuffer(replay_size)
    # optimizers
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)
    total_steps = steps_per_epoch * epochs
    state, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(total_steps):
        if t > start_steps:
            action = agent.get_action(state, act_noise)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        ep_ret += reward
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len == max_ep_len else done
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
            for _ in range(update_every):
                states, actions, rewards, next_states, done_flags = replay_buffer.sample(batch_size)
                one_step_update(
                    agent, target_network, q_optimizer, policy_optimizer,
                    states, actions, rewards, next_states, done_flags,
                    gamma, polyak
                )
        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            avg_ret, avg_len = test_agent(test_env, agent, num_test_episodes, max_ep_len)
            print("End of epoch: {:.0f}, Training Average Reward: {:.0f}, Training Average Length: {:.0f}".format(epoch,
                                                                                                                  np.mean(
                                                                                                                      ep_rets),
                                                                                                                  np.mean(
                                                                                                                      ep_lens)))
            print(
                "End of epoch: {:.0f}, Test Average Reward: {:.0f}, Test Average Length: {:.0f}".format(epoch, avg_ret,
                                                                                                        avg_len))
            ep_rets, ep_lens = [], []
    return agent


def td3(env, test_env, agent, steps_per_epoch=1, epochs=1, replay_size=1e6, batch_size=64, start_steps=1,
        update_every=1, seed=0, gamma=0.99, polyak=0.995, policy_lr=1e-3, q_lr=1e-3, act_noise=0.1,
        update_after=1000, target_noise=0.2, noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000):
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
            action = agent.get_action(state, act_noise)
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
                one_step_update_td3(
                        agent, target_network, q_params, q_optimizer, policy_optimizer,
                        states, actions, rewards, next_states, done_flags,
                        gamma, polyak, target_noise, noise_clip, act_limit, policy_delay, j
                )
        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            avg_ret, avg_len = test_agent(test_env, agent, num_test_episodes, max_ep_len)
            print("End of epoch: {:.0f}, Training Average Reward: {:.0f}, Training Average Length: {:.0f}".format(epoch, np.mean(ep_rets), np.mean(ep_lens)))
            print("End of epoch: {:.0f}, Test Average Reward: {:.0f}, Test Average Length: {:.0f}".format(epoch, avg_ret, avg_len))
            ep_rets, ep_lens = [], []
    return agent


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


def gaussian_likelihood(x, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi



# MIT deep learning labs
class TrainingDatasetLoader(object):
    def __init__(self, data_path):
        # print(f'Opening {data_path}')
        self.cache = h5py.File(data_path, 'r')
        # print('Loading data into memory ...')
        sys.stdout.flush()
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]

        self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]

    def get_train_size(self):
        return self.train_inds.shape[0]

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size() // factor // batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None,
                  return_inds=False):
        if only_faces:
            selected_inds = np.random_choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos)
        else:
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=n // 2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=n // 2, replace=False, p=p_neg)
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))
        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :, :, ::-1] / 255.).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]
        return (train_img, train_label, sorted_inds) if return_inds \
            else (train_img, train_label)

    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[:10 * n:10]]
        return (self.images[most_prob_inds, ...] / 255.).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]


class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = list()

    def append(self, value):
        self.loss.append(self.alpha * self.loss[-1] + (1 - self.alpha) * value \
                             if len(self.loss) > 0 else value)

    def get(self):
        return self.loss


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale

        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

            if self.scale is None:
                plt.plot(data)
            elif self.scale == 'semilogx':
                plt.semilogx(data)
            elif self.scale == 'semilogy':
                plt.semilogy(data)
            elif self.scale == 'loglog':
                plt.loglog(data)
            else:
                raise ValueError("unrecognized parameter scale {}".format(self.scale))

            plt.xlabel(self.xlabel);
            plt.ylabel(self.ylabel)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

            self.tic = time.time()


def plot_sample(x, y, vae):
    plt.figure(figsize=(2, 1))
    plt.subplot(1, 2, 1)

    idx = np.where(y == 1)[0][0]
    plt.imshow(x[idx])
    plt.grid(False)

    plt.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(recon[idx])
    plt.grid(False)

    plt.show()


def define_classifier(n_outputs=1, n_filters=8, kernel_size=3, strides=2, padding='same',
                      activation='relu', n_conv_layers=4, dense_units=[512]):
    model = tf.keras.Sequential()
    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv2D(filters=pow(2, i) * n_filters, kernel_size=kernel_size,
                                         strides=strides, padding=padding, activation=activation))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_outputs, activation='softmax'))
    return model


def define_decoder_network(n_filters=8, kernel_size=3, strides=2, padding='same',
                           activation='relu', n_conv_layers=3):
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Dense(units=4 * 4 * (pow(2, n_conv_layers) * n_filters), activation='relu'))
    decoder.add(tf.keras.layers.Reshape(target_shape=(4, 4, pow(2, n_conv_layers) * n_filters)))
    for i in range(n_conv_layers - 1, -1, -1):
        decoder.add(tf.keras.layers.Conv2DTranspose(filters=pow(2, i) * n_filters, kernel_size=kernel_size,
                                                    strides=strides, padding=padding, activation=activation))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kernel_size, strides=strides,
                                                padding=padding, activation=activation))
    return decoder


### Defining the VAE loss function ###
def vae_loss_function(x, x_recon, mu, logsigma, kl_weights=5e-4):
    '''
    Function to calculate VAE loss given:
          an input x,
          reconstructed output x_recon,
          encoded means mu,
          encoded log of standard deviation logsigma,
          weight parameter for the latent loss kl_weight
    '''
    latent_loss = 0.5 / tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma)
    reconstruction_loss = tf.reduce_mean(tf.abs(x - x_recon), axis=(1, 2, 3))
    vae_loss = kl_weights * latent_loss + reconstruction_loss
    return vae_loss


### VAE Reparameterization ###
def sampling(z_mean, z_logsigma):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
        # Returns
            z (tensor): sampled latent vector
    """
    batch, latent_dim = z_mean.shape
    epsilon = tf.random.normal(shape=(batch, latent_dim))
    z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
    return z


### Loss function for DB-VAE ###
def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    """
    Loss function for DB-VAE.
        # Arguments
            x: true input x
            x_pred: reconstructed x
            y: true label (face or not face)
            y_logit: predicted labels
            mu: mean of latent distribution (Q(z|X))
            logsigma: log of standard deviation of latent distribution (Q(z|X))
        # Returns
            total_loss: DB-VAE total loss
            classification_loss = DB-VAE classification loss
    """
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)
    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)
    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)
    total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)
    return total_loss, classification_loss


### defining and creating the DB-VAE ###
class DB_VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder

    # function to feed images into encoder, encode the latent space, and output
    # classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return y_logit, z_mean, z_logsigma

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)
        return z

    # decode the latent space and outpuy reconstruction
    def decode(self, z):
        recunstruction = self.decoder(z)
        return recunstruction

    # the call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # reparameterization
        z = self.reparameterize(z_mean, z_logsigma)

        # reconstruction
        recon = self.decode(z)

        return y_logit, z_mean, z_logsigma, recon

    # predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit


# function to return the means for an input image batch
def get_latent_mu(images, dbvae, batch_size=1024):
    N = images.shape[0]
    mu = np.zeros((N, dbvae.latent_dim))
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind + batch_size, N + 1)
        batch = (images[start_ind:end_ind]).astype(np.float32) / 255.
        _, batch_mu, _ = dbvae.encode(batch)
        mu[start_ind:end_ind] = batch_mu
    return mu


### Resampling algorithm for DB-VAE ###
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=1e-3):
    '''
    Function that recomputes the sampling probabilities for images within a batch
    based on how they distribute across the training data
    '''
    # print('Recomputing the sampling probabilities')

    # run the input batch and get the latent variable means
    mu = get_latent_mu(images, dbvae)

    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])

    # consider the distribution for each latent variable
    for i in range(dbvae.latent_dim):
        latent_distribution = mu[:, i]

        # generate a histogram of the latent distibution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data smaple falls
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        # call the digitize funciton to find which bins in the latent distribution
        # every data sample falls in to
        bin_idx = np.digitize(latent_distribution, bin_edges)

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density /= np.sum(hist_smoothed_density)

        # invert the density function
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])

        # normalizer all probabilities
        p /= np.sum(p)

        # update sampling probabilities by considering whether the newly computed
        # p is greater than the existinf sampling probabilities
        training_sample_p = np.maximum(p, training_sample_p)

    # final normalization
    training_sample_p /= np.sum(training_sample_p)
    return training_sample_p


@tf.function
def train_step(x, y, dbvae_model, optimizer):
    with tf.GradientTape() as tape:
        # feed input x into dbvae. Note that this is using the DB_VAE call function
        y_logit, z_mean, z_logsigma, x_recon = dbvae_model(x)

        # call the DB_VAE loss function to compute the loss
        loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)
    # use the `GradientTaoe.gradient` method to compute the gradients
    grads = tape.gradient(loss, dbvae_model.trainable_variables)

    # apply gradients to variables
    optimizer.apply_gradients(zip(grads, dbvae_model.trainable_variables))
    return loss


def dbvae_train(dbvae_model, optimizer, train_dataset, dataset_loader, epochs=10, batch_size=32,
                learning_rate=1e-4, latent_dim=100, do_display=False):
    for i in range(epochs):
        if do_display:
            ipythondisplay.clear_output(wait=True)
            print(f'Starting epoch {i + 1:2d}/{epochs}')
        # recompute data sampling probabilities
        p_faces = get_training_sample_probabilities(train_dataset, dbvae_model)

        # get a batch of training data and compute the training step
        loop = range(dataset_loader.get_train_size() // batch_size)
        if do_display: loop = tqdm(loop)
        for j in loop:
            # load a batch of data
            (x, y) = dataset_loader.get_batch(batch_size, p_pos=p_faces)
            # loss optimization
            loss = train_step(x, y, dbvae_model, optimizer)

            if j % 500 == 0 and do_display:
                plot_sample(x, y, dbvae_model)
