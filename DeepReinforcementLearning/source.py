import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
import random
import gym
from tqdm import trange
from IPython.display import clear_output
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_env(env_name, seed=None):
    # remove time limit wrapper from environment
    env = gym.make(env_name).unwrapped
    if seed is not None:
        env.seed(seed)
    return env


def define_network():
    pass


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, network, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.network = network

        # state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (input is state s)
        # and self.n_actions as output vector of logits of  q(s, a)

        # self.network = nn.Sequential()
        # self.network.add_module('layer1', nn.Linear(state_dim, 192))
        # self.network.add_module('relu1',  nn.ReLU())
        # self.network.add_module('layer2', nn.Linear(192, 256))
        # self.network.add_module('relu2',  nn.ReLU())
        # self.network.add_module('layer3', nn.Linear(256, 64))
        # self.network.add_module('relu3',  nn.ReLU())
        # self.network.add_module('layer4', nn.Linear(64, n_actions))

        self.parameters = self.network.parameters

    def forward(self, state_t):
        # pass the state at time t through the network to get Q(s, a)
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and output is Qvalues as numpy array
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_action(self, qvalues):
        # sample actions from a batch of qvalues using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            actions = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, info = env.step(actions)
            reward += r
            if done:
                break
        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size    # max number of items in buffer
        self.buffer = []    # array to hold buffer
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
        self.size = size    # max number of items in buffer
        self.buffer = []    # array to hold buffer
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
        self.priorities[idxs] = new_priorities+self.epsilon


def play_and_record(state_state, agent, env, exp_replay, n_steps=1):
    s = state_state
    sum_rewards = 0

    # play the game for n_steps and record transitions in buffer
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_action(qvalues)[0]
        next_s, r, done, info = env.step(a)
        sum_rewards += r
        exp_replay.add(s, a, r, next_s, done)
        if done:
            s = env.reset()
        else:
            s = next_s
    return sum_rewards, s


def compute_td_loss(agent, target_network, states, actions, rewards, next_states,
                    done_flags, gamma=0.99, dev=device):
    # convert numpy array to torch tensors
    states      = torch.tensor(states,      device=dev, dtype=torch.float)
    actions     = torch.tensor(actions,     device=dev, dtype=torch.long)
    rewards     = torch.tensor(rewards,     device=dev, dtype=torch.float)
    next_states = torch.tensor(next_states, device=dev, dtype=torch.float)
    done_flags  = torch.tensor(done_flags.astype('float32'), device=dev, dtype=torch.float)

    # get qvalues for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)

    # compute qvalues for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)

    # select qvalues for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next qvalues
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target qvalues"
    target_qvalues_for_actions = rewards + gamma*next_state_values*(1-done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach())**2)

    return loss


def compute_q_loss(agent, target_network, states, actions, rewards, next_states, done_flags, gamma=0.99):
    # convert numpy array to torch tensors
    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype('float32'), dtype=torch.float)
    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent.q(states, actions)
    # Bellman backup for Q function
    with torch.no_grad():
        q__next_state_values = target_network.q(next_states, target_network.policy(next_states))
        target = rewards + gamma * (1 - done_flags) * q__next_state_values
    # MSE loss against Bellman backup
    loss_q = ((predicted_qvalues - target) ** 2).mean()
    return loss_q


def predict_probs(model, states):
    """
    :param states: [batch, state_dim]
    :return probs: [batch, actions]
    """
    states = torch.tensor(states, device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(states)
    probs = nn.functional.softmax(logits, -1).detach().numpy()
    return probs


def generate_trajectory(env, model, n_steps=1000):
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    n_actions = env.action_space.n
    states, actions, rewards = [], [], []
    # initialize the environment
    s = env.reset()
    # generate n_steps of trajectory:
    for t in range(n_steps):
        action_probs = predict_probs(model, np.array([s]))[0]
        # sample action based on action_probs
        a = np.random.choice(n_actions, p=action_probs)
        next_state, r, done, _ = env.step(a)
        # update arrays
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = next_state
        if done:
            break
    return states, actions, rewards


def get_rewards_to_go(rewards, gamma=0.99):
    T = len(rewards)  # total number of individual rewards
    # empty array to return the rewards to go
    rewards_to_go = [0] * T
    rewards_to_go[T - 1] = rewards[T - 1]

    for i in range(T - 2, -1, -1):  # go from T-2 to 0
        rewards_to_go[i] = gamma * rewards_to_go[i + 1] + rewards[i]

    return rewards_to_go


def train_one_episode(model, states, actions, rewards, optimizer, gamma=0.99, entropy_coef=1e-2):
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards_to_go = torch.tensor(rewards_to_go, device=device, dtype=torch.float)

    # get action probabilities from states
    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = log_probs[range(len(actions)), actions]

    # Compute loss to be minized
    J = torch.mean(log_probs_for_actions * rewards_to_go)
    H = -(probs * log_probs).sum(-1).mean()

    loss = -(J + entropy_coef * H)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return np.sum(rewards)  # to show progress on training


def td_loss_ddqn(agent, target_network, states, actions, rewards, next_states,
                 done_flags, gamma=0.99, dev=device):
    # convert numpy array to torch tensors
    states      = torch.tensor(states,      device=dev, dtype=torch.float)
    actions     = torch.tensor(actions,     device=dev, dtype=torch.float)
    rewards     = torch.tensor(rewards,     device=dev, dtype=torch.float)
    next_states = torch.tensor(next_states, device=dev, dtype=torch.float)
    done_flags  = torch.tensor(done_flags,  device=dev, dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    q_s = agent(states)

    # select q-values for chosen actions
    q_s_a = q_s[range(len(actions)), actions]

    # compute q-values for all actions in next states
    # use agent network (online network)
    q_s1 = agent(next_states).detach()

    # compute Q argmax(next_states, actions) using predicted next q-values
    _, a1max = torch.max(q_s1, dim=1)

    # use target network to calculate the q value for best action chosen above
    q_s1_target = target_network(next_states)

    q_s1_a1max = q_s1_target[range(len(a1max)), a1max]

    # compute "target q-values"
    target_q = rewards + gamma*q_s1_a1max*(1-done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((q_s_a - target_q).pow(2))

    return loss


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel/np.sum(kernel)
    return convolve(values, kernel, 'valid')


class MLPActor(nn.Module):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, act_dim)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.actor(x)
        x = torch.tanh(x)  # to output in range(-1,1)
        x = self.act_limit * x
        return x


class MLPQFunction(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build Q and policy functions
        self.q = MLPQFunction(self.state_dim, self.act_dim)
        self.policy = MLPActor(self.state_dim, self.act_dim, self.act_limit)

    def act(self, state):
        with torch.no_grad():
            return self.policy(state).numpy()

    def get_action(self, s, noise_scale):
        a = self.act(torch.as_tensor(s, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


def train(env, agent, target_network, exp_replay, opt, loss_function, total_steps=1, timesteps_per_epoch=1, batch_size=32,
          max_grad_norm=5000, loss_freq=20, refresh_target_network_freq=100, eval_freq=None,
          start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=2*10**4):
    mean_rw_history = []
    td_loss_history = []
    state = env.reset()
    for step in trange(total_steps + 1):
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)

        # take timesteps_pre_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)
        states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)

        # loss = <compute TD loss
        loss = loss_function(agent, target_network, states, actions, rewards,
                             next_states, done_flags, gamma=0.99, dev=device)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())

        if step % refresh_target_network_freq == 0:
            # load agent weights into target network
            target_network.load_state_dict(agent.state_dict())

        if eval_freq is not None and step % eval_freq == 0:
            env_name = str(env.spec)[8:-1]
            # evaluate the agent
            mean_rw_history.append(evaluate(make_env(env_name, seed=step), agent, n_games=3, greedy=True, t_max=1000))
            clear_output(True)
            print(f'buffer size = {len(exp_replay)}, epsilon = {agent.epsilon:.5f}')

            plt.figure(figsize=[16, 5])
            plt.subplot(1, 2, 1)
            plt.title('Mean reward per episode')
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(td_loss_history[-1])
            plt.subplot(1, 2, 2)
            plt.title('TD loss history (smoothened)')
            plt.plot(smoothen(td_loss_history))
            plt.grid()
            plt.show()


def main_test_01(env_name, seed=None, buffer_size=10**4,
                 timesteps_per_epoch=1, batch_size=32, total_steps=50_000, lr=1e-4,
                 start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=20_000,
                 loss_freq=20, refresh_target_network_freq=100, eval_freq=None, max_grad_norm=5000):
    if seed is not None:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)

    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()
    network = nn.Sequential()
    network.add_module('layer1', nn.Linear(state_dim[0], 192))
    network.add_module('relu1',  nn.ReLU())
    network.add_module('layer2', nn.Linear(192, 256))
    network.add_module('relu2',  nn.ReLU())
    network.add_module('layer3', nn.Linear(256, 64))
    network.add_module('relu3',  nn.ReLU())
    network.add_module('layer4', nn.Linear(64, n_actions))

    agent = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_play = ReplayBuffer(buffer_size)
    for i in range(100):
        play_and_record(state, agent, env, exp_play, n_steps=100)
        if len(exp_play) == buffer_size:
            break

    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    train(env, agent, target_network, exp_play, opt, compute_td_loss, total_steps,
          timesteps_per_epoch, batch_size, max_grad_norm, loss_freq, refresh_target_network_freq,
          eval_freq, start_epsilon, end_epsilon, eps_decay_final_step)

    # train(env, agent, target_network, exp_play, opt, compute_td_loss, total_steps,
    #       timesteps_per_epoch, batch_size, max_grad_norm, loss_freq, refresh_target_network_freq,
    #       eval_freq, start_epsilon, end_epsilon, eps_decay_final_step)

    # evaluate
    score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print('Score:', score)

    return agent


def main_test_02_atari(env_name, seed=None, buffer_size=10**4,
                       timesteps_per_epoch=1, batch_size=32, total_steps=50_000, lr=1e-4,
                       start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=20_000,
                       loss_freq=20, refresh_target_network_freq=100, eval_freq=None, max_grad_norm=5000):
    if seed is not None:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)

    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()
    network = nn.Sequential()
    network.add_module('conv1',   nn.Conv2d(4, 16, kernel_size=8, stride=4))
    network.add_module('relu1',   nn.ReLU())
    network.add_module('conv2',   nn.Conv2d(16, 32, kernel_size=4, stride=2))
    network.add_module('relu2',   nn.ReLU())
    network.add_module('flatten', nn.Flatten())
    network.add_module('linear3', nn.Linear(2592, 256))     # 2592 calculated above
    network.add_module('relu3',   nn.ReLU())
    network.add_module('linear4', nn.Linear(256, n_actions))

    agent = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_play = ReplayBuffer(buffer_size)
    for i in range(100):
        play_and_record(state, agent, env, exp_play, n_steps=100)
        if len(exp_play) == buffer_size:
            break

    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    train(env, agent, target_network, exp_play, opt, compute_td_loss, total_steps,
          timesteps_per_epoch, batch_size, max_grad_norm, loss_freq, refresh_target_network_freq,
          eval_freq, start_epsilon, end_epsilon, eps_decay_final_step)

    # evaluate
    score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print('Score:', score)

    return agent


def main_test_03_ddqn(env_name, seed=None, buffer_size=10**4,
                      timesteps_per_epoch=1, batch_size=32, total_steps=50_000, lr=1e-4,
                      start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=20_000,
                      loss_freq=20, refresh_target_network_freq=100, eval_freq=None, max_grad_norm=5000):
    if seed is not None:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)

    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()
    network = nn.Sequential()
    network.add_module('layer1', nn.Linear(state_dim[0], 64))
    network.add_module('relu1',  nn.ReLU())
    network.add_module('layer2', nn.Linear(64, 128))
    network.add_module('relu2',  nn.ReLU())
    network.add_module('layer3', nn.Linear(128, 32))
    network.add_module('relu3',  nn.ReLU())
    network.add_module('layer4', nn.Linear(32, n_actions))

    agent = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_play = ReplayBuffer(buffer_size)
    for i in range(100):
        play_and_record(state, agent, env, exp_play, n_steps=100)
        if len(exp_play) == buffer_size:
            break

    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    train(env, agent, target_network, exp_play, opt, td_loss_ddqn, total_steps,
          timesteps_per_epoch, batch_size, max_grad_norm, loss_freq, refresh_target_network_freq,
          eval_freq, start_epsilon, end_epsilon, eps_decay_final_step)

    # evaluate
    score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print('Score:', score)

    return agent


def main_test_04_ppo(env_name, total_timesteps, do_evaluate=False):
    env = make_env(env_name)
    model = PPO(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    if do_evaluate:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f'mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    return model


def main_test_05_reinforce(env_name, total_steps, eval_freq=100, lr=1e-3):
    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = nn.Sequential(
        nn.Linear(state_dim, 192),
        nn.ReLU(),
        nn.Linear(192, n_actions)
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_rewards = []
    for i in range(total_steps):
        states, actions, rewards = generate_trajectory(env, model)
        reward = train_one_episode(model, states, actions, rewards, optimizer)
        total_rewards.append(reward)
        if (i+1) % eval_freq == 0:
            mean_reward = np.mean(total_rewards[-eval_freq:-1])
            print(f"mean_reward: {mean_reward:.3f}")
            if mean_reward > 300:
                break
    env.close()
    return model


def main_test_06_ddpg(env_name, seed=None, buffer_size=10**4,
                      timesteps_per_epoch=1, batch_size=32, total_steps=50_000, lr=1e-4,
                      start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=20_000,
                      loss_freq=20, refresh_target_network_freq=100, eval_freq=None, max_grad_norm=5000):
    if seed is not None:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)

    env = make_env(env_name)
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()
    network = nn.Sequential()
    network.add_module('layer1', nn.Linear(state_dim[0], 64))
    network.add_module('relu1',  nn.ReLU())
    network.add_module('layer2', nn.Linear(64, 128))
    network.add_module('relu2',  nn.ReLU())
    network.add_module('layer3', nn.Linear(128, 32))
    network.add_module('relu3',  nn.ReLU())
    network.add_module('layer4', nn.Linear(32, n_actions))

    agent = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network = DQNAgent(state_dim, n_actions, network, epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_play = ReplayBuffer(buffer_size)
    for i in range(100):
        play_and_record(state, agent, env, exp_play, n_steps=100)
        if len(exp_play) == buffer_size:
            break

    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    train(env, agent, target_network, exp_play, opt, td_loss_ddqn, total_steps,
          timesteps_per_epoch, batch_size, max_grad_norm, loss_freq, refresh_target_network_freq,
          eval_freq, start_epsilon, end_epsilon, eps_decay_final_step)

    # evaluate
    score = evaluate(make_env(env_name), agent, n_games=30, greedy=True, t_max=1000)
    print('Score:', score)

    return agent
