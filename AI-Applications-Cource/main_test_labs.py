from source import *
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import trange
from copy import deepcopy
import glob
import random
import os
import torch
from torch import nn

##################################################
class DQNAgent:
    def __init__(self, state_shape, n_actions, network, epsilon=0):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.model = network

    def __call__(self, state_t):
        # pass the state at time t through the network to get Q(s,a)
        qvalues = self.model(state_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and output is Qvalues as numpy array
        qvalues = self.model(states)
        return qvalues.numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def dqn_main(env_name, optimizer, epsilon=0.5, buffer_size=10_000, n_steps=100,
             timesteps_per_epoch=1, batch_size=32, total_steps=10_000,
             start_epsilon=1, end_epsilon=0.05, eps_decay_final_step=20_000,
             refresh_target_network_freq=100, max_grad_norm=5000, do_eval=False,
             n_games=30, greedy=True, t_max=1000, save_anime=False, save_dir='./videos/dqn'):
    # define the environment
    env = make_env(env_name)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()

    # build agent network and target network
    network = tf.keras.models.Sequential()
    network.add(layers.Input(shape=(state_shape[0],)))
    network.add(layers.Dense(192, activation="relu"))
    network.add(layers.Dense(256, activation="relu"))
    network.add(layers.Dense(64, activation="relu"))
    network.add(layers.Dense(n_actions))

    target = deepcopy(network)

    agent = DQNAgent(state_shape, n_actions, network, epsilon=epsilon)
    target_network = DQNAgent(agent.state_shape, agent.n_actions, target, epsilon)
    target_network.model.set_weights(agent.model.get_weights())

    exp_replay = ReplayBuffer(buffer_size)
    for i in range(n_steps):
        play_and_record(state, agent, env, exp_replay, n_steps=n_steps)
        if len(exp_replay) == buffer_size:
            break

    state = env.reset()
    for step in trange(total_steps + 1):
        # reduce exploration as we progress
        agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, step, eps_decay_final_step)
        # take timesteps_per_epoch and update experience replay buffer
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)
        # train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)
        with tf.GradientTape() as tape:
            # loss = <compute TD loss>
            loss = compute_td_loss(agent, target_network,
                                   states, actions, rewards, next_states, done_flags,
                                   gamma=0.99)
        gradients = tape.gradient(loss, agent.model.trainable_variables)
        clipped_grads = [tf.clip_by_norm(g, max_grad_norm) for g in gradients]
        optimizer.apply_gradients(zip(clipped_grads, agent.model.trainable_variables))
        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.model.set_weights(agent.model.get_weights())

    if do_eval:
        final_score = evaluate(make_env(env_name), agent, n_games=n_games, greedy=greedy, t_max=t_max)
        print(f'final_score [dqn]: {final_score}')
    if save_anime:
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)
    return agent


def dqn_test():
    env_name = 'CartPole-v1'
    epsilon = 0.5
    buffer_size = 10_000
    n_steps = 100

    # setup some parameters for training
    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 50_000

    # init Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # set exploration epsilon
    start_epsilon = 1
    end_epsilon = 0.05
    eps_decay_final_step = 2 * 10_000

    # setup some frequency for logging and updating target network
    refresh_target_network_freq = 100

    # for evaluation
    do_eval = False
    n_games = 30
    greedy = True
    t_max = 1000

    # for saving animation
    save_anime = False
    save_dir = './videos/dqn/'

    # to clip the gradients
    max_grad_norm = 5000

    # get the trained agent
    dqn_main(env_name=env_name, epsilon=epsilon, buffer_size=buffer_size, n_steps=n_steps,
             timesteps_per_epoch=timesteps_per_epoch, batch_size=batch_size, total_steps=total_steps,
             optimizer=optimizer, start_epsilon=start_epsilon, end_epsilon=end_epsilon,
             eps_decay_final_step=eps_decay_final_step, refresh_target_network_freq=refresh_target_network_freq,
             max_grad_norm=max_grad_norm, do_eval=do_eval, n_games=n_games, greedy=greedy, t_max=t_max,
             save_anime=save_anime, save_dir=save_dir)

##################################################
class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, n_actions):
        """Initialize."""
        super().__init__()
        self.n_actions = n_actions
        self.common = layers.Dense(192, activation="relu")
        self.actor = layers.Dense(self.n_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor): # -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


def actor_critic_main(env_name, optimizer, train_episodes=1000,
                      do_eval=False, n_games=30, greedy=True, t_max=1000,
                      save_anime=False, save_dir='./videos/actor_critic/'):
    env = make_env(env_name)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    agent = ActorCritic(n_actions)

    total_rewards = []
    for i in range(train_episodes):
        states, actions, rewards = generate_trajectory(env, agent)
        reward = train_one_episode_actor_critic_base(states, actions, rewards, agent, optimizer)
        total_rewards.append(reward)
        if do_eval and i != 0 and i % 100 == 0:
            mean_reward = np.mean(total_rewards[-100:-1])
            print(f'mean reward:%.3f {mean_reward}')
            if mean_reward > 700:
                break
    env.close()
    if do_eval:
        final_score = evaluate(make_env(env_name), agent, n_games=n_games, greedy=greedy, t_max=t_max)
        print(f'final_score [actor_critic_base]: {final_score}')
    if save_anime:
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)
    return agent


def actor_critic_test():
    env_name = 'CartPole-v1'
    optimizer = tf.keras.optimizers.Adam()

    train_episodes = 10_000
    do_eval = True
    save_anime = True

    actor_critic_main(env_name=env_name, optimizer=optimizer, train_episodes=train_episodes,
                      do_eval=do_eval, save_anime=save_anime)

##################################################
def ppo_main(env_name, total_timesteps=10_000, do_eval=False, n_eval_episodes=10,
             save_anime=False, save_dir='./videos/actor_critic/'):
    from stable_baselines3 import PPO
    from stable_baselines3.ppo.policies import MlpPolicy
    from stable_baselines3.common.evaluation import evaluate_policy

    env = make_env(env_name)
    agent = PPO(MlpPolicy, env, verbose=0)
    # train the agent
    agent.learn(total_timesteps=total_timesteps)

    if do_eval:
        final_score = evaluate_policy(agent, env, n_eval_episodes=n_eval_episodes)
        print(f'final_score [ppo]: {final_score}')
    if save_anime:
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)

    return agent


def ppo_test():
    env_name = 'CartPole-v1'
    total_timesteps = 20_000
    do_eval = True
    n_eval_episodes = 100
    save_anime = False

    ppo_main(env_name=env_name, total_timesteps=total_timesteps, do_eval=do_eval,
             n_eval_episodes=n_eval_episodes, save_anime=save_anime)

##################################################
class MLPActor_QQPG(tf.keras.Model):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.actor = layers.Dense(act_dim)

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        x = self.actor(x)
        x = tf.keras.activations.tanh(x)  # to output in range(-1,1)
        x = self.act_limit * x
        return x


class MLPQFunction_QQPG(tf.keras.Model):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.Q = layers.Dense(1)

    def call(self, s, a):
        x = tf.concat([s, a], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.Q(x)
        return tf.squeeze(q, -1)


class MLPActorCritic_QQPG(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build Q and policy functions
        self.q = MLPQFunction_QQPG(self.state_dim, self.act_dim)
        self.policy = MLPActor_QQPG(self.state_dim, self.act_dim, self.act_limit)

    def act(self, state):
        return self.policy(state).numpy()

    def get_action(self, s, noise_scale):
        a = self.act(s.reshape(1, -1).astype("float32")).reshape(-1)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


def ddpg_main(env_name, steps_per_epoch=5000, epochs=5, replay_size=1_000_000, batch_size=32,
              update_every=50, start_steps=10_000, save_anime=False):
    env = make_env(env_name)
    test_env = make_env(env_name)
    agent = MLPActorCritic_QQPG(env.observation_space, env.action_space)

    agent = ddpg(env=env, test_env=test_env, agent=agent, steps_per_epoch=steps_per_epoch,
                 epochs=epochs, replay_size=replay_size, batch_size=batch_size,
                 update_every=update_every, start_steps=start_steps)

    if save_anime:
        save_dir = './videos/ddpg/'
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)


def ddpg_test():
    env_name = 'Pendulum-v1'
    steps_per_epoch = 5000
    epochs = 5
    save_anime = False
    ddpg_main(env_name=env_name, steps_per_epoch=steps_per_epoch, epochs=epochs, save_anime=save_anime)

##################################################
class MLPActor_TD3(tf.keras.Model):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.actor = layers.Dense(act_dim)

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        x = self.actor(x)
        x = tf.keras.activations.tanh(x)  # to output in range(-1,1)
        x = self.act_limit * x
        return x


class MLPQFunction_TD3(tf.keras.Model):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.Q = layers.Dense(1)

    def call(self, s, a):
        x = tf.concat([s, a], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.Q(x)
        return tf.squeeze(q, -1)


class MLPActorCritic_TD3(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        # build Q and policy functions
        self.q1 = MLPQFunction_TD3(self.state_dim, self.act_dim)
        self.q2 = MLPQFunction_TD3(self.state_dim, self.act_dim)
        self.policy = MLPActor_TD3(self.state_dim, self.act_dim, self.act_limit)

    def act(self, state):
        return self.policy(state).numpy()

    def get_action(self, s, noise_scale):
        a = self.act(s.reshape(1, -1).astype("float32")).reshape(-1)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


def td3_main(env_name, steps_per_epoch=5000, epochs=5, replay_size=1_000_000, batch_size=32,
             update_every=50, start_steps=10_000, save_anime=False):
    env = make_env(env_name)
    test_env = make_env(env_name)
    agent = MLPActorCritic_TD3(env.observation_space, env.action_space)

    agent = td3(env=env, test_env=test_env, agent=agent, steps_per_epoch=steps_per_epoch,
                epochs=epochs, replay_size=replay_size, batch_size=batch_size,
                update_every=update_every, start_steps=start_steps)

    if save_anime:
        save_dir = './videos/td3/'
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)


def td3_test():
    env_name = 'Pendulum-v1'
    steps_per_epoch = 5000
    epochs = 5
    save_anime = False
    td3_main(env_name=env_name, steps_per_epoch=steps_per_epoch, epochs=epochs, save_anime=save_anime)

##################################################
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.mu_layer = layers.Dense(act_dim)
        self.log_std_layer = layers.Dense(act_dim)
        self.act_limit = act_limit

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        mu *= self.act_limit
        pi *= self.act_limit
        return mu, pi, logp_pi


class MLPQFunction_SAC(tf.keras.Model):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.Q = layers.Dense(1)

    def call(self, s, a):
        x = tf.concat([s, a], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.Q(x)
        return tf.squeeze(q, -1)


class MLPActorCritic_SAC(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        # build Q and policy functions
        self.q1 = MLPQFunction_SAC(self.state_dim, self.act_dim)
        self.q2 = MLPQFunction_SAC(self.state_dim, self.act_dim)
        self.policy = SquashedGaussianMLPActor(self.state_dim, self.act_dim, self.act_limit)

    def act(self, state, deterministic=False):
        mu, pi, _ = self.policy(state)
        if deterministic:
            return mu.numpy()
        else:
            return pi.numpy()

    def get_action(self, state, deterministic=False):
        return self.act(state.reshape(1, -1).astype("float32")).reshape(-1)


def sac_main(env_name, steps_per_epoch=5000, epochs=5, replay_size=1_000_000, batch_size=32,
             update_every=50, start_steps=10_000, save_anime=False):
    env = make_env(env_name)
    test_env = make_env(env_name)
    agent = MLPActorCritic_SAC(env.observation_space, env.action_space)

    agent = sac(env=env, test_env=test_env, agent=agent, steps_per_epoch=steps_per_epoch,
                epochs=epochs, replay_size=replay_size, batch_size=batch_size,
                update_every=update_every, start_steps=start_steps)

    if save_anime:
        save_dir = './videos/sac/'
        env = gym.make(env_name)
        generate_animation(env, agent, save_dir=save_dir)
        [filepath] = glob.glob(os.path.join(save_dir, '*.mp4'))
        display_animation(filepath)


def sac_test():
    env_name = 'Pendulum-v1'
    steps_per_epoch = 5000
    epochs = 5
    save_anime = False
    sac_main(env_name=env_name, steps_per_epoch=steps_per_epoch, epochs=epochs, save_anime=save_anime)

##################################################
def dbvae_test_mit():
    # load training dataset
    path_to_training_data = tf.keras.utils.get_file('train_face.py',
                                                    'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
    loader = TrainingDatasetLoader(path_to_training_data)
    # get training faces from data loader
    all_faces = loader.get_all_train_faces()

    # hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    latent_dim = 100
    epochs = 6
    encoder_dims = 2 * latent_dim + 1

    # initiate a new DB-VAE model and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # to use all available GPUs for training the model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        encoder = define_classifier(encoder_dims, n_filters=16, kernel_size=5)
        decoder = define_decoder_network(n_filters=16)
        dbvae = DB_VAE(encoder, decoder, latent_dim)

    # train the model
    dbvae_train(
        dbvae_model=dbvae,
        optimizer=optimizer,
        train_dataset=all_faces,
        dataset_loader=loader,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
        do_display=False)


if __name__ == '__main__':
    # for disabling tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    dqn_test()
    # actor_critic_test()
    # ppo_test()
    # ddpg_test()
    # td3_test()
    # sac_test()
    # dbvae_test_mit()
