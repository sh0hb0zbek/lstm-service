import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import gym
import matplotlib.pyplot as plt
import os
import io
import base64
from IPython.display import HTML
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def make_env(env_name, seed=None):
    env = gym.make(env_name).unwrapped
    if seed is not None:
        env.seed(seed)
    return env

# actor critic network
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
    #generate n_steps of trajectory:
    for t in range(n_steps):
        #sample action based on action_probs
        a = sample_action(np.array([s], dtype=np.float32), model, env.action_space.n)
        next_state, r, done, _ = env.step(a)
        #update arrays
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
def train_one_episode(states, actions, rewards, model, optimizer, gamma=0.99, entropy_coef=0.01):
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)
    # convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int)
    rewards_to_go = np.array(rewards_to_go, dtype=np.float32)
    with tf.GradientTape() as tape:
        # get action probabilitoes from states
        logits, state_values = model(states)
        probs = tf.nn.softmax(logits, -1)
        log_probs = tf.nn.log_softmax(logits, -1)
        row_indices= tf.range(len(actions))
        indices = tf.transpose([row_indices, actions])
        log_probs_for_actions = tf.gather_nd(log_probs, indices)
        advantage = rewards_to_go - state_values
        # Compute loss to be minized
        J = tf.reduce_mean(log_probs_for_actions*advantage)
        H = -tf.reduce_mean(tf.reduce_sum(probs*log_probs, -1))
        loss = -(J+entropy_coef*H)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return np.sum(rewards) #to show progress on training

# generate trained agent video
def generate_animation(env, model, save_dir):
    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda id: True, force=True, mode='evaluation')
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    generate_trajectory(env, model)
    
def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))

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