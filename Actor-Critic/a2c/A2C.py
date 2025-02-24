#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:44:43 2024

@author: hawkiyc
"""

#%%
'Import Libraries'

import base64
import cv2
import glob
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import imageio
import io
from IPython.display import HTML, display
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
'Set GPU'

seed = 42
use_cpu = False

if use_cpu:
    device = torch.device('cpu')

elif torch.cuda.is_available(): 
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()

elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
    torch.mps.manual_seed(seed)
print(device)

#%%
'Building the Model'

class Network(nn.Module):
    
    def __init__(self, action_size, input_size: tuple, seed = seed):
        
        super(Network, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(8, 32, kernel_size = 3, stride = 2)
        # self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2)
        # self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2)
        # self.batchnorm3 = nn.BatchNorm2d(32)
        
        fc_in = self.get_fc_size(input_size)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc_states = nn.Linear(128, 1)
        self.fc_actions = nn.Linear(128, action_size)
        
    
    def get_fc_size(self, size):
        
        in_ = Variable(torch.ones(2, size[0], size[1], size[2]))
        m_ = nn.Sequential(self.conv1, self.conv2, self.conv3, )
        out_ = m_(in_)
        out_ = nn.Flatten()(out_)
        
        return out_.size(1)
    
    def forward(self, state):
        
        # x = F.relu(self.batchnorm1(self.conv1(state)))
        # x = F.relu(self.batchnorm2(self.conv2(x)))
        # x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        
        action_value = self.fc_actions(x)
        state_value = self.fc_states(x)[0]
        
        return action_value, state_value

#%%
'Trainging AI-Agent'

class PreprocessAtari(ObservationWrapper):

    def __init__(self, env, height = 42, width = 42, crop = lambda img: img, 
                 dim_order = 'pytorch', color = False, n_frames = 4):
        
        super(PreprocessAtari, self).__init__(env)
        
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 
                     'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frames = np.zeros(obs_shape, dtype = np.float32)

    def reset(self):
        
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        
        return self.frames, info

    def observation(self, img):
        
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        
        if not self.color:
            
            if len(img.shape) == 3 and img.shape[2] == 3:
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype('float32') / 255.
        
        if self.color: # shift oldest img to the end
            self.frames = np.roll(self.frames, shift = -3, axis = 0)
        else:
            self.frames = np.roll(self.frames, shift = -1, axis = 0)
        
        if self.color: # replace the oldest img with latest one
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        
        return self.frames
    
    def update_buffer(self, obs):
        
        self.frames = self.observation(obs)

def make_env():
    
    env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
    env = PreprocessAtari(
        env, height = 64, width = 64, crop = lambda img: img, 
        dim_order = 'pytorch', color = False, n_frames = 8)
    
    return env

env = make_env()
state_shape = env.observation_space.shape
action_size = env.action_space.n
print('Action Name:', env.env.env.get_action_meanings())

lr = 1e-4
gamma = .99 # discount_factor
number_envs = 35

class Agent():
    
    def __init__(self, action_size,):
        
        self.action_size = action_size
        self.network = Network(action_size, state_shape).to(device)
        self.opt = optim.Adam(self.network.parameters(), lr = lr)
    
    'softmax'
    def act(self, state,):
        
        if state.ndim == 3:
            state = [state]
        
        state = torch.tensor(state, dtype = torch.float32, device = device)
        action_value, _ = self.network(state)
        policy = F.softmax(action_value, dim = -1)
        return np.array(
            [np.random.choice(len(p), p = p
                              ) for p in policy.detach().cpu().numpy()])
    
    def step(self, state, action, reward, next_state, done):
        
        batch_size = state.shape[0]
        state = torch.tensor(state, dtype = torch.float32, device = device)
        next_state = torch.tensor(
            next_state, dtype = torch.float32, device = device)
        reward = torch.tensor(reward, dtype = torch.float32, device = device)
        done = torch.tensor(
            done, dtype = torch.bool, device = device).to(
                dtype = torch.float32)
        action_values, state_values = self.network(state)
        _, next_state_values = self.network(next_state)
        target_state_value = reward + gamma * next_state_values * (1 - done)
        advantage = target_state_value - state_values
        probs = F.softmax(action_values, dim = -1)
        log_probs = F.log_softmax(action_values, dim = -1)
        entropy = -torch.sum(probs * log_probs, axis = -1)
        batch_idx = np.arange(batch_size)
        log_probs_action = log_probs[batch_idx, action]
        acttion_loss = -(log_probs_action * advantage.detach()
                         ).mean() - entropy.mean() * .001
        critic_loss = F.mse_loss(target_state_value.detach(), state_values)
        total_loss = acttion_loss + critic_loss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

agent = Agent(action_size)

def evaluate(agent, env, n_episodes = 1):
    
    episodes_rewards = []
    
    for _ in range(n_episodes):
        
        state, _ = env.reset()
        total_reward = 0 
        
        while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action[0])
            total_reward += reward
            if done:
                break
        episodes_rewards.append(total_reward)
    
    return episodes_rewards

class EnvBatch:
    
    def __init__(self, n_envs = 10):
        
        self.envs = [make_env() for i in range(n_envs)]
    
    def reset(self,):
        
        _states = []
        for env in self.envs:
            _states.append(env.reset()[0])
        
        return np.array(_states)
    
    def step(self, actions):
        
        next_states, rewards, dones, infos, _ = map(np.array, zip(*[
            env.step(a) for env, a in zip(self.envs, actions)]))
        
        for i in range (len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]
        
        return next_states, rewards, dones, infos

env_batched = EnvBatch(number_envs)
batched_states = env_batched.reset()

with tqdm.trange(0, 3001) as progress_bar:
    for i in progress_bar:
        batched_actions = agent.act(batched_states)
        batched_next_states, batched_rewards, batched_dones, _ = \
            env_batched.step(batched_actions)
        batched_rewards *= .01
        agent.step(batched_states, batched_actions, batched_rewards, 
                   batched_next_states, batched_dones)
        batched_states = batched_next_states
        if i % 1000 == 0:
            print("Average Rewards:", 
                  np.mean(evaluate(agent, env, n_episodes = 10)))

#%%
'Visualizing the Results'

def show_video_of_model(agent, env):
    
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
    
    env.close()
    imageio.mimsave('a2c_video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    
    mp4list = glob.glob('*.mp4')
    
    if len(mp4list) > 0:
        
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
