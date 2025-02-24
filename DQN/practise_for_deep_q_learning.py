#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:44:43 2024

@author: hawkiyc
"""

#%%
'Import Libraries'

import base64
from collections import deque
import glob
import gymnasium as gym
import imageio
import io
from IPython.display import HTML, display
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
'Set GPU'

seed = 42
use_cpu = True

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
    
    def __init__(self, state_size, action_size, seed = seed):
        
        super(Network, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x

#%%
'Trainging AI-Agent'

env = gym.make("LunarLander-v2")
state_shape = env.observation_space.shape
state_size = state_shape[0]
action_size = env.action_space.n

lr = 5e-4
batch_size = 100
gamma = .99 # discount_factor
replay_buffer_size = int(1e5)
tau = 1e-3 # interpolation_parameter

class ReplayMemory(object):
    
    def __init__(self, capacity):
        
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        
        experiences = random.sample(self.memory, k = batch_size)
        states = torch.from_numpy(
            np.vstack([e[0] for e in experiences if e is not None]
                      )).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None]
                      )).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None]
                      )).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e[3] for e in experiences if e is not None]
                      )).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None]
                      ).astype(np.uint8)).float().to(device)
        
        return states, next_states, actions, rewards, dones

class Agent():
    
    def __init__(self, state_size, action_size,):
        
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(device)
        self.target_qnetwork = Network(state_size, action_size).to(device)
        self.opt = optim.Adam(self.local_qnetwork.parameters(), lr = lr)
        self.memory = ReplayMemory(capacity = replay_buffer_size)
        self.time_step = 0
    
    def step(self, states, actions, rewards, next_states, dones):
        
        self.memory.push((states, actions, rewards, next_states, dones))
        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0:
            if len(self.memory.memory) > batch_size:
                exp = self.memory.sample(batch_size)
                self.learn(exp, gamma)
    
    'epsilon_greedy'
    def act(self, state, epsilon = 0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    'softmax'
    'from 8Gitbrix https://github.com/8Gitbrix/Reinforcement-Learning'
    # def act(self, state,):
        
    #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    #     self.local_qnetwork.eval()
    #     with torch.no_grad():
    #         action_values = self.local_qnetwork(state).cpu().data.numpy()
    #         action_values = action_values[0]
    #         action_values -= np.max(action_values)
    #     self.local_qnetwork.train()
    #     prob_actions = np.exp(action_values) / np.sum(np.exp(action_values))
    #     cumulative_probability = 0.0
    #     choice = random.uniform(0,1)
    #     for a,pr in enumerate(prob_actions):
    #         cumulative_probability += pr
    #         if cumulative_probability >= choice:
    #             return a
    
    def learn(self, exp, gamma):
        
        states, next_states, actions, rewards, dones = exp
        next_q_target = self.target_qnetwork(
            next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_target * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, tau)
    
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()):
            
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

agent = Agent(state_size, action_size)

number_episodes = 2000
max_steps_per_episode = 1000
'Setting epsilon if you are using epsilon_greedy'
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_rate = .99
epsilon = epsilon_starting_value
batched_scores = deque(maxlen = batch_size)

for episode in range(1, number_episodes + 1):
    
    state, _ = env.reset()
    score = 0
    
    for t in range(max_steps_per_episode):
        
        'epsilon_greedy'
        action = agent.act(state, epsilon) 
        'softmax'
        # action = agent.act(state, )
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    batched_scores.append(score)
    
    'epsilon_greedy only'
    epsilon = max(epsilon_ending_value, epsilon_decay_rate * epsilon)
    
    print(
        f"\rEpisode: {episode}\tAverage Score: {np.mean(batched_scores):.3f}",
        end = "")
    if episode % 100 == 0:
        print(
            f"\rEpisode: {episode}\tAverage Score: {np.mean(batched_scores):.3f}",)
    
    if np.mean(batched_scores) >= 200.0:
        print(
            f'\nEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {np.mean(batched_scores):.3f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

#%%
'Visualizing the Results'

agent = Agent(state_size, action_size)
agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))

def show_video_of_model(agent, env_name):
    
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

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

# %%
