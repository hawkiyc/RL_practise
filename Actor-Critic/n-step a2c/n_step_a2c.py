#%% Import Libraries
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%% Set GPU
seed = 42
use_cpu = False

if use_cpu:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed(seed)
else:
    device = torch.device('cpu')
print("Device:", device)

#%% Building the Model
class Network(nn.Module):
    def __init__(self, action_size, input_size: tuple, seed=seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        
        fc_in = self.get_fc_size(input_size)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc_states = nn.Linear(128, 1)
        self.fc_actions = nn.Linear(128, action_size)
    
    def get_fc_size(self, size):
        with torch.no_grad():
            dummy = torch.ones(2, size[0], size[1], size[2])
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, start_dim=1)
            return x.size(1)
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        action_value = self.fc_actions(x)
        state_value = self.fc_states(x).squeeze(-1)  # 修正：回傳每個 batch 的標量值
        return action_value, state_value

#%% Preprocess Atari Environment
class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, crop=lambda img: img, 
                 dim_order='pytorch', color=False, n_frames=4):
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
        self.frames = np.zeros(obs_shape, dtype=np.float32)
    
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
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
            self.frames[-3:] = img
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
            self.frames[-1] = img
        return self.frames
    
    def update_buffer(self, obs):
        self.frames = self.observation(obs)

def make_env():
    env = gym.make("KungFuMasterDeterministic-v0", render_mode='rgb_array')
    env = PreprocessAtari(
        env, height=64, width=64, crop=lambda img: img,
        dim_order='pytorch', color=False, n_frames=8)
    return env

env = make_env()
state_shape = env.observation_space.shape
action_size = env.action_space.n
print('Action Meanings:', env.env.env.get_action_meanings())

lr = 1e-4
gamma = 0.99  # 折扣因子
number_envs = 35

#%% Agent with A2C and n-step TD Update
class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        self.network = Network(action_size, state_shape).to(device)
        self.opt = optim.Adam(self.network.parameters(), lr=lr)
    
    def act(self, state):
        # 若 state 維度為 (channels, H, W) 則擴展成 batch
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_values, _ = self.network(state_tensor)
        policy = F.softmax(action_values, dim=-1)
        actions = [np.random.choice(self.action_size, p=p.cpu().detach().numpy()) for p in policy]
        return np.array(actions)
    
    def update(self, states, actions, returns):
        """
        更新網路參數，此處使用 n-step TD 的回報計算：
        returns 為每個狀態的 n-step 回報
        """
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        action_values, state_values = self.network(states)
        advantages = returns - state_values
        log_probs = F.log_softmax(action_values, dim=-1)
        probs = F.softmax(action_values, dim=-1)
        log_probs_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        actor_loss = -(log_probs_actions * advantages.detach()).mean() - 0.001 * entropy
        critic_loss = F.mse_loss(state_values, returns)
        total_loss = actor_loss + critic_loss
        
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

#%% Environment Batch for Parallel Environments
class EnvBatch:
    def __init__(self, n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]
    
    def reset(self):
        states = []
        for env in self.envs:
            state, _ = env.reset()
            states.append(state)
        return np.array(states)
    
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        next_states, rewards, dones, infos, _ = map(np.array, zip(*results))
        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i], _ = self.envs[i].reset()
        return next_states, rewards, dones, infos

env_batched = EnvBatch(number_envs)
batched_states = env_batched.reset()

agent = Agent(action_size)

#%% Training with n-step TD
n_steps = 5         # n-step TD 參數
num_updates = 3000  # 總更新次數（每次更新包含 n_steps 個 time step）
eval_interval = 150 # 每 eval_interval 次更新後進行評估
best_reward = -np.inf

for update in tqdm.trange(num_updates):
    rollout_states = []
    rollout_actions = []
    rollout_rewards = []
    rollout_dones = []
    
    # 收集 n 步的 rollout
    for t in range(n_steps):
        rollout_states.append(batched_states.copy())  # shape: (number_envs, ...)
        actions = agent.act(batched_states)            # (number_envs,)
        rollout_actions.append(actions.copy())
        next_states, rewards, dones, _ = env_batched.step(actions)
        rollout_rewards.append(rewards.copy())
        rollout_dones.append(dones.copy())
        batched_states = next_states.copy()
    
    # 取得 n 步後的狀態值（作為最後的 bootstrap）
    batched_states_tensor = torch.tensor(batched_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, next_values = agent.network(batched_states_tensor)
    next_values = next_values.cpu().numpy()  # shape: (number_envs,)
    
    # 以反向遞迴方式計算每個 time step 的 n-step 回報
    returns = np.zeros((n_steps, number_envs))
    R = next_values
    for t in reversed(range(n_steps)):
        r = rollout_rewards[t]
        d = rollout_dones[t].astype(np.float32)
        R = r + gamma * R * (1 - d)
        returns[t] = R
    
    # 將 rollout 中每個 time step 當作一筆訓練資料
    states_batch = np.concatenate(rollout_states, axis=0)   # shape: (n_steps*number_envs, channels, H, W)
    actions_batch = np.concatenate(rollout_actions, axis=0)   # shape: (n_steps*number_envs,)
    returns_batch = returns.reshape(-1)                       # shape: (n_steps*number_envs,)
    
    # 用收集到的 rollout 更新網路
    agent.update(states_batch, actions_batch, returns_batch)
    
    # 每隔一段更新次數進行評估，並儲存最佳模型
    if update % eval_interval == 0:
        eval_rewards = []
        for _ in range(10):
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state)[0]
                state, reward, done, _, _ = env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
        avg_reward = np.mean(eval_rewards)
        print(f"Update {update}: Average Reward: {avg_reward}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.network.state_dict(), "model_best.pt")
            print("Best model saved with reward:", best_reward)

# 儲存最終模型
torch.save(agent.network.state_dict(), "model_final.pt")
print("Final model saved.")

#%% Visualizing the Results and Saving Video
def show_video_of_model(agent, env, video_filename='a2c_video.mp4'):
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)[0]
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"Video saved as {video_filename}")

def show_video(video_filename='a2c_video.mp4'):
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        video = io.open(video_filename, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''<video alt="a2c video" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
             </video>'''))
    else:
        print("Could not find video")

# 讀取最佳模型（若要讀取最終模型，請將 'model_best.pt' 換成 'model_final.pt'），並展示遊戲結果
agent.network.load_state_dict(torch.load("model_best.pt", map_location=device))
agent.network.eval()
show_video_of_model(agent, env, video_filename='n_step_a2c_video_best_model.mp4')
show_video('n_step_a2c_video_best_model.mp4')


