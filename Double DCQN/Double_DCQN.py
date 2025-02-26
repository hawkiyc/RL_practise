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
from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import v2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        print("MPS not available because the current PyTorch install was NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")
    torch.mps.manual_seed(seed)
print(device)

#%%
'Building the Model'
class Network(nn.Module):
    def __init__(self, action_size, input_size: tuple, seed=seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        
        fc_in = self.get_fc_size(input_size)
        self.fc1 = nn.Linear(fc_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
    
    def get_fc_size(self, size):
        in_ = Variable(torch.ones(2, 3, size[0], size[1]))
        m_ = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        out_ = m_(in_)
        out_ = nn.Flatten()(out_)
        return out_.data.size(1)
    
    def forward(self, state):
        x = F.relu(self.batchnorm1(self.conv1(state)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%%
'Setting Environment and Hyperparameters'
env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = (128, 128)  # 圖片尺寸
action_size = env.action_space.n

lr = 5e-4
batch_size = 64
gamma = 0.99  # 折扣因子
replay_buffer_size = int(1e5)
tau = 1e-3    # 軟更新參數

#%%
'Preprocess Function'
def preprocess_frame(frame):
    """
    將原始 frame 轉換成指定尺寸並轉成 tensor
    """
    frame = Image.fromarray(frame)
    preprocess = v2.Compose([v2.Resize(state_shape), v2.ToTensor()])
    # 返回 shape 為 (1, C, H, W)
    return preprocess(frame).unsqueeze(0)

#%%
'Replay Buffer'
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=42):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (np.stack(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.stack(next_states),
                np.array(dones, dtype=np.uint8).reshape(-1, 1))
    
    def __len__(self):
        return len(self.memory)

#%%
'Training AI-Agent with Double DQN'
class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        self.local_qnetwork = Network(action_size, state_shape).to(device)
        self.target_qnetwork = Network(action_size, state_shape).to(device)
        self.opt = optim.Adam(self.local_qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer(replay_buffer_size, batch_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        # 預處理後並將 tensor 轉為 numpy 陣列（去掉 batch 維度）
        state_np = preprocess_frame(state).squeeze(0).cpu().numpy()
        next_state_np = preprocess_frame(next_state).squeeze(0).cpu().numpy()
        self.memory.add(state_np, action, reward, next_state_np, done)
        
        if len(self.memory) >= batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, gamma)
    
    'epsilon-greedy'
    def act(self, state, epsilon=0.):
        state_tensor = preprocess_frame(state).to(device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state_tensor)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        
        # ----- Double DQN 更新 -----
        with torch.no_grad():
            # 使用 local network 選擇最佳動作
            next_actions = self.local_qnetwork(next_states).argmax(dim=1, keepdim=True)
            # 使用 target network 評估該動作
            next_q_target = self.target_qnetwork(next_states).gather(1, next_actions)
        q_targets = rewards + (gamma * next_q_target * (1 - dones))
        # ---------------------------
        
        q_expected = self.local_qnetwork(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q_expected, q_targets)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.soft_update(self.local_qnetwork, self.target_qnetwork, tau)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

agent = Agent(action_size)

number_episodes = 2000
max_steps_per_episode = 10000
'設定 epsilon-greedy 參數'
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_rate = 0.99
epsilon = epsilon_starting_value
batched_scores = deque(maxlen=100)

best_score = -np.inf  # 記錄目前最高的平均分數

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    
    for t in range(max_steps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    batched_scores.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_rate * epsilon)
    
    current_avg = np.mean(batched_scores)
    print(f"\rEpisode: {episode}\tAverage Score: {current_avg:.3f}", end="")
    if episode % 100 == 0:
        print(f"\rEpisode: {episode}\tAverage Score: {current_avg:.3f}")
    
    # 若有新的最佳平均分數則儲存模型
    if current_avg > best_score:
        best_score = current_avg
        torch.save(agent.local_qnetwork.state_dict(), 'model_best.pth')
        print(f"\nNew best model saved at episode {episode} with average score: {best_score:.3f}")
    
    # 可依需求設定環境解決條件
    if current_avg >= 800.0:
        print(f'\nEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {current_avg:.3f}')
        break

# 儲存最終模型
torch.save(agent.local_qnetwork.state_dict(), 'model_final.pth')
print("\nFinal model saved.")

#%%
'Visualizing the Results'
agent.local_qnetwork.load_state_dict(torch.load('model_best.pth'))

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
