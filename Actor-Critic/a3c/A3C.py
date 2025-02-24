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
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mp.set_sharing_strategy('file_system')

#%% Set GPU and Seed
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
        state_value = self.fc_states(x).squeeze(-1)  # 回傳每個 batch 的標量值
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

# 建立一個全域環境用於後續示範（注意：訓練時每個 worker 會自己建立環境）
env = make_env()
state_shape = env.observation_space.shape
action_size = env.action_space.n
print('Action Meanings:', env.env.env.get_action_meanings())

lr = 1e-4
gamma = 0.99  # 折扣因子
max_updates = 8000  # 全域更新次數上限
n_steps = 5         # n-step TD 參數
eval_interval = 100 # 評估間隔（依全域更新次數計算）
n_workers = 16       # worker 數量，可依需求調整

#%% Agent (僅用於測試與視覺化，不參與訓練)
class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        self.network = Network(action_size, state_shape).to(device)
    
    def act(self, state):
        # 若 state 維度為 (channels, H, W) 則擴展成 batch
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_values, _ = self.network(state_tensor)
        policy = F.softmax(action_values, dim=-1)
        actions = [np.random.choice(self.action_size, p=p.cpu().detach().numpy()) for p in policy]
        return np.array(actions)

#%% Video Functions (用於展示結果)
def show_video_of_model(agent, env, video_filename='a3c_video.mp4'):
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

def show_video(video_filename='a3c_video.mp4'):
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        video = io.open(video_filename, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''<video alt="a3c video" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
             </video>'''))
    else:
        print("Could not find video")

#%% Worker Process 定義 (A3C 的訓練邏輯)
def worker(worker_id, global_model, optimizer, global_counter, max_updates, n_steps, gamma):
    # 每個 worker 自己建立一個 local model 並初始化為 global model 的參數
    local_model = Network(action_size, state_shape).to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = make_env()
    state, _ = env.reset()
    while True:
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        # 收集 n 步 rollout（若途中終止，則提前 break）
        for t in range(n_steps):
            rollout_states.append(state)
            # 根據 local_model 選擇動作
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=device)
            action_values, _ = local_model(state_tensor)
            policy = F.softmax(action_values, dim=-1)
            action = np.random.choice(action_size, p=policy.cpu().detach().numpy()[0])
            rollout_actions.append(action)
            next_state, reward, done, _, _ = env.step(action)
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            state = next_state
            if done:
                state, _ = env.reset()
                break

        # Bootstrap：若 rollout 最後一步未結束，則以 local_model 評估下一狀態價值；否則 R=0
        if not done:
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, next_value = local_model(state_tensor)
            R = next_value.cpu().numpy()[0]
        else:
            R = 0.0

        # 反向計算 n-step 回報
        returns = []
        for reward, done_flag in zip(reversed(rollout_rewards), reversed(rollout_dones)):
            R = reward + gamma * R * (1 - done_flag)
            returns.insert(0, R)

        # 若 rollout 資料數量為 0 則跳過更新
        if len(rollout_states) == 0:
            continue

        # 計算 loss
        states_batch = np.array(rollout_states)
        actions_batch = np.array(rollout_actions)
        returns_batch = np.array(returns)
        
        states_tensor = torch.tensor(states_batch, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions_batch, dtype=torch.int64, device=device)
        returns_tensor = torch.tensor(returns_batch, dtype=torch.float32, device=device)
        
        action_values, state_values = local_model(states_tensor)
        advantages = returns_tensor - state_values
        log_probs = F.log_softmax(action_values, dim=-1)
        probs = F.softmax(action_values, dim=-1)
        log_probs_actions = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        actor_loss = -(log_probs_actions * advantages.detach()).mean() - 0.001 * entropy
        critic_loss = F.mse_loss(state_values, returns_tensor)
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        # 梯度截斷（避免梯度爆炸）
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

        # 將 local model 的梯度更新至 global model（非同步更新）
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad
        optimizer.step()
        # 同步 local model 參數
        local_model.load_state_dict(global_model.state_dict())

        # 更新全域 counter，若達上限則離開
        with global_counter.get_lock():
            global_counter.value += 1
            if global_counter.value >= max_updates:
                break

#%% Evaluation Process 定義 (定期評估全域模型並儲存最佳模型)
def evaluate_global(global_model, global_counter, max_updates, global_best_reward, eval_interval):
    last_eval = 0
    while True:
        if global_counter.value >= max_updates:
            break
        if global_counter.value - last_eval >= eval_interval:
            last_eval = global_counter.value
            eval_rewards = []
            env_eval = make_env()
            # 進行 10 個測試 episode
            for _ in range(10):
                state, _ = env_eval.reset()
                total_reward = 0
                done = False
                while not done:
                    state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action_values, _ = global_model(state_tensor)
                    policy = F.softmax(action_values, dim=-1)
                    action = np.random.choice(action_size, p=policy.cpu().detach().numpy()[0])
                    state, reward, done, _, _ = env_eval.step(action)
                    total_reward += reward
                eval_rewards.append(total_reward)
            avg_reward = np.mean(eval_rewards)
            print(f"Evaluation at update {global_counter.value}: Average Reward: {avg_reward}")
            if avg_reward > global_best_reward.value:
                with global_best_reward.get_lock():
                    global_best_reward.value = avg_reward
                torch.save(global_model.state_dict(), "model_best.pt")
                print("Best model updated with reward:", avg_reward)
            env_eval.close()
        time.sleep(1)  # 每秒檢查一次

#%% Main Training (使用多進程進行 A3C 訓練)
if __name__ == '__main__':
    mp.set_start_method('spawn')
    global_model = Network(action_size, state_shape).to(device)
    global_model.share_memory()  # 使 global_model 可被多進程共享
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    global_counter = mp.Value('i', 0)
    global_best_reward = mp.Value('d', -np.inf)
    
    processes = []
    # Spawn worker processes
    for worker_id in range(n_workers):
        p = mp.Process(target=worker, args=(worker_id, global_model, optimizer, global_counter, max_updates, n_steps, gamma))
        p.start()
        processes.append(p)
    
    # Spawn evaluation process
    p_eval = mp.Process(target=evaluate_global, args=(global_model, global_counter, max_updates, global_best_reward, eval_interval))
    p_eval.start()
    processes.append(p_eval)
    
    for p in processes:
        p.join()
    
    # 儲存最終模型
    torch.save(global_model.state_dict(), "model_final.pt")
    print("Final model saved.")
    
    #%% Visualize Results with Best Model
    # 讀取最佳模型（model_best.pt）並展示影片
    agent = Agent(action_size)
    agent.network.load_state_dict(torch.load("model_best.pt", map_location=device))
    agent.network.eval()
    # 使用全域環境進行展示
    show_video_of_model(agent, env, video_filename='n_step_a3c_video_best_model.mp4')
    show_video('n_step_a3c_video_best_model.mp4')