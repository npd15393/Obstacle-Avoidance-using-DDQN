import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        def conv2d_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size(conv2d_size(conv2d_size(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size(conv2d_size(conv2d_size(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, input_shape, num_actions, device):
        self.device = device
        self.num_actions = num_actions
        self.online_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (1.0 - self.epsilon_min) / 100000
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                q_values = self.online_net(state_t)
            action = int(q_values.argmax().item())
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.steps_done += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states_t = torch.from_numpy(states).to(self.device).float()
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device).float()
        next_states_t = torch.from_numpy(next_states).to(self.device).float()
        dones_t = torch.from_numpy(dones).to(self.device).float()

        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)
        next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
        next_q_values = (
            self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
        )
        q_targets = rewards_t + (1 - dones_t) * self.gamma * next_q_values
        loss = F.smooth_l1_loss(q_values, q_targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_models(self, path_prefix):
        torch.save(self.online_net.state_dict(), path_prefix + "_online.pth")
        torch.save(self.target_net.state_dict(), path_prefix + "_target.pth")


# --- Preprocessing ---
def preprocess_depth(depth_img):
    depth_clipped = np.maximum(depth_img, 1e-3)
    depth_inverted = 255.0 / depth_clipped
    depth_inverted = np.clip(depth_inverted, 0, 255).astype(np.uint8)
    img = Image.fromarray(depth_inverted)
    img = img.resize((84, 84)).convert("L")
    return np.array(img, dtype=np.uint8)


# --- Main Training Loop ---
if __name__ == "__main__":
    # Replace this with the actual environment
    import gym

    # env = gym.make("AerialGymQuadNavigation-v0")
    env = ...  # TODO: Define your aerial_gym environment here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )
    input_shape = (4, 84, 84)

    agent = DQNAgent(input_shape, num_actions, device)

    n_episodes = 1000
    max_steps_per_episode = 1000
    episode_rewards = []
    os.makedirs("savedRewards", exist_ok=True)
    os.makedirs("trainedModels", exist_ok=True)

    for episode in range(n_episodes):
        obs = env.reset()
        state = (
            preprocess_depth(obs["depth"][0, 0])
            if isinstance(obs, dict)
            else preprocess_depth(obs)
        )
        state_stack = np.stack([state] * 4, axis=0)
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state_stack)
            result = env.step(action)
            next_obs, reward, done = result[0], result[1], result[2]
            next_frame = (
                preprocess_depth(next_obs["depth"][0, 0])
                if isinstance(next_obs, dict)
                else preprocess_depth(next_obs)
            )
            next_state_stack = np.vstack(
                (state_stack[1:, :, :], np.expand_dims(next_frame, 0))
            )
            agent.remember(state_stack, action, reward, next_state_stack, done)
            agent.learn()
            total_reward += reward
            state_stack = next_state_stack
            if done:
                break

        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            agent.update_target_network()

        if episode % 10 == 0:
            agent.save_models("trainedModels/dqn")
            plt.figure()
            plt.title(f"Episode {episode} Reward")
            plt.plot(range(len(episode_rewards[-10:])), episode_rewards[-10:])
            plt.savefig(f"savedRewards/reward_{episode}.png")
            plt.close()
            plt.figure()
            plt.title("Average Reward")
            plt.plot(range(len(episode_rewards)), episode_rewards)
            plt.savefig("savedRewards/avg_reward.png")
            plt.close()

        print(f"Episode {episode}: Reward = {total_reward}")

    print("Training completed.")
