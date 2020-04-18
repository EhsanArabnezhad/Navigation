import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as f
import torch.optim as optim

Buf_size = int(1e5)
Batch_size = 64
Gamma = 0.999
Lr = 5e-4
Tau = 1e-3
Update_every = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed, duel=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.duel = duel

        self.qnetwork_local = QNetwork(state_size, action_size, seed, duel).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, duel).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=Lr)

        self.memory = ReplayBuffer(action_size, Buf_size, Batch_size, seed)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % Update_every
        if self.time_step == 0:
            if len(self.memory) > Batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, Gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, all_done = experiences
        best_next_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        next_q_targets = self.qnetwork_target(next_states).detach().gather(1, best_next_action)

        q_targets = rewards + (gamma * next_q_targets * (1 - all_done))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = f.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, Tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param)


class ReplayBuffer:
    def __init__(self, action_size, buf_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buf_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None]))\
            .float().to(device)
        all_done = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8))\
            .float().to(device)

        return states, actions, rewards, next_states, all_done

    def __len__(self):
        return len(self.memory)
