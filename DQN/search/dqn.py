import torch
import torch.nn.functional as F
from torch.optim import Adam
from replay_memory import ReplayMemory
from dqn_model import DQN
import numpy as np
import os

class DQNAgent(object):
    def __init__(self, num_inputs, num_actions):
        self.gamma = 0.99
        self.epsilon = 0.1
        self.target_update_interval = 100
        self.device = torch.device("cuda")
        self.action_space = num_actions

        self.q_network = DQN(num_inputs, num_actions).cuda()
        self.target_q_network = DQN(num_inputs, num_actions).cuda()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = Adam(self.q_network.parameters(), lr=0.0003)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        probs = np.ones(self.action_space) * self.epsilon / self.action_space
        # if evaluate is False and torch.rand(1).item() < self.epsilon:
        #     return torch.randint(0, self.q_network.num_actions, (1,)).item()
        # else:
        #     with torch.no_grad():
        q_values = self.q_network(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] = 1 - self.epsilon + (self.epsilon / self.action_space)
        return probs


    def update_parameters(self, memory, batch_size, updates):
        if len(memory) < batch_size:
            return 0, 0, 0

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).cuda()
        next_state_batch = torch.FloatTensor(next_state_batch).cuda()
        action_batch = torch.LongTensor(action_batch).cuda()
        reward_batch = torch.FloatTensor(reward_batch).cuda()
        mask_batch = torch.FloatTensor(mask_batch).cuda()

        q_values = self.q_network(state_batch)
        q_values = torch.gather(
                q_values, dim=1, index=action_batch.unsqueeze(1).long()
        ).squeeze(-1)

        # with torch.no_grad():
        #     next_q_values = self.target_q_network(next_state_batch).max(1)[0].unsqueeze(1)
        #     target_q_values = reward_batch + mask_batch * self.gamma * next_q_values

        with torch.no_grad():
            if next_state_batch.ndim == 1:
                next_state_batch = torch.unsqueeze(next_state_batch, dim=0)

            next_q_values = self.target_q_network(next_state_batch)
            maxQ = torch.max(next_q_values, dim=1)
            target_q_values = reward_batch + self.gamma * mask_batch * maxQ[0]
            # next_q_values_target = self.q_target_network(next_state_batch).max(1, keepdim=True)[0]
            # target_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if updates % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/dqn_model_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(model_path))
        torch.save(self.q_network.state_dict(), model_path)

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        self.q_network.load_state_dict(torch.load(model_path))
