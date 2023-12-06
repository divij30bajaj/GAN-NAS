import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from ddpg_utils import soft_update, hard_update
from ddpg_model import DeterministicPolicy, QNetworkDDPG
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE
from replay_memory import ReplayMemory
from torch.distributions.normal import Normal
import numpy as np

class DDPG(object):
    def __init__(self, num_inputs):

        self.gamma = 1
        self.tau = 0.005
        self.alpha = 0.1

        self.policy_type = "Deterministic"
        self.target_update_interval = 1
        # self.automatic_entropy_tuning = False
        self.hid_size = 128
        # self.target_entropy = -3
        self.device = torch.device("cuda")

        num_inputs = num_inputs

        tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE), len(SKIP_TYPE)]
        self.action_space = sum(tokens)

        self.critic = QNetworkDDPG(num_inputs, self.action_space, self.hid_size).cuda()
        self.critic_optim = Adam(self.critic.parameters(), lr=0.0003)

        self.critic_target = QNetworkDDPG(num_inputs, self.action_space, self.hid_size)
        hard_update(self.critic_target, self.critic)

        # self.alpha = 0
        # self.automatic_entropy_tuning = False
        self.policy = DeterministicPolicy(num_inputs, self.action_space, self.hid_size, self.action_space).cuda()
        self.policy_target = DeterministicPolicy(num_inputs, self.action_space, self.hid_size, self.action_space)
        hard_update(self.policy_target, self.policy)
        self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        action1, action2, action3, action4, action5, action6 = self.policy(state)
        actions = [action1, action2, action3, action4, action5, action6]

        noise_scale = 0.1
        action_limit = self.action_space
        clipped_actions = []
        for mu in actions:
            m = Normal(
                torch.zeros(mu.shape),
                torch.ones(mu.shape),
            )
            action = mu.cuda() + noise_scale * m.rsample().cuda()
            action = torch.clamp(action, -action_limit, action_limit)
            clipped_actions.append(np.transpose(action.detach().cpu().numpy()).squeeze(1))
        return clipped_actions

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).cuda()
        next_state_batch = torch.FloatTensor(next_state_batch).cuda()
        action_batch = torch.FloatTensor(action_batch).cuda()
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).cuda()
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).cuda()

        with torch.no_grad():
            next_state_action_1, next_state_action_2, next_state_action_3, next_state_action_4, next_state_action_5, next_state_action_6 = self.policy_target(next_state_batch)
            next_state_action = torch.cat((next_state_action_1, next_state_action_2, next_state_action_3,
                                           next_state_action_4, next_state_action_5, next_state_action_6), 1)
            qf_next_target = self.critic_target(next_state_batch, next_state_action)
            next_q_value = reward_batch + mask_batch * self.gamma * (qf_next_target)

        qf = self.critic(state_batch, action_batch)
        qf_loss = F.mse_loss(qf, next_q_value)

        pi_1, pi_2, pi_3, pi_4, pi_5, pi_6 = self.policy(state_batch)
        pi = torch.cat((pi_1, pi_2, pi_3, pi_4, pi_5, pi_6), 1)
        qf_pi = -1*torch.as_tensor(self.critic(state_batch, pi))

        policy_loss = qf_pi.mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))