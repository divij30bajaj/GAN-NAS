import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetworkDDPG(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkDDPG, self).__init__()

        # Q1 architecture
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim).cuda()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.linear3 = nn.Linear(hidden_dim, 1).cuda()

        self.apply(weights_init_)

    def forward(self, state, action):
        ps = action.cuda()
        a1, a2, a3, a4, a5, a6 = ps[:, :2], ps[:, 2:2+3], ps[:, 2+3:2+3+3], ps[:, 2+3+3: 2+3+3+2], ps[:, 2+3+3+2:2+3+3+2+2], ps[:, 2+3+3+2+2:2+3+3+2+2+2]
        a1_ = a1 
        a2_ = a2 
        a3_ = a3 
        a4_ = a4
        a5_ = a5 
        a6_ = a6 
        a = torch.cat([a1_, a2_, a3_, a4_, a5_, a6_], dim=1)
        xu = torch.cat([state.cuda(), a], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, act_lim, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.act_lim = act_lim
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim).cuda()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).cuda()

        self.linear_branch_1 = nn.Linear(hidden_dim, len(CONV_TYPE)).cuda()
        self.linear_branch_2 = nn.Linear(hidden_dim, len(NORM_TYPE)).cuda()
        self.linear_branch_3 = nn.Linear(hidden_dim, len(UP_TYPE)).cuda()
        self.linear_branch_4 = nn.Linear(hidden_dim, len(SHORT_CUT_TYPE)).cuda()
        self.linear_branch_5 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()
        self.linear_branch_6 = nn.Linear(hidden_dim, len(SKIP_TYPE)).cuda()

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state.cuda()))
        x = F.relu(self.linear2(x))

        a1 = self.act_lim * F.tanh(self.linear_branch_1(x))
        a2 = self.act_lim * F.tanh(self.linear_branch_2(x))
        a3 = self.act_lim * F.tanh(self.linear_branch_3(x))
        a4 = self.act_lim * F.tanh(self.linear_branch_4(x))
        a5 = self.act_lim * F.tanh(self.linear_branch_5(x))
        a6 = self.act_lim * F.tanh(self.linear_branch_6(x))
        
        return a1, a2, a3, a4, a5, a6