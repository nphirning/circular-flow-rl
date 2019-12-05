# Reinforcement Learning Utilities

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from constants import *

# Pseudocode from here: https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        super(PolicyNetwork, self).__init__()
        # Should be (2 * number of firms) + (number of people)
        self.state_dim = state_dim 
        # Should be (choices of hours/goods produced) * (choices of prices) * (demand curve parameters)
        self.action_dim = action_dim

        self.learning_rate = learning_rate

        hidden_dim = int(np.sqrt(state_dim * action_dim))
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        model = torch.nn.Sequential(
            self.layer1,
            nn.Tanh(),
            self.layer2
        )
        return F.log_softmax(model(x), dim=0)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        super(QNetwork, self).__init__()
        # Should be (2 * number of firms) + (number of people)
        self.state_dim = state_dim 
        # Should be (choices of hours/goods produced) * (choices of prices) * (demand curve parameters)
        self.action_dim = action_dim

        self.learning_rate = learning_rate

        hidden_dim = int(np.sqrt(state_dim * action_dim))
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        model = torch.nn.Sequential(
            self.layer1,
            nn.Tanh(),
            self.layer2
        )
        return model(x)

class ActorCritic(object):
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.q_net = QNetwork(state_dim, action_dim)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_net.learning_rate)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.q_net.learning_rate)

        self.gamma = DISCOUNT

        self.curr_state = None
        self.curr_action = None 
        self.curr_action_log_prob = None
        self.curr_reward = None
        self.next_state = None
        self.next_action = None
        self.next_action_log_prob = None
        self.next_reward = None


    def choose_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        self.next_state = state 
        log_probs = self.policy_net(state)
        probs = torch.exp(log_probs)
        c = Categorical(probs)
        self.next_action = c.sample()
        self.next_action_log_prob = c.log_prob(self.next_action).reshape(1)
        return self.next_action

    def update_policy(self):
        if type(self.curr_state) == type(None):
            return
        q_value = torch.index_select(self.q_net(self.curr_state), 0, self.curr_action)
        policy_loss = torch.mul(self.curr_action_log_prob, q_value).mul(-1)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        # TODO: do appropriate storing here

    def update_q(self):
        if type(self.curr_state) == type(None):
            return
        curr_q_value = torch.index_select(self.q_net(self.curr_state), 0, self.curr_action)
        next_q_value = torch.index_select(self.q_net(self.next_state), 0, self.next_action)
        td_error = self.curr_reward + next_q_value.mul(self.gamma) - curr_q_value
        q_loss = torch.mul(td_error, curr_q_value).mul(-1)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # TODO: do appropriate storing here


    def shift_results(self):
        self.curr_state = self.next_state
        self.curr_action = self.next_action
        self.curr_action_log_prob = self.next_action_log_prob
        self.curr_reward = self.next_reward
        self.next_state = None
        self.next_action = None
        self.next_action_log_prob = None
        self.next_reward = None

    def record_reward(self, reward):
        self.next_reward = reward

    def reset_memory(self):
        self.curr_state = None
        self.curr_action = None
        self.curr_action_log_prob = None
        self.curr_reward = None
        self.next_state = None
        self.next_action = None
        self.next_action_log_prob = None
        self.next_reward = None
        

    # order of events: choose_action, record_reward, update_policy, update_q, shift_results, sometimes reset_memory



