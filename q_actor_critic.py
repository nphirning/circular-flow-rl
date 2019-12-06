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


class AdvantageNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.005):
        super(AdvantageNetwork, self).__init__()
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
        self.adv_net = AdvantageNetwork(state_dim, 1)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_net.learning_rate)
        self.adv_optimizer = optim.Adam(self.adv_net.parameters(), lr=self.adv_net.learning_rate)

        self.gamma = DISCOUNT

        self.curr_state = None
        self.curr_action = None 
        self.curr_action_log_prob = None
        self.curr_reward = None
        self.next_state = None
        self.next_action = None
        self.next_action_log_prob = None
        self.next_reward = None

        self.policy_loss_hist = []
        self.adv_loss_hist = []


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
        #adv_value = torch.index_select(self.adv_net(self.curr_state), 0, self.curr_action)
        adv_value = torch.sum(self.adv_net(self.curr_state))
        policy_loss = torch.mul(self.curr_action_log_prob, adv_value).mul(-1)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_loss_hist.append(policy_loss.item())
        # TODO: do appropriate storing here

    def update_q(self):
        if type(self.curr_state) == type(None):
            return



        # curr_adv_value = torch.index_select(self.adv_net(self.curr_state), 0, self.curr_action)
        # next_adv_value = torch.index_select(self.adv_net(self.next_state), 0, self.next_action)
        curr_adv_value = torch.sum(self.adv_net(self.curr_state))
        next_adv_value = torch.sum(self.adv_net(self.next_state))
        td_error = self.curr_reward + next_adv_value.mul(self.gamma) - curr_adv_value
        adv_loss = torch.mul(td_error, curr_adv_value).mul(-1)



        self.adv_optimizer.zero_grad()
        adv_loss.backward()
        self.adv_optimizer.step()

        self.adv_loss_hist.append(adv_loss.item())
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

        if len(self.policy_loss_hist) > 0:
            print('Current losses: {} {}'.format(self.policy_loss_hist[-1], self.adv_loss_hist[-1]))

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



