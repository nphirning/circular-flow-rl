# Reinforcement Learning Utilities

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from constants import *

# Heavily adopted from here: 
# https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
class ReinforcePolicyGradient(nn.Module):
    def __init__(self, state_dim, action_dim, activation=torch.tanh, learning_rate=0.01):
        super(ReinforcePolicyGradient, self).__init__()
        # Should be (2 * number of firms) + (number of people)
        self.state_dim = state_dim 
        # Should be (choices of hours/goods produced) * (choices of prices) * (demand curve parameters)
        self.action_dim = action_dim

        self.learning_rate = learning_rate

        self.activation = activation
        hidden_dim = 1000
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_dim)

        self.gamma = DISCOUNT

        self.epis_policy_hist = torch.Tensor([])
        self.epis_reward_hist = []
        self.reward_hist = []
        self.loss_hist = []

        self.optimizer = optim.Adam(self.parameters(), 
            lr=self.learning_rate)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.layer1,
            nn.Tanh(),
            self.layer2
        )
        return F.log_softmax(model(x), dim=0)
        # ll1 = self.layer1(x)
        # x = self.activation(self.layer2(x))
        # return F.log_softmax(x, dim=0)

    def choose_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        log_probs = self.__call__(state)
        probs = torch.exp(log_probs)
        c = Categorical(probs)
        action = c.sample()

        # Append action to the policy history
        if self.epis_policy_hist.dim() != 0:
            self.epis_policy_hist = torch.cat([self.epis_policy_hist, torch.Tensor([c.log_prob(action)])])
        else:
            self.epis_policy_hist = (c.log_prob(action))

        return action

    def update_policy(self):
        running_reward = 0
        rewards = []
        
        # Propagate future rewards to present
        for r in self.epis_reward_hist[::-1]:
            running_reward = r + self.gamma * running_reward
            rewards.insert(0, running_reward)
            
        # Normalization
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        # Loss calculation by reading through episode
        loss = Variable(torch.sum(torch.mul(self.epis_policy_hist, rewards).mul(-1), -1), requires_grad=True)
        
        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Save and reset episode histories
        self.loss_hist.append(loss.item())
        self.reward_hist.append(np.sum(self.epis_reward_hist))
        self.epis_policy_hist = torch.Tensor([])
        self.epis_reward_hist = []

    def record_reward(self, reward):
        self.epis_reward_hist.append(reward)


# function to pass in an episode


