# Reinforcement Learning Utilities

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from constants import *

# Heavily adoped from here: 
# https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
class ReinforcePolicyGradient(object):
    def __init__(self, state_dim, action_dim, activation=F.Tanh, learning_rate=0.01):
        super(ReinforcePolicyGradient, self).__init__()
        # Should be (2 * number of firms) + (number of people)
        self.state_dim = state_dim 
        # Should be (choices of hours/goods produced) * (choices of prices) * (demand curve parameters)
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.activation = activation
        self.out_layer = nn.Linear(input_dim, action_dim)

        self.gamma = DISCOUNT

        self.epis_policy_hist = Variable(torch.Tensor())
        self.epis_reward_hist = []
        self.reward_hist = []
        self.loss_hist = []

        self.optimizer = optim.AdamOptimizer(self.parameters(), 
            lr=self.learning_rate)

    def forward(self, x):
        x = self.activation(self.out_layer(x))
        return F.log_softmax(x)

    # THIS depends on the environment
    def construct_state(self):
        pass

    def deconstruct_action(self):
        pass

    def choose_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        log_probs = self.policy_net.forward(Variable(state))
        probs = torch.exp(log_probs)
        c = Categorical(probs)
        action = c.sample()

        # Append action to the policy history
        if self.epis_policy_hist.dim() != 0:
            self.epis_policy_hist = torch.cat([self.epis_policy_hist, c.log_prob(action)])
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
        loss = (torch.sum(torch.mul(self.epis_policy_hist, Variable(rewards)).mul(-1), -1))
        
        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Save and intialize episode histories
        self.loss_hist.append(loss.data[0])
        self.reward_hist.append(np.sum(self.epis_reward_hist))
        self.epis_policy_hist = Variable(torch.Tensor())
        self.epis_reward_hist = []

    # function to update epis_reward_hist and epis_policy_hist




# function to pass in an episode


