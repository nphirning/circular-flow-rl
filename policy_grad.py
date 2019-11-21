# Reinforcement Learning Utilities

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from constants import *

# POLICY GRADIENT
class ReinforcePolicyGradient(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.Tanh, learning_rate=0.01):
        super(ReinforcePolicyGradient, self).__init__()
        # Should be (2 * number of firms) + (number of people)
        self.state_dim = state_dim 
        # Should be (choices of hours/goods produced) * (choices of prices) * (demand curve parameters)
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.activation = activation
        self.out_layer = nn.Linear(input_dim, action_dim)

        self.gamma = 
        # optimizer = optim.AdamOptimizer(self.policy.parameters(), 
        #     lr=self.learning_rate)

    def forward(self, x):
        x = self.activation(self.out_layer(x))
        return F.log_softmax(x)


    # function to compute loss
    # function to take in a new trajectory
    # function to update the parameters appropriately




