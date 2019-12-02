import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Environment():
	def __init__(self):
		self.state_space = 2
		self.state = np.array([0,0])
		self.action_space = 4 # North-East-South-West

	def reset(self):
		self.state = np.array([0,0])
		return self.state

	def step(self, action):
		if action == 0:
			ds = np.array([0,1]) # north
		elif action == 1:
			ds = np.array([1,0]) # east
		elif action == 2:
			ds = np.array([0,-1]) # south
		elif action == 3:
			ds = np.array([-1,0]) # west
		else:
			error("invalid action")

		self.state = self.state + ds
		reward = action # scales with action number
		return self.state, reward

# initialize environment
env = Environment()
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		state_space = env.state_space
		action_space = env.action_space
		num_hidden = 100

		self.l1 = nn.Linear(state_space, num_hidden, bias=False)
		self.l2 = nn.Linear(num_hidden, action_space, bias=False)

		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []
		self.reset()

	def reset(self):
		# Episode policy and reward history
		self.episode_actions = torch.Tensor([])
		self.episode_rewards = []

	def forward(self, x):
		model = torch.nn.Sequential(
			self.l1,
			nn.Dropout(p=0.5),
			nn.Tanh(),
			self.l2,
			nn.Softmax(dim=-1)
		)
		return model(x)

def predict(state):
	# Select an action by running policy model
	# and choosing based on the probabilities in state
	state = torch.from_numpy(state).type(torch.FloatTensor)
	action_probs = policy(state)
	distribution = Categorical(action_probs)
	action = distribution.sample()

	# Add log probability of our chosen action to our history
	policy.episode_actions = torch.cat([
		policy.episode_actions,
		distribution.log_prob(action).reshape(1)
	])

	return action


def update_policy():
	R = 0
	rewards = []

	# Discount future rewards back to the present using gamma
	for r in policy.episode_rewards[::-1]:
		R = r + gamma * R
		rewards.insert(0, R)

	# Scale rewards
	rewards = torch.FloatTensor(rewards)
	rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

	# Calculate loss
	loss = (torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1))

	# Update network weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# Save and intialize episode history counters
	policy.loss_history.append(loss.item())
	policy.reward_history.append(np.sum(policy.episode_rewards))
	policy.reset()


def train(episodes):
	scores = []
	for episode in range(episodes):
		state = env.reset()
		for time in range(1000):
			action = predict(state)
			# Step through environment using chosen action
			state, reward = env.step(action.item())

			# Save reward
			policy.episode_rewards.append(reward)

		update_policy()

		# Calculate score to determine when the environment has been solved
		scores.append(reward)
		mean_score = np.mean(scores[-100:])

		if episode % 10 == 0:
			print('Episode {}\tAverage score (last 100 episodes): {:.2f}'.format(
				episode, mean_score))

# training
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
train(episodes=1000)

# # plotting
# plt.plot(policy.reward_history)
# plt.show()