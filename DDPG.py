import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0.0001

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_size, action_size, hidden_size):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, action_size)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		
	def forward(self, state):
		x = self.fc1(state)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.tanh(x)
		return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.max_size = buffer_size
        self.ptr = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)

class DDPG:
	def __init__(self, state_size, action_size, hidden_size):
		self.actor = Actor(state_size, action_size, hidden_size).to(device)
		self.target_actor = copy.deepcopy(self.actor).to(device)
		self.critic = Critic(state_size, action_size, hidden_size).to(device)
		self.target_critic = copy.deepcopy(self.critic).to(device)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
		self.replay_buffer = ReplayBuffer()
		
	def act(self, state, epsilon=0):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		with torch.no_grad():
			
			action = self.actor(state).cpu().data.numpy()
		return action
	
	def train(self):
		if len(self.replay_buffer.buffer) < BATCH_SIZE:
			return
		
		state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
		state = torch.FloatTensor(state).to(device)
		action = torch.FloatTensor(action).to(device)
		reward = torch.FloatTensor(reward).to(device)
		next_state = torch.FloatTensor(next_state).to(device)
		done = torch.FloatTensor(done).to(device)
		
		# Update critic
		#print(action)
		Q = self.critic(state, action)
		next_action = self.target_actor(next_state)
		next_Q = self.target_critic(next_state, next_action.detach())
		target_Q = reward + GAMMA * next_Q * (1 - done)
		critic_loss = nn.MSELoss()(Q, target_Q.detach())
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		# Update actor
		actor_loss = -self.critic(state, self.actor(state)).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
		
		# Update target networks
		for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
			target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
		for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
			target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
		
	def update_replay_buffer(self, state, action, reward, next_state, done):
		self.replay_buffer.add(state, action, reward, next_state, done)
		
	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor.pth")
		torch.save(self.critic.state_dict(), filename + "_critic.pth")
		
	def load(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
		self.target_actor = copy.deepcopy(self.actor)
		self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
		self.target_critic = copy.deepcopy(self.critic)