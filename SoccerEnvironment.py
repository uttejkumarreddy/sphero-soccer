import numpy as np
import gym
import torch
import mujoco as mj
from gym import spaces

from SoccerBall import SoccerBall
from SoccerPlayer import SoccerPlayer

class SoccerEnvironment(gym.Env):
	def __init__(self, model, players_per_team, randomize_player_positions = False):
		self.players_per_team = players_per_team
		self.randomize_player_positions = randomize_player_positions

		# Angle of rotation, Direction of movement
		self.action_space = spaces.Box(
			low=np.array([-np.pi, 0], dtype=np.float32),
			high=np.array([np.pi, 1], dtype=np.float32),
			dtype=np.float32
		)

		# x-coord, y-coord, x-vel, y-vel
		information_low_ball = np.array([-45, -30, -10, -10], dtype=np.float32) # TODO: Should increase to boundaries of pitch
		information_high_ball = np.array([45, 30, 10, 10], dtype=np.float32)

		# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
		information_low_agent = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32)
		information_high_agent = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

		number_of_agents = self.players_per_team * 2

		low = np.concatenate((np.tile(information_low_agent, number_of_agents), information_low_ball))
		high = np.concatenate((np.tile(information_high_agent, number_of_agents), information_high_ball))

		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.lines_touch = ["touch_line_outside_1", "touch_line_outside_2", "touch_line_goal_A_right", "touch_line_goal_A_left", "touch_line_goal_B_right", "touch_line_goal_B_left"]
		self.geom_id_lines_touch = {}
		for line_touch in self.lines_touch:
			self.geom_id_lines_touch[line_touch] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, line_touch)

		self.lines_goal = ["touch_line_goal_A", "touch_line_goal_B"]
		self.geom_id_lines_goal = {}
		for line_goal in self.lines_goal:
			self.geom_id_lines_goal[line_goal] = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, line_goal)

		self.prefix_Team_A = 'A_'
		self.prefix_Team_B = 'B_'
		
		self.player_names_Team_A = []
		self.player_names_Team_B = []

		for i in range(1, 1 + self.players_per_team):
			self.player_names_Team_A.append(self.prefix_Team_A + str(i))
			self.player_names_Team_B.append(self.prefix_Team_B + str(i))

		self.players_Team_A = []
		self.players_Team_B = []
		self.players = [] 
		
		self.ball = None

		self.time = 0
		self.players_geom_ids = []

	def reset(self, model, data):
		self.time = 0
		for player in self.players:
			player.reset(model, data)
		self.ball.reset(model, data)

	def update_player_geom_ids(self, player_geom_id):
		self.players_geom_ids.append(player_geom_id)

	def initialize_players_and_ball(self, model, data):
		for name in self.player_names_Team_A:
			self.players_Team_A.append(SoccerPlayer(model, data, name, 'A', self))

		for name in self.player_names_Team_B:
			self.players_Team_B.append(SoccerPlayer(model, data, name, 'B', self))

		self.players = self.players_Team_A + self.players_Team_B
		self.ball = SoccerBall(model, data, 'ball')

		# Update player geom ids in environment to detect collisions
		for player in self.players:
			self.update_player_geom_ids(player.id_geom)

	def generate_state_space(self, model, data):
		state_space = []
		for player in self.players:
			state_space += player.get_state(model, data)
		state_space += self.ball.get_state(model, data)
		state_space = torch.from_numpy(np.array(state_space)).float().unsqueeze(0).to(self.device)
		return state_space

	def generate_actions(self, model, data, state_space):
		actions = []
		for player in self.players:
			with torch.no_grad():
				action = player.brain(state_space)[0]
				actions.append(action)
		return actions

	def generate_rewards(self, model, data):
		self.ball.update_state(model, data, self)
		
	def perform_action(self, model, data, player, action):
		angle, speed = action

		new_direction = player.rotate(model, data, angle)
		new_direction = np.array(new_direction)
		new_direction /= np.linalg.norm(new_direction)

		velocity = speed * new_direction
		player.set_velocity(model, data, velocity)

	def step(self, model, data, actions):
		for player, action in zip(self.players, actions):
			self.perform_action(model, data, player, action)

		obs = self.generate_state_space(model, data)
		rewards = self.generate_rewards(model, data)
		done = False
		info = {}

		return obs, rewards, done, info