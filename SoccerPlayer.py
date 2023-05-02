import mujoco as mj
import numpy as np
from DDPG import Actor
import math

class SoccerPlayer():
	def __init__(self, model, data, name, team, env):
		self.name = name
		self.env = env

		self.team = team
		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
		
		self.brain = Actor(self.env.state_size, self.env.action_size, self.env.hidden_layers)

		# The direction the agent is initially facing
		self.forward = 0
		self._last_distance_to_ball = 0
		self._last_distance_to_goal = 0

		self.reward = 0

	def reset(self, model, data):
		if self.env.randomize_player_positions:
			position_x = np.random.uniform(-45, 45)
			position_y = np.random.uniform(-30, 30)
			self.set_position(model, data, (position_x, position_y, 0.365))

	def get_pose(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 7]

	def set_pose(self, model, data, position = None, quaternion = None):
		if position is not None:
			data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position
		if quaternion is not None:
			data.qpos[self.id_joint * 7 + 3: self.id_joint * 7 + 7] = quaternion

	def get_velocity(self, model, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]

	def set_velocity(self, model, data, velocity = None, angular_velocity = None):
		if velocity is not None:
			data.qvel[self.id_joint * 6: self.id_joint * 6 + 3] = velocity
		if angular_velocity is not None:
			data.qvel[self.id_joint * 6 + 3: self.id_joint * 6 + 6] = angular_velocity

	def get_position(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def set_position(self, model, data, position):
		data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position

	def get_direction(self, model, data):
		return data.qpos[self.id_joint * 7 + 3: self.id_joint * 7 + 7]

	def get_state(self, model, data):
		return np.concatenate((self.get_position(model, data)[:2], self.get_velocity(model, data)[:2], [self.forward])).tolist()

	def rotate(self, model, data, angle):
		self.forward += angle
		x_prime = math.cos(angle)
		y_prime = math.sin(angle)
		z_prime = 0
		
		return [x_prime, y_prime, z_prime]

	def compute_reward(self, model, data, env):
		contacts = data.contact
		for c in contacts:
			if c.geom1 == self.id_geom and c.geom2 in env.geom_id_boundaries.values():
				self.reward = -1000000
				return self.reward

		pos_sphero = self.get_position(model, data)
		pos_ball = self.env.ball.get_position(model, data)
		distance_to_ball = np.linalg.norm(pos_ball - pos_sphero)
		reward_distance_of_player_to_ball = -0.01 * distance_to_ball
		self._last_distance_to_ball = distance_to_ball

		distance_to_goal = 0
		if self.team == 'A':
			distance_to_goal = env.ball.get_distance_from_goal_A(model, data, env)
		elif self.team == 'B':
			distance_to_goal = env.ball.get_distance_from_goal_B(model, data, env)
		reward_distance_of_ball_to_goal = -0.01 * distance_to_goal
		self._last_distance_to_goal = distance_to_goal

		player_kicked_ball = env.ball.is_kicked_by_player(self.id_geom)
		reward_player_kicked_ball = 0
		if player_kicked_ball:
			reward_player_kicked_ball = 100

		reward_player_kicked_ball_out_of_bounds = 0
		reward_team_kicked_ball_out_of_bounds = 0
		is_ball_out_of_bounds = env.ball.is_boundary_line_touched()
		if is_ball_out_of_bounds:
			if env.ball._last_hit == self.id_geom:
				reward_player_kicked_ball_out_of_bounds = -1000
			elif env.ball._possession == self.team:
				reward_team_kicked_ball_out_of_bounds = -1000

		reward_goal_scored_by_player_team = 0
		reward_goal_scored_by_opponent = 0
		if env.ball._goal:
			if env.ball._possession == self.team:
				reward_goal_scored_by_player_team = 1000
			else:
				reward_goal_scored_by_opponent = -10000

		self.reward = reward_distance_of_player_to_ball + reward_distance_of_ball_to_goal + reward_player_kicked_ball + reward_player_kicked_ball_out_of_bounds + reward_team_kicked_ball_out_of_bounds + reward_goal_scored_by_player_team + reward_goal_scored_by_opponent
			
		return self.reward

