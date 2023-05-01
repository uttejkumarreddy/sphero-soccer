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
		
		self.size_hidden_layers = 256
		self.brain = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.size_hidden_layers)

		# The direction the agent is initially facing
		self.forward = 0

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

	def compute_reward(self, model, data):
		# Detect collisions
		
		# Compute the distance to the ball
		# Compute the time penalty
		time_penalty = -0.0001

