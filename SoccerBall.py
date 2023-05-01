import mujoco as mj
import numpy as np

class SoccerBall():
	def __init__(self, model, data, name):
		self.name = name

		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)

		self._last_hit = None
		self._hit = False

		# TODO: Update the following attributes
		self._repossessed = False
		self._intercepted = False

		self._outside = False
		self._goal = False

		# TODO: Update the following attributes
		self._pos_at_last_step = None
		self._dist_since_last_hit = None
		self._dist_between_last_hits = None

	def reset(self, model, data):
		self._last_hit = None
		self._hit = False
		self._repossessed = False
		self._intercepted = False

		self._pos_at_last_step = None
		self._dist_since_last_hit = None
		self._dist_between_last_hits = None

		self.set_position(model, data, (0, 0, 0.365))

	def get_position(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def set_position(self, model, data, position):
		data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position

	def get_velocity(self, model, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]

	def get_state(self, model, data):
		return np.concatenate((self.get_position(model, data)[:2], self.get_velocity(model, data)[:2])).tolist()

	def update_state(self, model, data, env):
		contacts = data.contact
		for c in contacts:
			# Is the ball kicked by a player
			if c.geom1 == self.id_geom and c.geom2 in env.players_geom_ids:
				self._last_hit = self._hit
				self._hit = c.geom2

			# Is the ball outside the boundaries
			if c.geom1 == self.id_geom and c.geom2 in env.geom_id_lines_touch:
				self._outside = True
			else:
				self._outside = False

			# Is the ball at the goal
			if c.geom1 == self.id_geom and c.geom2 in env.geom_id_lines_goal:
				self._goal = True
			else:
				self._goal = False

	def get_distance_from_goal_A(self, model, data):
		goal_A_point_a = np.array([-45, 5, 0])
		goal_A_point_b = np.array([-45, -5, 0])
		ball_pos = self.get_position(model, data)
		return (np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a))

	def get_distance_from_goal_B(self, model, data):
		goal_B_point_a = np.array([45, 5, 0])
		goal_B_point_b = np.array([45, -5, 0])
		ball_pos = self.get_position(model, data)
		return (np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a))	