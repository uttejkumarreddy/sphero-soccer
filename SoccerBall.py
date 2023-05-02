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
		self._possession = None
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

	def is_boundary_line_touched(self):
		return self._outside

	def is_kicked_by_player(self, player_geom_id):
		return self._hit == player_geom_id

	def is_goal_line_touched(self):
		return self._goal

	def update_state(self, model, data, env):
		contacts = data.contact
		for c in contacts:
			# Is the ball kicked by a player
			if c.geom1 == self.id_geom and c.geom2 in env.players_geom_ids:
				self._last_hit = self._hit
				self._hit = c.geom2
				if self._hit in env.players_Team_A_geom_ids:
					self._possession = "A"
				elif self._hit in env.players_Team_B_geom_ids:
					self._possession = "B"

			# Is the ball outside the boundaries
			if c.geom1 == self.id_geom and c.geom2 in env.geom_id_lines_touch.values():
				self._outside = True
			else:
				self._outside = False

			# Is the ball at the goal
			if c.geom1 == self.id_geom and c.geom2 in env.geom_id_lines_goal.values():
				self._goal = True
			else:
				self._goal = False

	def get_distance_from_goal_A(self, model, data, env):
		ball_pos = self.get_position(model, data)
		return (np.linalg.norm(np.cross(ball_pos - env.goal_A_point_a, ball_pos - env.goal_A_point_b)) / np.linalg.norm(env.goal_A_point_b - env.goal_A_point_a))

	def get_distance_from_goal_B(self, model, data, env):
		ball_pos = self.get_position(model, data)
		return (np.linalg.norm(np.cross(ball_pos - env.goal_B_point_a, ball_pos - env.goal_B_point_b)) / np.linalg.norm(env.goal_B_point_b - env.goal_B_point_a))
