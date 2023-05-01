def compute_reward():
	# Compute the distance to the ball and the goal
	distance_to_ball = distance_bw_ball_n_sphero()
	distance_to_goal = distance_bw_goal1_n_ball()

	# Compute the time penalty
	time_penalty = -0.0001

	# Compute the out-of-bounds penalty
	out_of_bound_penalty = Is_boundaries_touched()
	# Compute the touch ball reward
	touch_ball_reward = Is_ball_touched()
	# Compute the goal achieved reward
	goal_achieved_reward = Is_goal()

	# Compute the distance to the ball and goal coefficients
	distance_to_goal_coeff = - 0.01
	distance_to_ball_coeff = - 0.01
	Sphero_goal_penalty = Is_goal_sphero()

	# Compute the overall reward
	reward = (
			touch_ball_reward +
			goal_achieved_reward +
			distance_to_ball_coeff * distance_to_ball +
			distance_to_goal_coeff * distance_to_goal +
			#time_penalty +
			Sphero_goal_penalty+
			#rotation_penalty +
			out_of_bound_penalty)
	return reward, True if goal_achieved_reward!=0.0 else False, True if out_of_bound_penalty!=0 or Sphero_goal_penalty!=0 else False