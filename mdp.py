import numpy as np
		
class ChainMDP(object):
	"""Simple markov chain style MDP.  Three "rooms" and one absorbing
	state. States are encoded for the q_network as arrays with
	indicator entries. E.g. [1, 0, 0, 0] encodes state 0, and [0, 1,
	0, 0] encodes state 1.	The absorbing state is [0, 0, 0, 1]

	Action 0 moves the agent left, departing the maze if it is in state 0.
	Action 1 moves the agent to the right, departing the maze if it is in
	state 2.

	The agent receives a reward of 0.7 for departing the chain on the left, and
	a reward of 1.0 for departing the chain on the right.

	Assuming deterministic actions and a discount_rate rate of 0.5, the
	correct Q-values are:

	0.7|0.25,  0.35|0.5, 0.25|1.0,	0|0
	
	Average reward should be between 0.403 (random actions) and 0.733 (optimal actions)
	"""

	def __init__(self, success_prob=1.0, blackout_prob=0):
		self.num_actions = 2
		self.num_states = 4
		self.success_prob = success_prob
		self.blackout_prob = blackout_prob

		self.actions = [np.array([[0]], dtype='int32'),
						np.array([[1]], dtype='int32')]

		self.reward_zero = np.array([[0]], dtype='float32')
		self.reward_left = np.array([[.7]], dtype='float32')
		self.reward_right = np.array([[1.0]], dtype='float32')

		self.states = []
		for i in range(self.num_states):
			self.states.append(np.zeros((self.num_states),
										dtype='float32'))
			self.states[-1][i] = 1

	def act(self, state, action_index):
	
		#action 0 is left, 1 is right.
		state_index =  np.nonzero(state)[0][0]

		next_index = state_index
		if np.random.random() < self.success_prob:
			next_index = state_index + action_index * 2 - 1

		# Exit left
		if next_index == -1:
			return self.reward_left, self.states[-1], np.array([[True]])

		# Exit right
		if next_index == self.num_states - 1:
			return self.reward_right, self.states[-1], np.array([[True]])

		if np.random.random() < self.success_prob:
			return (self.reward_zero,
					self.states[state_index + action_index * 2 - 1],
					np.array([[False]]))
		else:
			return (self.reward_zero, self.states[state_index],
					np.array([[False]]))
