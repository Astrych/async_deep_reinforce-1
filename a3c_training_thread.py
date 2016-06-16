import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from ac_network import AC3LSTM, AC3FF
from mdp import ChainMDP

from constants import discount_rate, local_t_max, entropy_beta, \
	num_actions, num_states, LSTM

class A3CTrainingthread(object):
	def __init__(self,
			 sess,
			 thread_index,
			 global_network,
			 initial_learning_rate,
			 learning_rate_input,
			 grad_applier,
			 max_global_time_step,
			 num_trainable_vars):

		self.thread_index = thread_index
		self.learning_rate_input = learning_rate_input
		self.max_global_time_step = max_global_time_step	
		
		if LSTM:
			initializer = tf.random_uniform_initializer(-0.1, 0.1)		
			with tf.variable_scope("model"+str(thread_index), reuse=None, initializer=initializer):
				self.local_network = AC3LSTM(num_actions, num_states, num_trainable_vars)
		else:
			self.local_network = AC3FF(num_actions, num_states, num_trainable_vars)
			
		self.local_network.prepare_loss(entropy_beta)

		self.trainer = AccumTrainer()
		self.trainer.prepare_minimize(self.local_network.total_loss, self.local_network.trainable_vars)
		
		self.accum_gradients = self.trainer.accumulate_gradients()
		self.reset_gradients = self.trainer.reset_gradients()
	
		self.apply_gradients = grad_applier.apply_gradients(
			global_network.trainable_vars,
			self.trainer.get_accum_grad_list() )

		self.sync = self.local_network.sync_from(global_network)
		self.game_state = ChainMDP()
		self.local_t = 0
		self.initial_learning_rate = initial_learning_rate
		self.episode_reward = 0


	def _anneal_learning_rate(self, global_time_step):
		learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
		if learning_rate < 0:
			learning_rate = 0
		return learning_rate

	def choose_action(self, pi_values):
		values = []
		sum = 0
		for rate in pi_values:
			sum = sum + rate
			value = sum
			values.append(value)
		
		r = random.random() * sum
		for i in range(len(values)):
			if values[i] >= r:
				return i;
		#fail safe
		return len(values)-1

	# Run for one episode
	def thread(self, sess, global_t):
		states = []
		actions = []
		rewards = []
		values = []

		terminal_end = False
		
		if LSTM:
			self.local_network.reset_state()
			
		# reset accumulated gradients
		sess.run(self.reset_gradients)
		# copy weights from shared to local
		sess.run(self.sync)
		start_local_t = self.local_t
	
		mdp = ChainMDP()
		state = mdp.states[np.random.randint(0, mdp.num_states-1)]
		discounted_reward = 0
		
		for i in range(local_t_max):
			if LSTM:
				action_probs = self.local_network.run_policy(sess, state, update_rnn_state=True)
			else:
				action_probs = self.local_network.run_policy(sess, state)
			
			action = self.choose_action(action_probs)
			states.append(state)
			actions.append(action)
			
			if LSTM:
			#	# Do not update the state again
				value_ = self.local_network.run_value(sess, state, update_rnn_state=False)
			else:
				value_ = self.local_network.run_value(sess, state)
				
			values.append(value_)

			reward, next_state, terminal = mdp.act(state, action)
			self.episode_reward += reward

			rewards.append(reward)

			self.local_t += 1
			state = next_state
			
			if terminal:
				terminal_end = True
				discounted_reward = (discount_rate**i)*self.episode_reward
				self.episode_reward = 0
				state = mdp.states[np.random.randint(0, mdp.num_states-1)]
				if LSTM:
					self.local_network.reset_state()
				break

		R = 0.0
		if not terminal_end:
			if LSTM:
				# Do not update the state again
				R = self.local_network.run_value(sess, state, update_rnn_state=False) 
			else:
				R = self.local_network.run_value(sess, state) 

		# Order from the final time point to the first
		### why?
		actions.reverse()
		states.reverse()
		rewards.reverse()
		values.reverse()

		# compute and accumulate gradients
		for (action, r, state, V) in zip(actions, rewards, states, values):
			R = r[0][0] + discount_rate * R
			td = R - V # temporal difference
			a = np.zeros([num_actions])
			a[action] = 1
			#a = np.reshape(a,[1,num_actions]) ### Should be done when the variable is created - or change something on the other end
			sess.run(self.accum_gradients,
								feed_dict = {
									#self.local_network.state: [state],
									self.local_network.state: np.reshape([float(i) for i in state],[1,mdp.num_states]), ### use np.array( ,dtype=...) instead
									self.local_network.a: [a],
									self.local_network.td: [td],
									self.local_network.r: [R]})
			
		cur_learning_rate = self._anneal_learning_rate(global_t)

		sess.run(self.apply_gradients, feed_dict = {self.learning_rate_input: cur_learning_rate})

		# local step
		diff_local_t = self.local_t - start_local_t
		return diff_local_t, discounted_reward
		
