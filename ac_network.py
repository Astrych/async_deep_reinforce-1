import tensorflow as tf
import numpy as np
from constants import rnn_size, num_rnn_layers, dropout_prob, num_steps


# Actor-Critic Network (policy and value network)
class AC3Net(object):

	# Create placeholder variables in order to calculate the loss
	def prepare_loss(self, entropy_beta):
	
		# taken action (input for policy)
		self.a = tf.placeholder("float", [1, self._num_actions])
	
		# temporal difference (R-V) (input for policy)
		self.td = tf.placeholder("float", [1])
		
		# Entropy of the policy
		entropy = -tf.reduce_sum(self.pi * tf.log(self.pi))

		# Policy loss (output)
		# Minus because this is for gradient ascent
		policy_loss = -(tf.reduce_sum( tf.mul(tf.log(self.pi),self.a)) * self.td + entropy*entropy_beta)

		# R (input for value)
		self.r = tf.placeholder("float", [1])

		# Learning rate for critic is half of actor's, so multiply by 0.5
		value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

		self.total_loss = policy_loss + value_loss

	def sync_from(self, src_network, name=None):
		src_vars = src_network.trainable_vars ### built-in function to do this?
		dst_vars = self.trainable_vars

		sync_ops = []
		with tf.op_scope([], name, "AC3Net") as name:
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var)
				sync_ops.append(sync_op)

			return tf.group(*sync_ops, name=name)

	# weight initialization based on muupan's code
	# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
	### Should be outside of this class (init_torch_vector too)
	def init_torch_matrix(self, shape):
		# Uniform distribution, mean 0
		input_channels = shape[0]
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial)

	def init_torch_vector(self, shape, input_channels):
		# Uniform distribution, mean 0
		d = 1.0 / np.sqrt(input_channels)
		initial = tf.random_uniform(shape, minval=-d, maxval=d)
		return tf.Variable(initial)	   

	def _debug_save_sub(self, sess, prefix, var, name):
		var_val = var.eval(sess)
		var_val = np.reshape(var_val, (1, np.product(var_val.shape)))				 
		np.savetxt('./' + prefix + '_' + name + '.csv', var_val, delimiter=',')
	

class AC3LSTM(AC3Net):
	# For an RNN, input is usually of shape [batch_size,num_steps]
	# Here they are both 1, as is the case for sampling in a generative RNN
	def __init__(self, num_actions, num_states, num_trainable_vars):		
		self._num_actions = num_actions
		self._num_states = num_states
	
		# Input (not the cell state)
		self.state = tf.placeholder(tf.float32, [1,num_states])

		# Weights for policy output layer
		self.W_fc1 = self.init_torch_matrix([rnn_size, num_actions])
		self.b_fc1 = self.init_torch_vector([num_actions], rnn_size)
		
		# Weights for value output layer
		self.W_fc2 = self.init_torch_matrix([rnn_size, 1])
		self.b_fc2 = self.init_torch_vector([1], rnn_size)	
		
		rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size, activation=tf.identity) ### Use LSTM
		### Dropout?
		self.cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_rnn_layers)

		self.rnn_state = self.cell.zero_state(1, tf.float32)
		output, rnn_state_out = self.cell(self.state, self.rnn_state)
		
		self.rnn_state_out = rnn_state_out
	
		# policy (output)
		self.pi = tf.nn.softmax(tf.matmul(output, self.W_fc1) + self.b_fc1)

		# value - linear output layer
		self.v = tf.matmul(output, self.W_fc2) + self.b_fc2
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
	
	
	def run_policy(self, sess, state, update_rnn_state):
		[pi_out, rnn_state] = sess.run([self.pi,self.rnn_state_out], feed_dict = {self.state:[state]})
		if update_rnn_state:
			self.rnn_state = rnn_state			
		return pi_out[0]

		
	def run_value(self, sess, state, update_rnn_state):
		[v_out, rnn_state] = sess.run([self.v,self.rnn_state_out], feed_dict = {self.state:[state]})
		if update_rnn_state:
			self.rnn_state = rnn_state			
		return v_out[0][0]
		
		
	def reset_state(self):
		self.rnn_state = self.cell.zero_state(1, tf.float32)
		
		
# Feed-forward
class AC3FF(AC3Net):
	def __init__(self, num_actions, num_states, num_trainable_vars):
		self._num_actions = num_actions
	
		# Input
		# Batch_size is 1
		self.state = tf.placeholder("float", [1, num_states])

		self.W_fc1 = self.init_torch_matrix([num_states, num_actions])
		self.b_fc1 = self.init_torch_vector([num_actions], num_states)
		
		# weight for value output layer
		self.W_fc2 = self.init_torch_matrix([num_states, 1])
		self.b_fc2 = self.init_torch_vector([1], num_states)

		# policy - softmax over actions
		self.pi = tf.nn.softmax(tf.matmul(self.state, self.W_fc1) + self.b_fc1)
		# value - linear output layer
		self.v = tf.matmul(self.state, self.W_fc2) + self.b_fc2
		
		if num_trainable_vars[0] == None:
			num_trainable_vars[0] = len(tf.trainable_variables())
		
		self.trainable_vars = tf.trainable_variables()[-num_trainable_vars[0]:]
		
	def run_policy(self, sess, state):
		pi_out = sess.run(self.pi, feed_dict = {self.state : [state]})
		return pi_out[0]

	def run_value(self, sess, state):
		v_out = sess.run(self.v, feed_dict = {self.state : [state]})
		return v_out[0][0] # output is scalar
		
	def debug_save(self, sess, prefix): ### Change for LSTM or make general
		self._save_sub(sess, prefix, self.W_fc1, "W_fc1")
		self._save_sub(sess, prefix, self.b_fc1, "b_fc1")
		self._save_sub(sess, prefix, self.W_fc2, "W_fc2")
		self._save_sub(sess, prefix, self.b_fc2, "b_fc2")
		#for i in self.trainable_vars:
		#	self._save_sub(sess, prefix, i, "")
