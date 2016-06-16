checkpoint_dir = 'checkpoints'
log_file = 'tmp/a3c_log'
summary_freq = 500

local_t_max = 5 # repeat step size
rmsp_alpha = 0.99 # decay parameter for RMSProp
rmsp_epsilon = 0.1 # epsilon parameter for RMSProp
initial_alpha_low = 1e-4    # log_uniform low limit for learning rate
initial_alpha_high = 1e-2   # log_uniform high limit for learning rate
LSTM = False # Uses a feed-forward network if false
dropout_prob = 0 # Set to zero to disable dropout

num_threads = 8
num_actions = 2
num_states = 4

initial_alpha_log_rate = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4) ### description is incorrect
discount_rate = 0.5 # discount factor for rewards
entropy_beta = 0.01 # entropy regularization constant
max_time_steps = 6 * 10**7
grad_norm_clip = 40.0 # gradient norm clipping

rnn_size = 4
num_rnn_layers = 1
num_steps = min(5,local_t_max) # Number of steps to go back for truncated backprop.