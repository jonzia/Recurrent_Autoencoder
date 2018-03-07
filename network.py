# ----------------------------------------------------
# Network Class for Autoencoder v1.0.1
# Created by: Jonathan Zia
# Last Modified: Tuesdsay, March 6, 2018
# Georgia Institute of Technology
# ----------------------------------------------------

class Network():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs."""

	def __init__(self):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = 5			# Batch size
		self.num_steps = 100		# Max steps for BPTT
		self.num_lstm_hidden = 5	# Number of LSTM hidden units
		self.input_features = 9		# Number of input features
		self.i_keep_prob = 1.0		# Input keep probability / LSTM cell
		self.o_keep_prob = 1.0		# Output keep probability / LSTM cell

		# Special autoencoder parameters
		self.latent = 3				# Number of elements in latent layer


		# Decay type can be 'none', 'exp', 'inv_time', or 'nat_exp'
		self.decay_type = 'exp'		# Set decay type for learning rate
		self.learning_rate_init = 0.001		# Set initial learning rate for optimizer (default 0.001) (fixed LR for 'none')
		self.learning_rate_end = 0.00001	# Set ending learning rate for optimizer