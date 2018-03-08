# ----------------------------------------------------
# Network Class for Autoencoder v1.0.2
# Created by: Jonathan Zia
# Last Modified: Thursday, March 8, 2018
# Georgia Institute of Technology
# ----------------------------------------------------

class EncoderNetwork():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs for ENCODER."""

	def __init__(self):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = 5			# Batch size
		self.num_steps = 100		# Max steps for BPTT
		self.num_lstm_hidden = 15	# Number of LSTM hidden units
		self.input_features = 9		# Number of input features
		self.i_keep_prob = 1.0		# Input keep probability / LSTM cell
		self.o_keep_prob = 1.0		# Output keep probability / LSTM cell

		# Special autoencoder parameters
		self.latent = 10			# Number of elements in latent layer

class DecoderNetwork():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs for DECODER."""

	def __init__(self, batch_size, num_steps, input_features):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = batch_size			# Batch size
		self.num_steps = num_steps				# Max steps for BPTT
		self.num_lstm_hidden = 15				# Number of LSTM hidden units
		self.input_features = input_features	# Number of input features
		self.i_keep_prob = 1.0				# Input keep probability / LSTM cell
		self.o_keep_prob = 1.0				# Output keep probability / LSTM cell