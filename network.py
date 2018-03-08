# ----------------------------------------------------
# Network Class for Autoencoder v1.0.2
# Created by: Jonathan Zia
# Last Modified: Thursday, March 8, 2018
# Georgia Institute of Technology
# ----------------------------------------------------

class Network():
	"""Class containing all network parameters for use
	across LSTM_main and test_bench programs."""

	def __init__(self, batch_size, num_steps, num_lstm_hidden, input_features, latent=0, i_keep_prob=1.0, o_keep_prob=1.0):
		"""Initialize network attributes"""
		
		# Architecture
		self.batch_size = batch_size			# Batch size
		self.num_steps = num_steps				# Max steps for BPTT
		self.num_lstm_hidden = num_lstm_hidden	# Number of LSTM hidden units
		self.input_features = input_features	# Number of input features
		self.i_keep_prob = i_keep_prob			# Input keep probability / LSTM cell
		self.o_keep_prob = o_keep_prob			# Output keep probability / LSTM cell

		# Special autoencoder parameters
		self.latent = latent					# Number of elements in latent layer