# ----------------------------------------------------
# LSTM Network Test Bench for Autoencoder v1.0.2
# Created by: Jonathan Zia
# Last Modified: Tuesday, March 6, 2018
# Georgia Institute of Technology
# ----------------------------------------------------
import tensorflow as tf
import network_autoencoder as net
import pandas as pd
import random as rd
import numpy as np
import time
import math
import csv
import os


# ----------------------------------------------------
# User-Defined Constants
# ----------------------------------------------------
# Training
I_KEEP_PROB = 1.0						# Input keep probability / LSTM cell
O_KEEP_PROB = 1.0						# Output keep probability / LSTM cell
BATCH_SIZE = 1							# Batch size


# ----------------------------------------------------
# Instantiate Network Classes
# ----------------------------------------------------
lstm_encoder = net.EncoderNetwork()
lstm_decoder = net.DecoderNetwork(batch_size = BATCH_SIZE, num_steps = lstm_encoder.num_steps, 
	input_features = lstm_encoder.latent+lstm_encoder.input_features)
WINDOW_INT = lstm_encoder.num_steps		# Rolling window step interval


# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "/Users/username"
with tf.name_scope("Training_Data"):	# Testing dataset
	Dataset = os.path.join(dir_name, "data/dataset.csv")
with tf.name_scope("Model_Data"):		# Model load path
	load_path = os.path.join(dir_name, "checkpoints/model")
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	filewriter_path = os.path.join(dir_name, "test_bench")
with tf.name_scope("Output_Data"):		# Output data filenames (.txt)
	# These .txt files will contain prediction and target data respectively for Matlab analysis
	prediction_file = os.path.join(dir_name, "predictions.txt")
	target_file = os.path.join(dir_name, "targets.txt")

# Obtain length of testing and validation datasets
file_length = len(pd.read_csv(Dataset))

# ----------------------------------------------------
# User-Defined Methods
# ----------------------------------------------------
def init_values(shape):
	"""
	Initialize Weight and Bias Matrices
	Returns: Tensor of shape "shape" w/ normally-distributed values
	"""
	temp = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(temp)

def extract_data(filename, batch_size, num_steps, input_features, step):
	"""
	Extract features and labels from filename.csv as a rolling-window
	Returns:
	feature_batch ~ [1, num_steps, input_features]
	label_batch ~ [1, num_steps, input features]
	"""
	# NOTE: The empty dimension is required in order to feed inputs to LSTM cell.

	# Initialize numpy arrays for return value placeholders
	feature_batch = np.zeros((batch_size,num_steps, input_features))
	label_batch = np.zeros((batch_size, num_steps, input_features))

	# Import data from CSV as a sliding window:
	# First, import data starting from t = step to t = step + num_steps
	# ... add feature data to feature_batch[0, :, :]
	# ... assign label_batch the same value as feature_batch
	# Repeat for all batches.
	temp = pd.read_csv(filename, skiprows=step, nrows=num_steps, header=None)
	temp = temp.as_matrix()
	# Return features in specified columns
	feature_batch[0,:,:] = temp[:,1:input_features+1]
	# Return label batch, which has the same values as feature batch
	label_batch = feature_batch

	# Return feature and label batches
	return feature_batch, label_batch


	# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------

# Create placeholders for inputs and target values
# Input dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
# Target dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
inputs = tf.placeholder(tf.float32, [BATCH_SIZE, lstm_encoder.num_steps, lstm_encoder.input_features], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [BATCH_SIZE, lstm_encoder.num_steps, lstm_encoder.input_features], name="Target_Placeholder")


# ----------------------------------------------------
# Building an LSTM Encoder
# ----------------------------------------------------
# Build LSTM cell
# Creating basic LSTM cell
encoder_cell = tf.contrib.rnn.BasicLSTMCell(lstm_encoder.num_lstm_hidden,name='Encoder_Cell')
# Adding dropout wrapper to cell
encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, input_keep_prob=lstm_encoder.i_keep_prob, output_keep_prob=lstm_encoder.o_keep_prob)

# Initialize weights and biases for latent layer.
with tf.name_scope("Encoder_Variables"):
	W_latent = init_values([lstm_encoder.num_lstm_hidden, lstm_encoder.latent])
	tf.summary.histogram('Weights',W_latent)
	b_latent = init_values([lstm_encoder.latent])
	tf.summary.histogram('Biases',b_latent)

# Add LSTM cells to dynamic_rnn and implement truncated BPTT
initial_state_encoder = state_encoder = encoder_cell.zero_state(BATCH_SIZE, tf.float32)
with tf.variable_scope("Encoder_RNN"):
	for i in range(lstm_encoder.num_steps):
		# Obtain output at each step
		output, state_encoder = tf.nn.dynamic_rnn(encoder_cell, inputs[:,i:i+1,:], initial_state=state_encoder)
	# Obtain final output and convert to logit
	# Reshape output to remove extra dimension
	output = tf.reshape(output,[BATCH_SIZE,lstm_encoder.num_lstm_hidden])
	with tf.name_scope("Encoder_Output"):
		# Obtain logits by performing (weights)*(output)+(biases)
		logit = tf.matmul(output, W_latent) + b_latent
# Convert logits to tensor
latent_layer = tf.convert_to_tensor(logit)
# Converting to dimensions [batch_size, 1 (num_steps), latent]
latent_layer = tf.expand_dims(latent_layer,1,name='latent_layer')


# ----------------------------------------------------
# Building an LSTM Decoder
# ----------------------------------------------------
# Build LSTM cell
# Creating basic LSTM cells
# decoder_cell_1 is the first cell in the decoder layer, accepting latent_layer as an input
# decoder_cell_2 is each subsequent cell in the decoder layer, accepting the output at (t-1)
# as the input and the hidden state at (t-1) as the initial state.
decoder_cell_1 = tf.contrib.rnn.BasicLSTMCell(lstm_decoder.num_lstm_hidden,name='Decoder_Cell_1')
decoder_cell_2 = tf.contrib.rnn.BasicLSTMCell(lstm_decoder.num_lstm_hidden,name='Decoder_Cell_2')
# Adding dropout wrapper to each cell
decoder_cell_1 = tf.nn.rnn_cell.DropoutWrapper(decoder_cell_1, input_keep_prob=lstm_decoder.i_keep_prob, output_keep_prob=lstm_decoder.o_keep_prob)
decoder_cell_2 = tf.nn.rnn_cell.DropoutWrapper(decoder_cell_2, input_keep_prob=lstm_decoder.i_keep_prob, output_keep_prob=lstm_decoder.o_keep_prob)

# Initialize weights and biases for output layer.
with tf.name_scope("Decoder_Variables"):
	W_output = init_values([lstm_decoder.num_lstm_hidden, lstm_encoder.input_features])
	tf.summary.histogram('Weights',W_output)
	b_output = init_values([lstm_encoder.input_features])
	tf.summary.histogram('Biases',b_output)

# Initialize the initial state for the first LSTM cell
initial_state_decoder = state_decoder = decoder_cell_1.zero_state(BATCH_SIZE, tf.float32)
# Initialize placeholder for outputs of the decoder layer at each timestep
logits = []
with tf.variable_scope("Decoder_RNN"):
	for i in range(lstm_decoder.num_steps):
		# Obtain output at each step
		# For the first timestep...
		if i == 0:
			# Input the latent layer and obtain the output and hidden state
			output, state_decoder = tf.nn.dynamic_rnn(decoder_cell_1, latent_layer, initial_state=state_decoder)
		else: # For all subsequent timesteps...
			# Input the output at (t-1) and obtain the output at time (t) and the hidden state
			output, state_decoder = tf.nn.dynamic_rnn(decoder_cell_2, tf.expand_dims(output,1), initial_state=state_decoder)
		# Obtain output and convert to logit
		# Reshape output to remove extra dimension
		output = tf.reshape(output,[BATCH_SIZE,lstm_decoder.num_lstm_hidden])
		with tf.name_scope("Decoder_Output"):
			# Obtain logits by applying operation (weights)*(outputs)+(biases)
			logit = tf.matmul(output, W_output) + b_output
			# Append output at each timestep
			logits.append(logit)
# Convert logits to tensor entitled "predictions"
predictions = tf.convert_to_tensor(logits)
# Converting to dimensions [batch_size, num_steps, input_features]
predictions = tf.transpose(logits, perm=[1, 0, 2], name='Predictions')


# ----------------------------------------------------
# Calculate Loss
# ----------------------------------------------------
# Calculating softmax cross entropy of labels and logits
loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
loss = tf.reduce_mean(loss)


# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
saver = tf.train.Saver()	# Instantiate Saver class
with tf.Session() as sess:
	# Create Tensorboard graph
	writer = tf.summary.FileWriter(filewriter_path, sess.graph)
	merged = tf.summary.merge_all()

	# Restore saved session
	saver.restore(sess, load_path)

	# Running the network
	# Set range (prevent index out-of-range exception for rolling window)
	for step in range(0,file_length,WINDOW_INT):

		try: # While there is no out-of-bounds exception
			# Obtaining batch of features and labels from TRAINING dataset(s)
			features, labels = extract_data(Dataset, BATCH_SIZE, lstm_encoder.num_steps, lstm_encoder.input_features, step)
		except:
			break

		# Input data
		data = {inputs: features, targets:labels}
		# Run and evalueate for summary variables, loss, predictions, and targets
		summary, loss_, pred, tar = sess.run([merged, loss, predictions, targets], feed_dict=data)

		# Report parameters
		if True:	# Conditional statement for filtering outputs
			p_completion = math.floor(100*step*BATCH_SIZE/file_length)
			print("\nLoss: %.3f, Percent Completion: " % loss_, p_completion)
			print("\nPredictions:")
			print(pred)
			print("\nTargets:")
			print(tar)

			# Write results to file for Matlab analysis
			# Write predictions
			with open(prediction_file, 'a') as file_object:
				np.savetxt(file_object, pred[0,:,:])
			# Write targets
			with open(target_file, 'a') as file_object:
				np.savetxt(file_object, tar[0,:,:])

		# Writing summaries to Tensorboard
		writer.add_summary(summary,step)

	# Close the writer
	writer.close()