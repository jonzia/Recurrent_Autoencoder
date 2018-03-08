# ----------------------------------------------------
# Time-Series Autoencoder using Tensorflow 1.0.2
# Created by: Jonathan Zia
# Last Modified: Thursday, March 8, 2018
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
NUM_TRAINING = 100			# Number of training batches (balanced minibatches)
NUM_VALIDATION = 100		# Number of validation batches (balanced minibatches)

# Learning rate decay
# Decay type can be 'none', 'exp', 'inv_time', or 'nat_exp'
DECAY_TYPE = 'exp'					# Set decay type for learning rate
LEARNING_RATE_INIT = 0.001			# Set initial learning rate for optimizer (default 0.001) (fixed LR for 'none')
LEARNING_RATE_END = 0.00001			# Set ending learning rate for optimizer

# Load File
LOAD_FILE = False 		# Load initial LSTM model from saved checkpoint?


# ----------------------------------------------------
# Instantiate Network Classes
# ----------------------------------------------------
lstm_encoder = net.EncoderNetwork()
lstm_decoder = net.DecoderNetwork(batch_size = lstm_encoder.batch_size, num_steps = lstm_encoder.num_steps, 
	input_features = lstm_encoder.latent+lstm_encoder.input_features)


# ----------------------------------------------------
# Input data files
# ----------------------------------------------------
# Specify filenames
# Root directory:
dir_name = "/Users/username"
with tf.name_scope("Training_Data"):	# Training dataset
	tDataset = os.path.join(dir_name, "data/trainingdata.csv")
with tf.name_scope("Validation_Data"):	# Validation dataset
	vDataset = os.path.join(dir_name, "data/validationdata.csv")
with tf.name_scope("Model_Data"):		# Model save/load paths
	# load_path = "D:\\Documents\\checkpoints\\model"				# Load previous model
	# save_path = "D:\\Documents\\checkpoints\\model"				# Save model at each step
	# save_path_op = "D:\\Documents\\checkpoints\\model_op"		# Save optimal model
	load_path = os.path.join(dir_name, "checkpoints/model")		# Load previous model
	save_path = os.path.join(dir_name, "checkpoints/model")		# Save model at each step
	save_path_op = os.path.join(dir_name, "checkpoints/model_op")	# Save optimal model
with tf.name_scope("Filewriter_Data"):	# Filewriter save path
	#filewriter_path = "D:\\Documents\\output"
	filewriter_path = os.path.join(dir_name, "output")
with tf.name_scope("Output_Data"):		# Output data filenames (.txt)
	# These .txt files will contain loss data for Matlab analysis
	#training_loss = "D:\\Documents\\training_loss.txt"
	#validation_loss = "D:\\Documents\\validation_loss.txt"
	training_loss = os.path.join(dir_name, "training_loss.txt")
	validation_loss = os.path.join(dir_name, "validation_loss.txt")

# Obtain length of testing and validation datasets
file_length = len(pd.read_csv(tDataset))
v_file_length = len(pd.read_csv(vDataset))


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


def extract_data(filename, batch_size, num_steps, input_features, f_length):
	"""
	Extract features and labels from filename.csv in random batches
	Returns:
	feature_batch ~ [batch_size, num_steps, input_features]
	label_batch := feature_batch
	"""

	# Initialize numpy arrays for return value placeholders
	feature_batch = np.zeros((batch_size,num_steps,input_features))
	label_batch = np.zeros((batch_size,num_steps,input_features))

	# Import data from CSV as a random minibatch:
	for i in range(batch_size):

		# Generate random index for number of rows to skip
		temp_index = rd.randint(0, f_length-num_steps-1)
		# Read data from CSV and write as matrix
		temp = pd.read_csv(filename, skiprows=temp_index, nrows=num_steps, header=None)
		temp = temp.as_matrix()

		# Return features in specified columns
		feature_batch[i,:,:] = temp[:,1:input_features+1]
		# Setting features as labels for autoencoding
		label_batch[i,:,:] = feature_batch[i,:,:]

	# Return feature and label batches
	return feature_batch, label_batch


def set_decay_rate(decay_type, learning_rate_init, learning_rate_end, num_training):
	"""
	Calcualte decay rate for specified decay type
	Returns: Scalar decay rate
	"""

	if decay_type == 'none':
		return 0
	elif decay_type == 'exp':
		return math.pow((learning_rate_end/learning_rate_init),(1/num_training))
	elif decay_type == 'inv_time':
		return ((learning_rate_init/learning_rate_end)-1)/num_training
	elif decay_type == 'nat_exp':
		return (-1/num_training)*math.log(learning_rate_end/learning_rate_init)
	else:
		return 0


def decayed_rate(decay_type, decay_rate, learning_rate_init, step):
	"""
	Calculate decayed learning rate for specified parameters
	Returns: Scalar decayed learning rate
	"""

	if decay_type == 'none':
		return learning_rate_init
	elif decay_type == 'exp':
		return learning_rate_init*math.pow(decay_rate,step)
	elif decay_type == 'inv_time':
		return learning_rate_init/(1+decay_rate*step)
	elif decay_type == 'nat_exp':
		return learning_rate_init*math.exp(-decay_rate*step)


# ----------------------------------------------------
# Importing Session Parameters
# ----------------------------------------------------

# Create placeholders for inputs and target values
# Input dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
# Target dimensions: BATCH_SIZE x NUM_STEPS x INPUT_FEATURES
inputs = tf.placeholder(tf.float32, [lstm_encoder.batch_size, lstm_encoder.num_steps, lstm_encoder.input_features], name="Input_Placeholder")
targets = tf.placeholder(tf.float32, [lstm_encoder.batch_size, lstm_encoder.num_steps, lstm_encoder.input_features], name="Target_Placeholder")

# Create placeholder for learning rate
learning_rate = tf.placeholder(tf.float32, name="Learning_Rate_Placeholder")


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
initial_state_encoder = state_encoder = encoder_cell.zero_state(lstm_encoder.batch_size, tf.float32)
with tf.variable_scope("Encoder_RNN"):
	for i in range(lstm_encoder.num_steps):
		# Obtain output at each step
		output, state_encoder = tf.nn.dynamic_rnn(encoder_cell, inputs[:,i:i+1,:], initial_state=state_encoder)
	# Obtain final output and convert to logit
	# Reshape output to remove extra dimension
	output = tf.reshape(output,[lstm_encoder.batch_size,lstm_encoder.num_lstm_hidden])
	with tf.name_scope("Encoder_Output"):
		# Obtain logits by passing output
		logit = tf.matmul(output, W_latent) + b_latent
latent_layer = tf.convert_to_tensor(logit)
# Converting to dimensions [batch_size, 1 (num_steps), latent]
latent_layer = tf.expand_dims(latent_layer,1,name='latent_layer')


# ----------------------------------------------------
# Building an LSTM Decoder
# ----------------------------------------------------
# Build LSTM cell
# Creating basic LSTM cell
decoder_cell_1 = tf.contrib.rnn.BasicLSTMCell(lstm_decoder.num_lstm_hidden,name='Decoder_Cell_1')
decoder_cell_2 = tf.contrib.rnn.BasicLSTMCell(lstm_decoder.num_lstm_hidden,name='Decoder_Cell_2')
# Adding dropout wrapper to cell
decoder_cell_1 = tf.nn.rnn_cell.DropoutWrapper(decoder_cell_1, input_keep_prob=lstm_decoder.i_keep_prob, output_keep_prob=lstm_decoder.o_keep_prob)
decoder_cell_2 = tf.nn.rnn_cell.DropoutWrapper(decoder_cell_2, input_keep_prob=lstm_decoder.i_keep_prob, output_keep_prob=lstm_decoder.o_keep_prob)

# Initialize weights and biases for output layer.
with tf.name_scope("Decoder_Variables"):
	W_output = init_values([lstm_decoder.num_lstm_hidden, lstm_encoder.input_features])
	tf.summary.histogram('Weights',W_output)
	b_output = init_values([lstm_encoder.input_features])
	tf.summary.histogram('Biases',b_output)

initial_state_decoder = state_decoder = decoder_cell_1.zero_state(lstm_decoder.batch_size, tf.float32)
logits = []
with tf.variable_scope("Decoder_RNN"):
	for i in range(lstm_decoder.num_steps):
		# Obtain output at each step
		if i == 0:
			output, state_decoder = tf.nn.dynamic_rnn(decoder_cell_1, latent_layer, initial_state=state_decoder)
		else:
			output, state_decoder = tf.nn.dynamic_rnn(decoder_cell_2, tf.expand_dims(output,1), initial_state=state_decoder)
		# Obtain output and convert to logit
		# Reshape output to remove extra dimension
		output = tf.reshape(output,[lstm_decoder.batch_size,lstm_decoder.num_lstm_hidden])
		with tf.name_scope("Decoder_Output"):
			# Obtain logits by passing output
			logit = tf.matmul(output, W_output) + b_output
			logits.append(logit)
predictions = tf.convert_to_tensor(logits)
# Converting to dimensions [batch_size, num_steps, input_features]
predictions = tf.transpose(logits, perm=[1, 0, 2], name='Predictions')


# ----------------------------------------------------
# Calculate Loss and Define Optimizer
# ----------------------------------------------------
# Calculating softmax cross entropy of labels and logits
loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# ----------------------------------------------------
# Run Session
# ----------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()	# Instantiate Saver class
t_loss = []	# Placeholder for training loss values
v_loss = []	# Placeholder for validation loss values
with tf.Session() as sess:
	# Create Tensorboard graph
	writer = tf.summary.FileWriter(filewriter_path, sess.graph)
	merged = tf.summary.merge_all()

	# If there is a model checkpoint saved, load the checkpoint. Else, initialize variables.
	if LOAD_FILE:
		# Restore saved session
		saver.restore(sess, load_path)
	else:
		# Initialize the variables
		sess.run(init)

	# Training the network

	# Determine whether to use sliding-window or minibatching
	step_range = NUM_TRAINING 		# Set step range for training
	v_step_range = NUM_VALIDATION	# Set step range for validation

	# Obtain start time
	start_time = time.time()
	# Initialize optimal loss
	loss_op = 0
	# Determine learning rate decay
	decay_rate = set_decay_rate(DECAY_TYPE, LEARNING_RATE_INIT, LEARNING_RATE_END, NUM_TRAINING)

	if DECAY_TYPE != 'none':
		print('\nLearning Decay Rate = ', decay_rate)

	# Set number of trials to NUM_TRAINING
	for step in range(0,step_range):

		# Initialize optimal model saver to False
		save_op = False

		try:	# While there is no out-of-bounds exception...

			# Obtaining batch of features and labels from TRAINING dataset(s)
			features, labels = extract_data(tDataset, lstm_encoder.batch_size, lstm_encoder.num_steps, lstm_encoder.input_features, file_length)

		except:
			break

		# Set optional conditional for network training
		if True:
			# Print step
			print("\nOptimizing at step", step)

			# Calculate time-decay learning rate:
			decayed_learning_rate = decayed_rate(DECAY_TYPE, decay_rate, LEARNING_RATE_INIT, step)

			# Input data and learning rate
			feed_dict = {inputs: features, targets:labels, learning_rate:decayed_learning_rate}
			# Run optimizer, loss, and predicted error ops in graph
			predictions_, targets_, _, loss_ = sess.run([predictions, targets, optimizer, loss], feed_dict=feed_dict)

			# Record loss
			t_loss.append(loss_)

			# Evaluate network and print data in terminal periodically
			with tf.name_scope("Validation"):
				# Conditional statement for validation and printing
				if step % 50 == 0:
					print("\nMinibatch train loss at step", step, ":", loss_)

					# Evaluate network
					test_loss = []
					for step_num in range(0,v_step_range):

						try:	# While there is no out-of-bounds exception...

							# Obtaining batch of features and labels from VALIDATION dataset(s)
							v_features, v_labels =  extract_data(vDataset, lstm_encoder.batch_size, lstm_encoder.num_steps, lstm_encoder.input_features, v_file_length)

						except:
							break
						
						# Input data and run session to find loss
						data_test = {inputs: v_features, targets: v_labels}
						loss_test = sess.run(loss, feed_dict=data_test)
						test_loss.append(loss_test)

					# Record loss
					v_loss.append(np.mean(test_loss))
					# Print test loss
					print("Test loss: %.3f" % np.mean(test_loss))
					# For the first step, set optimal loss to test loss
					if step == 0:
						loss_op = np.mean(test_loss)
					# If test_loss < optimal loss, overwrite optimal loss
					if np.mean(test_loss) < loss_op:
						loss_op = np.mean(test_loss)
						save_op = True 	# Save model as new optimal model

					# Print predictions and targets for reference
					print("Predictions:")
					print(predictions_)
					print("Targets:")
					print(targets_)

			# Save and overwrite the session at each training step
			saver.save(sess, save_path)
			# Save the model if loss over the test set is optimal
			if save_op:
				saver.save(sess,save_path_op)

			# Writing summaries to Tensorboard at each training step
			summ = sess.run(merged)
			writer.add_summary(summ,step)

		# Conditional statement for calculating time remaining and percent completion
		if step % 10 == 0:

			# Report percent completion
			p_completion = 100*step/NUM_TRAINING
			print("\nPercent completion: %.3f%%" % p_completion)

			# Print time remaining
			avg_elapsed_time = (time.time() - start_time)/(step+1)
			sec_remaining = avg_elapsed_time*(NUM_TRAINING-step)
			min_remaining = round(sec_remaining/60)
			print("\nTime Remaining: %d minutes" % min_remaining)

			# Print learning rate if learning rate decay is used
			if DECAY_TYPE != 'none':
				print("\nLearning Rate = ", decayed_learning_rate)

	# Write training and validation loss to file
	t_loss = np.array(t_loss)
	v_loss = np.array(v_loss)
	with open(training_loss, 'a') as file_object:
		np.savetxt(file_object, t_loss)
	with open(validation_loss, 'a') as file_object:
		np.savetxt(file_object, v_loss)

	# Close the writer
	writer.close()