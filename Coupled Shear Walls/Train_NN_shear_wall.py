import tensorflow as tf
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time

# Load data and set hyperparameters
data_1 = scipy.io.loadmat('data_train_batch_1.mat')
data_2 = scipy.io.loadmat('data_train_batch_2.mat')
data_3 = scipy.io.loadmat('data_train_batch_3.mat')
parameters_1 = data_1['batchInputRandomParameters']
solutions_1 = data_1['solutions'].T
del data_1
parameters_2 = data_2['batchInputRandomParameters']
solutions_2 = data_2['solutions'].T
del data_2
parameters_3 = data_3['batchInputRandomParameters']
solutions_3 = data_3['solutions'].T
del data_3
solutions = np.vstack([solutions_1,solutions_2, solutions_3])
parameters = np.vstack([parameters_1, parameters_2, parameters_3])
solutions = np.reshape(solutions,(600,600,1966))
test_solutions = solutions[500:600,:,:]
test_parameters = parameters[500:600,:]
solutions = solutions[0:500,:,:]
parameters = parameters[0:500,:]
latent_dimension = 64
hidden_size = 256
epochs = 20000
batch_size = 100
learning_rate = 1e-4
s_abs_max = np.load('s_abs_max.npy')
solutions = solutions / s_abs_max
test_solutions = test_solutions / s_abs_max

# Scale the data
p_min = parameters.min(axis=0)
p_max= parameters.max(axis=0)
np.save('p_min.npy', p_min)
np.save('p_max.npy', p_max)
parameters = (parameters - p_min) / (p_max - p_min)
test_parameters = (test_parameters - p_min) / (p_max - p_min)
solutions_encoder = tf.keras.models.load_model('Encoder_shear_wall')
solutions_decoder = tf.keras.models.load_model('Decoder_shear_wall')

# Feed Forward Neural Network(FFNN)
input = (tf.keras.Input(shape = 3))
hidden_layer_1 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(input)
hidden_layer_2 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_2)
hidden_layer_4 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_3)
hidden_layer_5 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_4)
hidden_layer_6 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_5)
output = tf.keras.layers.Dense(latent_dimension)(hidden_layer_6)
neural_network =  tf.keras.Model(input, output)
neural_network.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')
NN_input = parameters
NN_output = solutions_encoder(solutions)
start_time = time.time()
history_NN = neural_network.fit(NN_input ,NN_output , batch_size = batch_size, epochs = epochs)
training_time = time.time() - start_time
print('Training time: ',   training_time, ' sec')
#neural_network.save('NN_shear_wall')

# FFNN L2-norm error
NN_results = neural_network(test_parameters)
encoded_results = neural_network(test_parameters)
encoded_solutions = solutions_encoder(test_solutions)
N_test_samples = test_parameters.shape[0]
error_nn = 0
for sample in range(N_test_samples):
    error_nn =  error_nn + (1/N_test_samples) * (np.linalg.norm(encoded_solutions[sample,:] - encoded_results[sample,:],2 ) / np.linalg.norm(encoded_solutions[sample,:],2 ))
print('L2 norm error - FFNN = ', error_nn)

# Summarize history for loss
plt.plot(history_NN.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

