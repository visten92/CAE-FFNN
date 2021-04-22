import tensorflow as tf
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split

# Load data and set hyperparameters
data =  scipy.io.loadmat('burgers_train.mat')
parameters = data['nu']
solutions = data['vu']
solutions, test_solutions, parameters, test_parameters = train_test_split(solutions, parameters, test_size=0.50, shuffle=False)
solutions_encoder = tf.keras.models.load_model('Encoder_burgers')
solutions_decoder = tf.keras.models.load_model('Decoder_burgers')
latent_space_dimension = 8
hidden_size = 32
epochs = 30000
batch_size = 100
learning_rate = 1e-4

# Feed Forward Neural Network (FFNN)
input = (tf.keras.Input(shape = 1))
hidden_layer_1 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(input)
hidden_layer_2 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_1)
hidden_layer_3 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_2)
hidden_layer_4 = tf.keras.layers.Dense(hidden_size, activation = 'relu')(hidden_layer_3)
output = tf.keras.layers.Dense(latent_space_dimension)(hidden_layer_4)
neural_network =  tf.keras.Model(input, output)
neural_network.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')
NN_input_train = parameters
NN_input_test = test_parameters
NN_output_train = solutions_encoder(solutions)
NN_output_test = solutions_encoder(test_solutions)
history_NN = neural_network.fit(NN_input_train, NN_output_train, validation_data=(NN_input_test, NN_output_test), batch_size = batch_size, epochs = epochs)
#neural_network.save('NN_burgers')

# FFNN L2-norm error
encoded_results = neural_network(test_parameters)
encoded_solutions = solutions_encoder(test_solutions)
N_test_samples = test_parameters.shape[0]
error_FFNN = 0
for sample in range(N_test_samples):
    error_FFNN = error_FFNN + (1 / N_test_samples) * (np.linalg.norm(encoded_solutions[sample, :] - encoded_results[sample, :], 2) / np.linalg.norm(encoded_solutions[sample, :], 2))
print('L2 norm error - FFNN = ', error_FFNN)

# Summarize history for loss
plt.plot(history_NN.history['loss'])
plt.plot(history_NN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

