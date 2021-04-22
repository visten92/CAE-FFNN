import tensorflow as tf
import numpy as np
import scipy.io

# Load trained models and test data
data =  scipy.io.loadmat('burgers_test.mat')
test_parameters = data['nu']
test_solutions = data['vu']
solutions_encoder = tf.keras.models.load_model('Encoder_burgers')
solutions_decoder = tf.keras.models.load_model('Decoder_burgers')
neural_network = tf.keras.models.load_model('NN_burgers')
print('models loaded.....')

# Predict the solutions for new parameter values
encoded_results = neural_network(test_parameters)
surrogate_results =  solutions_decoder(encoded_results)
error_tot = 0
N_test_samples = test_parameters.shape[0]
for sample in range(N_test_samples):
    error_tot = error_tot + (1/N_test_samples) * (np.linalg.norm(test_solutions[sample,:,:]  - surrogate_results[sample,:,:], 2 ) / np.linalg.norm(test_solutions[sample,:,:], 2 ))
print('L2 norm error - surrogate model = ', error_tot)
