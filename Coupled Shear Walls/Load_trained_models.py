#Uncomment the next two lines in order to use CPU instead of GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import scipy.io
import time

# Load trained models and test data
s_abs_max = np.load('s_abs_max.npy')
p_min = np.load('p_min.npy')
p_max= np.load('p_max.npy')
solutions_encoder = tf.keras.models.load_model('Encoder_shear_wall')
solutions_decoder = tf.keras.models.load_model('Decoder_shear_wall')
neural_network = tf.keras.models.load_model('NN_shear_wall')
print('models loaded....................................')
test_data = scipy.io.loadmat('data_test.mat')
test_parameters = test_data['batchInputRandomParameters']
test_parameters = (test_parameters - p_min) / (p_max - p_min)
test_solutions = test_data['solutions'].T
N_test_samples = test_parameters.shape[0]
test_solutions = np.reshape(test_solutions, (N_test_samples, 600, 1966))
del test_data

# Predict the solutions for new parameter values
start_time = time.time()
encoded_results = neural_network(test_parameters)
surrogate_results = s_abs_max * solutions_decoder(encoded_results)
end_time = time.time() - start_time
print('Surrogate model time: ',   end_time, ' sec')
error_tot = 0
for sample in range(N_test_samples):
    error_tot = error_tot + (1/N_test_samples) * (np.linalg.norm(test_solutions[sample,:,:]  - surrogate_results[sample,:,:], 2 ) / np.linalg.norm(test_solutions[sample,:,:], 2 ))
print('L2 norm error - surrogate model = ', error_tot)
