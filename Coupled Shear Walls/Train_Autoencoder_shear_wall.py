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
latent_space_dimension = 64
epochs = 500
batch_size = 8
learning_rate = 1e-4

# Scale the data
s_min = np.abs(solutions.min())
s_max = np.abs(solutions.max())
if (s_min>s_max):
    s_max = s_min
np.save('s_abs_max.npy', s_max)
solutions = solutions / s_max
test_solutions = test_solutions / s_max

# Convolutional Autoencoder (CAE)
solutions_encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (600,1966)),
    tf.keras.layers.Conv1D(filters=512, kernel_size=5,  padding='same', activation='relu'),
    tf.keras.layers.AvgPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.AvgPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.AvgPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_space_dimension, activation='relu'),
])
solutions_decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = latent_space_dimension),
    tf.keras.layers.Dense(4800, activation='relu'),
    tf.keras.layers.Reshape(target_shape=(75, 64)),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.UpSampling1D(size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=5,  padding='same', activation='relu'),
    tf.keras.layers.UpSampling1D(size = 2),
    tf.keras.layers.Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.UpSampling1D(size=2),
    tf.keras.layers.Conv1D(filters=1966, kernel_size=5,  padding='same')
])
solutions_autoencoder_input = tf.keras.Input(shape = (600,1966))
encoded = solutions_encoder(solutions_autoencoder_input)
decoded = solutions_decoder(encoded)
solutions_autoencoder = tf.keras.Model(solutions_autoencoder_input, decoded)
solutions_autoencoder.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')
solutions_autoencoder.summary()
start_time =  time.time()
history_autoencoder_solutions = solutions_autoencoder.fit(solutions, solutions, batch_size = batch_size, epochs = epochs)
training_time = time.time() - start_time
print('Training time: ',   training_time, ' sec')

# CAE L2-norm error
test_results = s_max * solutions_autoencoder(test_solutions)
test_solutions = s_max * test_solutions
N_test_samples = test_parameters.shape[0]
error_autoencoder = 0
for sample in range(N_test_samples):
    error_autoencoder = error_autoencoder + (1 / N_test_samples) * np.linalg.norm(test_solutions[sample, :, :] - test_results[sample, :, :], 2) / np.linalg.norm(test_solutions[sample, :, :], 2)
print('L2 norm error - Autoencoder = ', error_autoencoder)

# Save Autoencoder
#solutions_encoder.save('Encoder_shear_wall')
#solutions_decoder.save('Decoder_shear_wall')

# Summarize history for loss
plt.plot(history_autoencoder_solutions.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

