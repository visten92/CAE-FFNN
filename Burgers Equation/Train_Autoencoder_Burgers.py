import tensorflow as tf
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split

#load data and set hyperparameters
data =  scipy.io.loadmat('burgers_train.mat')
parameters = data['nu']
solutions = data['vu']
solutions, test_solutions, parameters, test_parameters = train_test_split(solutions, parameters, test_size=0.50, shuffle=False)
latent_space_dimension = 8
epochs = 3000
batch_size = 16
learning_rate = 1e-4

# Convolutional Autoencoder (CAE)
solutions_encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (100,200)),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5,  padding='same', activation='relu'),
    tf.keras.layers.AvgPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.AvgPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_space_dimension, activation='relu'),
])

solutions_decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = latent_space_dimension),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Reshape(target_shape=(25, 32)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.UpSampling1D(size = 2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.UpSampling1D(size = 2),
    tf.keras.layers.Conv1D(filters=200, kernel_size=5,  padding='same')
])
solutions_autoencoder_input = tf.keras.Input(shape = (100,200))
encoded = solutions_encoder(solutions_autoencoder_input)
decoded = solutions_decoder(encoded)
solutions_autoencoder = tf.keras.Model(solutions_autoencoder_input, decoded)
solutions_autoencoder.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
solutions_autoencoder.summary()
history_autoencoder_solutions = solutions_autoencoder.fit(solutions, solutions, validation_data=(test_solutions, test_solutions), batch_size = batch_size, epochs = epochs)

# Save Autoencoder
#solutions_encoder.save('Encoder_burgers')
#solutions_decoder.save('Decoder_burgers')

# CAE L2-norm error
results = solutions_autoencoder(solutions)
test_results = solutions_autoencoder(test_solutions)
solutions =  solutions
test_solutions = test_solutions
error_CAE = 0
N_test_samples = test_parameters.shape[0]
for sample in range(N_test_samples):
    error_CAE = np.linalg.norm(test_solutions[sample, :, :] - solutions_autoencoder(test_solutions)[sample, :, :], 2) / np.linalg.norm(test_solutions[sample, :, :], 2)
print('L2 norm error - CAE = ', error_CAE)

# Summarize history for loss
plt.plot(history_autoencoder_solutions.history['loss'])
plt.plot(history_autoencoder_solutions.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

