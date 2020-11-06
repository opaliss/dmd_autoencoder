import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class Coder(keras.Model):

    def __init__(self, hyp_params, **kwargs):
        super(Coder, self).__init__(**kwargs)

        # Build out the encoder network.
        self.encoder = keras.Sequential()
        self.encoder.add(Input(shape=(hyp_params['phys_dim'],), batch_size=hyp_params['batch_size']))

        for _ in range(hyp_params['num_en_layers']):
            self.encoder.add(Dense(hyp_params['num_en_neurons'],
                                   activation=hyp_params['activation'],
                                   kernel_initializer=hyp_params['weight_initializer'],
                                   kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                   bias_initializer=hyp_params['bias_initializer']))
        self.encoder.add(Dense(hyp_params['latent_dim'],
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l1(0.01),
                               kernel_initializer=hyp_params['weight_initializer'],
                               bias_initializer=hyp_params['bias_initializer']))

        # Build out the decoder network.
        self.decoder = keras.Sequential()
        self.decoder.add(Input(shape=(hyp_params['latent_dim'],), batch_size=hyp_params['batch_size']))
        for _ in range(hyp_params['num_en_layers']):
            self.decoder.add(Dense(hyp_params['num_en_neurons'],
                                   activation=hyp_params['activation'],
                                   kernel_initializer=hyp_params['weight_initializer'],
                                   kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                   bias_initializer=hyp_params['bias_initializer']))

        self.decoder.add(Dense(hyp_params['phys_dim'],
                               activation="linear",
                               kernel_initializer=hyp_params['weight_initializer'],
                               kernel_regularizer=tf.keras.regularizers.l1(0.01),
                               bias_initializer=hyp_params['bias_initializer']))

    def call(self, x):
        y = self.encode(x)
        x_ae = self.decode(y)
        return y, x_ae

    @tf.function
    def encode(self, x):
        y = []
        for tstep in range(x.shape[-1]):
            y.append(self.encoder(x[:, :, tstep]))
        return tf.transpose(y, perm=[1, 2, 0])

    @tf.function
    def decode(self, y):
        x = []
        for tstep in range(y.shape[-1]):
            x.append(self.decoder(y[:, :, tstep]))
        return tf.transpose(x, perm=[1, 2, 0])

    # @tf.function
    # def encode(self, x):
    #     y = tf.Variable(tf.zeros(
    #         shape=(self.batch_size, self.latent_dim, self.num_t_steps), dtype=tf.float32))
    #     for tstep in range(self.num_t_steps):
    #         y[:, :, tstep].assign(self.encoder(x[:, :, tstep]))
    #     return y
    #
    # @tf.function
    # def decode(self, y):
    #     x_ae = tf.Variable(tf.zeros(
    #         shape=(self.batch_size, self.latent_dim, self.num_t_steps), dtype=tf.float32))
    #     for tstep in range(self.num_t_steps):
    #         x_ae[:, :, tstep].assign(self.decoder(y[:, :, tstep]))
    #     return x_ae
    #
    # def decode_predict(self, y_pred):
    #     y_pred_dec = tf.Variable(tf.zeros(
    #         shape=(self.batch_size, self.latent_dim, self.num_t_steps), dtype=tf.float32))
    #     for tstep in range(self.num_t_steps):
    #         self.y_pred_dec[:, :, tstep].assign(self.decoder(y_pred[:, :, tstep]))
    #     return y_pred_dec