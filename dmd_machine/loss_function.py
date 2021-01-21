""" Dynamic Mode Decomposition auto-encoder loss module. """
from tensorflow import keras
import tensorflow as tf


class LossFunction(keras.losses.Loss):

    def __init__(self, hyp_params, **kwargs):
        super(LossFunction, self).__init__(**kwargs)
        # auto-encoder loss of reconstruction: initial condition comparison.
        self.ae_loss = tf.constant(0., dtype=tf.float32)

        #  dmd loss || encoder(x+) - (I - V*Vt)|| 'for' norm.
        self.dmd_loss = tf.constant(0., dtype=tf.float32)

        # linearity loss x_data - y_pred_dec.
        self.linearity_loss = tf.constant(0., dtype=tf.float32)

        # prediction loss of latent space using dmd reconstruction.
        self.predict_loss = tf.constant(0., dtype=tf.float32)

        # initialize the weights of each loss component.
        self.c1 = tf.constant(hyp_params['c1'], dtype=tf.float32)
        self.c2 = tf.constant(hyp_params['c2'], dtype=tf.float32)
        self.c3 = tf.constant(hyp_params['c3'], dtype=tf.float32)

    def call(self, x_data, my_machine_output):
        """
            x_ae : encoder/decoder check only for the initial condition. 
            dmd_loss: dmd || encoder(x+) - (I - V*Vt)||^2 'for' norm. 
        """
        x_ae = my_machine_output[0]
        y_data = my_machine_output[1]
        dmd_loss = my_machine_output[2]
        y_pred = my_machine_output[4]
        y_pred_dec = my_machine_output[6]

        # MSE keras function for the autoencoder reconstruction.
        self.ae_loss = tf.reduce_mean(keras.losses.MSE(x_data, x_ae))

        # dmd loss
        self.dmd_loss = tf.constant(dmd_loss, dtype=tf.float32)

        # linearity loss.
        self.linearity_loss = tf.reduce_mean(keras.losses.MSE(y_pred_dec, x_data))

        # prediction loss, MSE between latent space and dmd reconstructed dataset.
        self.predict_loss = tf.reduce_mean(keras.losses.MSE(y_pred, y_data))

        return self.c1 * self.ae_loss + self.c2 * self.linearity_loss + self.c3*self.predict_loss + self.dmd_loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'ae_loss': self.ae_loss,
                "linearity_loss": self.linearity_loss,
                "pred_loss": self.predict_loss,
                "dmd_loss": self.dmd_loss}