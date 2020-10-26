import tensorflow as tf
from tensorflow import keras
from dmd_machine.autoencoder_network import Coder
import numpy as np


class DMDMachine(keras.Model):

    def __init__(self, hyp_params, **kwargs):
        super(DMDMachine, self).__init__(**kwargs)

        self.num_t_steps = hyp_params['num_t_steps']
        self.autoencoder = Coder(hyp_params, dtype=tf.float32)

    def call(self, input):
        """ pass the input to the autoencoder and compute the linearity loss which will be factored in
        to the total Simple Machine loss. """
        # autoencode the entire time series.
        y, x_ae = self.autoencoder(input)

        # compute linearity loss.
        dmd_loss = self.get_linearity_loss(y_data=y)

        # predict 1 step using dmd fit.
        y_pred = self.compute_pred_batch_mat(y_data_mat=y)

        # dmd predict loss in latent space.
        pred_loss = self.pred_loss(y_pred, y)

        # ae reconstruction loss.
        ae_loss = self.ae_loss_term(input_data=input, x_ae=x_ae)
    
        return [x_ae, y, dmd_loss, ae_loss, y_pred, pred_loss]
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder,
                'dec_out': self.dec_output}
    
    @staticmethod
    def pred_loss(latent_data, y_pred):
        """Compute the prediction loss. """
        return tf.reduce_mean(keras.losses.MSE(latent_data, y_pred))
    
    @staticmethod
    def ae_loss_term(input_data, x_ae):
        return tf.reduce_mean(keras.losses.MSE(input_data[:, :, 0], x_ae[:, :, 0]))

    @staticmethod
    def compute_svd(mat_data):
        """ compute the singular value decomposition.
        Note, in numpy, the svd returns: U, S, V(transpose)
        yet in tensorflow, the svd returns: U, S, V"""
        u, s, v = tf.linalg.svd(mat_data, compute_uv=True)
        return u, s, v
        
    @staticmethod    
    def y_plus(y_data):
        """ the encoded output data without the first time series state. """
        return y_data[:, 1:]
    
    @staticmethod
    def y_minus(y_data):
        """ the encoded output data without the last time series state. """
        return y_data[:, :-1]

    def dmd_loss(self, y_minus, y_plus):
        """ dmd loss = || encoder(x+) - (I - V*Vt)||^2 'for' norm."""
        U, S, V = self.compute_svd(y_minus)
        I = tf.eye(V.shape[0])  # identity matrix
        VVt = tf.linalg.matmul(V, tf.transpose(V))  # V * V transpose
        loss_mat = tf.linalg.matmul(y_plus, (I - VVt))
        return self.frobenius_norm(loss_mat)

    @staticmethod
    def frobenius_norm(mat):
        """frobenius norm implements with tensor."""
        return tf.linalg.trace(tf.matmul(mat, tf.transpose(mat)))

    
    def get_linearity_loss(self, y_data):
        """ return the average dmd loss for each initial condition in the batch. 
        :param y_data: encoder output.
        """
        dmd_loss = 0 
        for ii in range(y_data.shape[0]):
            y_minus = self.y_minus(y_data[ii])
            y_plus = self.y_plus(y_data[ii])
            dmd_loss += self.dmd_loss(y_minus, y_plus)
        return dmd_loss/ y_data.shape[0]


    def get_amat(self, y_data):
        """ Compute DMD Amat:
        Amat = X+ * pseudoinverse(X-)
        """
        y_plus = self.y_plus(y_data)
        y_minus = self.y_minus(y_data)
        
        # singular value decomposition.
        u, s, vh = np.linalg.svd(y_minus, full_matrices=False)
        u, vh = np.matrix(u), np.matrix(vh)

        # compute Atilde.
        Atilde = y_plus @ vh.H
        Atilde = Atilde @ np.diag(1. / s)
        Atilde = Atilde @ u.H
        return Atilde

    def get_predicted_y(self, y_data):
        """ Get predicted y_data.
        y1 = A*y0
        y2 = A^2*y0
        y3 = A^3*y0
        ...
        ym = A^m*y0
        """
        y_pred = np.zeros((y_data.shape[0], y_data.shape[1]))
        y_pred[:, 0] = y_data[:, 0]
        Amat = self.get_amat(y_data)
        for ii in range(1, y_data.shape[1]):
            y_pred[:, ii] = Amat**ii @ y_pred[:, 0]
        return  y_pred

    def compute_pred_batch_mat(self, y_data_mat):
        """ compute the y_pred for full batch.
        y_data_mat - dim (batch_size, features, timeseries)
         Ex2: (256, 2, 51)
         batch size is a hyperparam.
         """
        y_data = y_data_mat.numpy()
        # compute the predicted y given each initial condition. 
        y_predict = np.zeros((y_data.shape[0], y_data.shape[1], y_data.shape[2]))
        for ii in range(y_data.shape[0]):
            y_predict[ii, :, :] = self.get_predicted_y(y_data[ii, :, :])
        return tf.convert_to_tensor(y_predict)
            

    @staticmethod
    def reshape(x_mat):
        """ convert (256, 2, 51) --> (512, 51)
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = np.zeros((int(x_mat.shape[0]*x_mat.shape[1]), x_mat.shape[2]))
        for ii in range(x_mat.shape[0]):
            new_mat[2*ii, :] = x_mat[ii, 0, :]
            new_mat[2*ii+1, :] = x_mat[ii, 1, :]
        return new_mat

    @staticmethod
    def undo_reshape(x_mat):
        """ convert (512, 51) --> (256, 2, 51) 
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = np.zeros((int(x_mat.shape[0]/2), 2, x_mat.shape[1]))
        for ii in range(int(x_mat.shape[0]/2)):
            new_mat[ii, 0, :] = x_mat[2*ii, :]
            new_mat[ii, 1, :] = x_mat[2*ii+1, :]
        return new_mat



