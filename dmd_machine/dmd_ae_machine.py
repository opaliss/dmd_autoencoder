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
        Amat, y_pred = self.compute_pred_batch_mat(y_data_mat=y)

        # dmd predict loss in latent space.
        pred_loss = self.pred_loss(y_pred, y)

        # ae reconstruction loss.
        ae_loss = self.ae_loss_term(input_data=input, x_ae=x_ae)
    
        return [x_ae, y, dmd_loss, ae_loss, y_pred, Amat, pred_loss]

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
    def weight_manager(aeweights):
        """ autoencoder weights. """
        aeflat = np.array([])

        for jj in range(len(aeweights)):
            aeflat = np.concatenate((aeflat, aeweights[jj].flatten()))

        aenorm = np.linalg.norm(aeflat)
        return aenorm

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
        fro_norm = self.frobenius_norm(loss_mat)  # frobenius norm squared.
        return fro_norm

    @staticmethod
    def frobenius_norm(mat):
        """frobenius norm implements with tensor"""
        return tf.linalg.trace(tf.matmul(mat, tf.transpose(mat)))

    def get_linearity_loss(self, y_data):
        """ return the sum of the dmd linearity loss of each batch.
        :param y_data: encoder output.
        """
        #         for ii in range(y_data.shape[0]):
        #             y_minus = self.y_minus(y_data[ii])
        #             y_plus = self.y_plus(y_data[ii])
        #             dmd_loss += self.dmd_loss(y_minus, y_plus)
        #             dmd_loss/ y_data.shape[0]

        reshape_y = tf.convert_to_tensor(self.windowing(y_data.numpy()), dtype=tf.float32)
        y_minus = self.y_minus(reshape_y)
        y_plus = self.y_plus(reshape_y)
       
        return self.dmd_loss(y_minus, y_plus)

    @staticmethod
    def get_amat(y_data):
        """ Compute DMD Amat:
        X = [x1, x2, x3, .., xm]
        X+ = [x2, x3, x4, .., xm]
        X- = [x1, x2, .., xm-1]
        Amat = X+ * pseudoinverse(X-)

        X - dim (features, timesteps)
        Ex2: n = 2, m = 51
        """
        x_plus = y_data[:, 1:]
        x_minus = y_data[:, :-1]

        return np.matmul(x_plus, np.linalg.pinv(x_minus))

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
        Amat_exp = Amat
        for ii in range(y_data.shape[1]):
            if ii != 0:
                Amat_exp = np.matmul(Amat_exp, Amat)
            y_pred[:, ii] = np.matmul(Amat_exp, y_pred[:, 0])
            
        return Amat, y_pred

    def compute_pred_batch_mat(self, y_data_mat):
        """ compute the y_pred for full batch.
        y_data_mat - dim (batch_size, features, timeseries)
         Ex2: (256, 2, 51)
         batch size is a hyperparam.
         """
        y_data_mat = y_data_mat.numpy()
        y_data = self.windowing(y_data_mat)
        Amat, y_predict = self.get_predicted_y(y_data=y_data)
        return Amat, tf.convert_to_tensor(self.undo_windowing(y_predict))

    @staticmethod
    def windowing(x_mat):
        """ convert (256, 2, 51) --> (512, 51)
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = np.zeros((int(x_mat.shape[0]*x_mat.shape[1]), x_mat.shape[2]))
        for ii in range(x_mat.shape[0]):
            new_mat[2*ii, :] = x_mat[ii, 0, :]
            new_mat[2*ii+1, :] = x_mat[ii, 1, :]
        return new_mat

    @staticmethod
    def undo_windowing(x_mat):
        """ convert (512, 51) --> (256, 2, 51) 
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = np.zeros((int(x_mat.shape[0]/2), 2, x_mat.shape[1]))
        for ii in range(int(x_mat.shape[0]/2)):
            new_mat[ii, 0, :] = x_mat[2*ii, :]
            new_mat[ii, 1, :] = x_mat[2*ii+1, :]
        return new_mat



