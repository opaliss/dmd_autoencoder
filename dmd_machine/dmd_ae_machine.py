import tensorflow as tf
from tensorflow import keras
from dmd_machine.autoencoder_network import Coder


class DMDMachine(keras.Model):

    def __init__(self, hyp_params, **kwargs):
        super(DMDMachine, self).__init__(**kwargs)

        # autoencoder network.
        self.autoencoder = Coder(hyp_params, dtype=tf.float32)
        # model hyper parameters
        self.num_t_steps = hyp_params['num_t_steps']
        self.batch_size = hyp_params['batch_size']
        self.phys_dim = hyp_params['phys_dim']
        self.window_size = hyp_params['window_size']
        self.latent_dim = hyp_params['latent_dim']

    def call(self, input):
        """ pass the input to the autoencoder and compute the linearity loss which will be factored in
        to the total DMD Machine loss. """
        # autoencode the entire time series.
        y, x_ae = self.autoencoder(x=input)

        # compute linearity loss.
        dmd_loss = self.get_linearity_loss(y_data=y)

        # predict latent space using dmd fit.
        if self.window_size is None:
            y_pred = self.compute_pred_batch_mat(y_data_mat=y)
            
        else:
            y_pred = self.compute_predict_batch_reshape(y_data_mat=y)
            
        # decode predicted latent space.
        y_pred_dec = self.autoencoder.decode(y=y_pred)

        # dmd predict loss in latent space.
        pred_loss = self.pred_loss(y_pred, y)

        # ae reconstruction loss.
        ae_loss = self.ae_loss_term(input_data=input, x_ae=x_ae)

        pred_dec_loss = self.pred_loss_dec(input, y_pred_dec)

        return [x_ae, y, dmd_loss, ae_loss, y_pred, pred_loss, y_pred_dec, pred_dec_loss]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.autoencoder.encoder,
                'decoder': self.autoencoder.decoder}

    @staticmethod
    def pred_loss_dec(x_input, y_pred_dec):
        """ compute the prediction decoded loss. """
        return tf.reduce_mean(keras.losses.MSE(x_input, y_pred_dec))

    @staticmethod
    def pred_loss(latent_data, y_pred):
        """Compute the prediction loss. """
        return tf.reduce_mean(keras.losses.MSE(latent_data, y_pred))

    @staticmethod
    def ae_loss_term(input_data, x_ae):
        """ Compute Autoencoder loss, just comparing the initial condition."""
        return tf.reduce_mean(keras.losses.MSE(input_data[:, :, 0], x_ae[:, :, 0]))

    @staticmethod
    def compute_svd(mat_data):
        """ compute the singular value decomposition.
        Note, in numpy, the svd returns: U, S, V(transpose)
        yet in tensorflow, the svd returns: U, S, V"""
        s, u, v = tf.linalg.svd(mat_data, compute_uv=True)
        return s, u, v

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
        Vt = self.compute_svd(y_minus)[-1]
        I = tf.eye(Vt.shape[0])  # identity matrix
        # TODO QUICK FIX V tranpose V
        VVt = tf.linalg.matmul(Vt, tf.transpose(Vt))  # V * V transpose
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
        for ii in range(self.batch_size):
            y_minus = self.y_minus(y_data[ii])
            y_plus = self.y_plus(y_data[ii])
            dmd_loss += self.dmd_loss(y_minus, y_plus)
        return dmd_loss / self.batch_size

    def get_linearity_loss_reshape(self, y_data):
        """ return the average dmd loss for each initial condition in the batch.
        :param y_data: encoder output.
        """
        y_data = self.reshape(y_data)
        y_minus = self.y_minus(y_data)
        y_plus = self.y_plus(y_data)
        return self.dmd_loss(y_minus, y_plus)

    def get_amat(self, y_data):
        """ Compute DMD A matrix by the singular value decomposition. """
        y_plus = self.y_plus(y_data)
        y_minus = self.y_minus(y_data)

        # compute A using the pseudoinverse.
        return y_plus @ tf.linalg.pinv(y_minus)

    def get_predicted_y(self, y_data):
        """ Get predicted y_data.
        y1 = A*y0
        y2 = A^2*y0
        y3 = A^3*y0
        ...
        ym = A^m*y0
        """
        y_pred = tf.Variable(tf.zeros(shape=(y_data.shape[0], y_data.shape[1]), dtype=tf.float32))
        y_pred[:, 0].assign(y_data[:, 0])
        A = self.get_amat(y_data)
        A_exp = tf.eye(A.shape[0])

        for ii in range(1, self.num_t_steps):
            A_exp = tf.matmul(A, A_exp)
            y_pred[:, ii].assign(tf.tensordot(A_exp, y_pred[:, 0], axes=1))
        return y_pred

    def compute_pred_batch_mat(self, y_data_mat):
        """ compute the y_pred for full batch.
        y_data_mat - dim (batch_size, features, timeseries)
         Ex2: (256, 2, 51)
         batch size is a hyperparam.
         """
        # compute the predicted y given each initial condition. 
        y_predict = tf.Variable(tf.zeros(shape=(self.batch_size, self.latent_dim,
                                                self.num_t_steps), dtype=tf.float32))

        for ii in range(0, self.batch_size):
            y_predict[ii, :, :].assign(self.get_predicted_y(y_data_mat[ii, :, :]))

        return y_predict

    def compute_predict_batch_reshape(self, y_data_mat):
        # compute the predicted y given each initial condition.
        y_predict = tf.Variable(tf.zeros(shape=(self.batch_size, self.latent_dim,
                                                self.num_t_steps), dtype=tf.float32))

        # reshape y matrix to be batch_size x phys_dim by t_steps.
        y_reshape = self.reshape(y_data_mat)

        # find the big A mat (512 x 512)
        pred_y = self.get_predicted_y(y_reshape)

        # undo reshape- back to batch_size, phys_dim, t_steps.
        undo_reshape = self.undo_reshape(pred_y)

        # save and return.
        y_predict.assign(undo_reshape)

        return y_predict

    def reshape(self, x_mat):
        """ convert (256, 2, 51) --> (512, 51)
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = tf.Variable(tf.zeros(
            shape=(int(self.batch_size * self.latent_dim), self.num_t_steps), dtype=tf.float32))
        for ii in range(int(self.batch_size)):
            for jj in range(int(self.latent_dim)):
                new_mat[int(self.latent_dim * ii + jj), :].assign(x_mat[ii, jj, :])
        return new_mat

    def undo_reshape(self, x_mat):
        """ convert (512, 51) --> (256, 2, 51) 
        only works if 2 latent dim - good for ex1 and ex2 datasets. """
        new_mat = tf.Variable(tf.zeros(
            shape=(self.batch_size, self.latent_dim, self.num_t_steps), dtype=tf.float32))
        for ii in range(int(self.batch_size)):
            for jj in range(int(self.latent_dim)):
                new_mat[ii, jj, :].assign(x_mat[int(self.latent_dim * ii + jj), :])
        return new_mat

    @staticmethod
    def windowing(mat):
        new_mat = tf.Variable(tf.zeros(
            shape=(mat.shape[0] * mat.shape[1], mat.shape[2]), dtype=tf.float32))
        for ii in range(int(mat.shape[0] * mat.shape[1])):
            new_mat[2 * ii, :].assign(mat[ii, 0, :])
            new_mat[2 * ii + 1, :].assign(mat[ii, 1, :])
        return new_mat

    @staticmethod
    def undo_windowing(mat):
        new_mat = tf.Variable(tf.zeros(
            shape=(int(mat.shape[0] / 2), 2, mat.shape[2]), dtype=tf.float32))
        for ii in range(int(mat.shape[0] / 2)):
            new_mat[ii, 0, :].assign(mat[2 * ii, :])
            new_mat[ii, 1, :].assign(mat[2 * ii + 1, :])
        return new_mat
