""" In this module, there are dmd functions. """
import numpy as np
from numpy import matlib

# Perform DMD method
def dmd_fit(x_mat):
    x_minus = x_mat[:, :-1]
    x_plus = x_mat[:, 1:]

    # singular value decomposition.
    u, s, vh = np.linalg.svd(x_minus, full_matrices=False)
    u, vh = np.matrix(u), np.matrix(vh)

    # compute Atilde.
    Atilde = x_plus @ vh.H
    Atilde = Atilde @ np.diag(1. / s)
    Atilde = Atilde @ u.H

    # compute the eigenvalues and eigenvectors of Atilde.
    eigs, eig_vecs = np.linalg.eig(Atilde)

    # find b.
    b = np.linalg.solve(eig_vecs, x_mat[:, 0])

    # reconstruct dmd.
    dmd_rec = np.zeros((x_mat.shape[0], x_mat.shape[1]))

    dmd_rec[:, 0] = x_mat[:, 0]
    for ii in range(1, x_mat.shape[1]):
        dmd_rec[:, ii] = eig_vecs @ np.exp(np.log(np.diag(eigs)) * ii) @ b

    dmd_rec = dmd_rec.real
    return [eigs, eig_vecs, b, dmd_rec]


if __name__ == "__main__":
    from dmd_machine.dmd_ae_machine import DMDMachine
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['figure.facecolor'] = 'white'

    random_batch = pickle.load(open('./data/ex2_random_batch.pkl', 'rb'))
    random_batch = random_batch.numpy()

    batch_reshape = DMDMachine.reshape(random_batch)
    batch_reshape.shape
    dmd_fit(batch_reshape)
