import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt


def new_plot_model(test_run, random_batch):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 18))

    rect = fig.patch
    rect.set_facecolor("white")

    observed_data = random_batch
    for ii in range(0, observed_data.shape[0]):
        x1 = observed_data[ii, 0, :]
        x2 = observed_data[ii, 1, :]
        ax[0][0].plot(x1, x2, '-')
    ax[0][0].grid()
    ax[0][0].set_xlabel("x1", fontsize=15)
    ax[0][0].set_ylabel("x2", fontsize=15)
    ax[0][0].set_title("Batch Data", fontsize=15)

    ae_data = test_run[0].numpy()
    for ii in range(0, ae_data.shape[0]):
        x1 = ae_data[ii, 0, :]
        x2 = ae_data[ii, 1, :]
        ax[0][1].plot(x1, x2, '-')
    ax[0][1].grid()
    ax[0][1].set_xlabel("x1", fontsize=15)
    ax[0][1].set_ylabel("x2", fontsize=15)
    ax[0][1].set_title("Autoencoder reconsturction= " + str(np.log10(test_run[3].numpy())), fontsize=15)

    modeled_data = test_run[1].numpy()
    for ii in range(0, modeled_data.shape[0]):
        modeled_rot = modeled_data[ii, :, :]
        x1 = modeled_rot[0]
        x2 = modeled_rot[1]
        ax[1][0].plot(x1, x2, '-')
    ax[1][0].grid()
    ax[1][0].set_xlabel("x1", fontsize=15)
    ax[1][0].set_ylabel("x2", fontsize=15)
    ax[1][0].set_title("Latent space, DMD loss = " + str(np.log10(test_run[2].numpy())), fontsize=15)

    pred_data = test_run[4].numpy()
    for ii in range(0, pred_data.shape[0]):
        modeled_rot = pred_data[ii, :, :]
        x1 = modeled_rot[0]
        x2 = modeled_rot[1]
        ax[1][1].plot(x1, x2, '-')
    ax[1][1].grid()
    ax[1][1].set_xlabel("x1", fontsize=15)
    ax[1][1].set_ylabel("x2", fontsize=15)
    ax[1][1].set_title("Latent space predicted, dmd reconstruction loss = " + str(np.log10(test_run[6].numpy())), \
                       fontsize=15)

    ax[0][0].axis("equal")
    ax[0][1].axis("equal")
    ax[1][0].axis("equal")
    ax[1][1].axis("equal")



def plot_dmd_eigs(Amat, epoch, save_folder, data_type):
    """ Plot the eigenvalues of the dmd amat to show stability. """
    # convert Amat to be numpy datatype.
    # Amat = Amat.numpy() - currently already a numpy type.

    fig, ax = plt.subplots(figsize=(15, 15))

    rect = fig.patch
    rect.set_facecolor("white")

    # plot unit circle.
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), "b", label="unit circle")

    # compute the eigenvalues.
    w, v = np.linalg.eig(Amat)

    for ii in range(len(w)):
        ax.scatter(w.real, w.imag)

    ax.set_title("DMD eigenvalues, epoch = " + str(epoch), fontsize=20)
    ax.set_ylabel("Im", fontsize=20)
    ax.set_xlabel("Re", fontsize=20)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)

    plt.axis("equal")
    plt.legend()

    if data_type == "train":
        train_title = "training_eig_" + str(epoch) + "epoch"
        plt.savefig('./Results/' + str(save_folder) + '/Train/' + train_title + '.png', facecolor=fig.get_facecolor())

    if data_type == "test":
        train_title = "test_eig_" + str(epoch) + "epoch"
        plt.savefig('./Results/' + str(save_folder) + '/Test/' + train_title + '.png', facecolor=fig.get_facecolor())


def create_plots(batch_training_data, predictions_train, hyp_params, epoch, train_loss_results, save_folder,
                 data_type="train"):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(25, 18))

    rect = fig.patch
    rect.set_facecolor("white")

    observed_data = batch_training_data
    for ii in range(0, observed_data.shape[0]):
        x1 = observed_data[ii, 0, :]
        x2 = observed_data[ii, 1, :]
        ax[0][0].plot(x1, x2, '-')
    ax[0][0].grid()
    ax[0][0].set_xlabel("x1")
    ax[0][0].set_ylabel("x2")
    if data_type == "train":
        ax[0][0].set_title("Training Data")
    if data_type == "test":
        ax[0][0].set_title("Test Data")

    ae_data = predictions_train[0].numpy()
    for ii in range(0, ae_data.shape[0]):
        x1 = ae_data[ii, 0, :]
        x2 = ae_data[ii, 1, :]
        ax[0][1].plot(x1, x2, '-')
    ax[0][1].grid()
    ax[0][1].set_xlabel("x1")
    ax[0][1].set_ylabel("x2")
    ax[0][1].set_title("Autoencoder reconsturction= " + str(np.log10(predictions_train[3].numpy())))

    modeled_data = predictions_train[1].numpy()
    for ii in range(0, modeled_data.shape[0]):
        modeled_rot = modeled_data[ii, :, :]
        x1 = modeled_rot[0]
        x2 = modeled_rot[1]
        ax[1][0].plot(x1, x2, '-')
    ax[1][0].grid()
    ax[1][0].set_xlabel("x1")
    ax[1][0].set_ylabel("x2")
    ax[1][0].set_title("Latent space, DMD loss = " + str(np.log10(predictions_train[2].numpy())))

    pred_data = predictions_train[4].numpy()
    for ii in range(0, pred_data.shape[0]):
        modeled_rot = pred_data[ii, :, :]
        x1 = modeled_rot[0]
        x2 = modeled_rot[1]
        ax[1][1].plot(x1, x2, '-')
    ax[1][1].grid()
    ax[1][1].set_xlabel("x1")
    ax[1][1].set_ylabel("x2")
    ax[1][1].set_title(
        "Latent space predicted, dmd reconstruction loss = " + str(np.log10(predictions_train[6].numpy())))

    fig.suptitle("Epoch: {}/{}, Learn Rate: {}, Loss: {:.3f}".format(epoch,
                                                                     hyp_params['num_epochs'],
                                                                     hyp_params['lr'],
                                                                     np.log10(train_loss_results[-1])))
    if data_type == "train":
        train_title = "training_data_at_" + str(epoch) + "epoch"
        plt.savefig('./Results/' + str(save_folder) + '/Train/' + train_title + '.png', facecolor=fig.get_facecolor())

    if data_type == "test":
        train_title = "test_data_at_" + str(epoch) + "epoch"
        plt.savefig('./Results/' + str(save_folder) + '/Test/' + train_title + '.png', facecolor=fig.get_facecolor())

    plt.close()


def create_plots_of_loss(dmd_loss_vec_train, ae_loss_vec_train, dmd_loss_vec_test, ae_loss_vec_test,
                         pred_loss_vec_train, pred_loss_vec_test, c1, c2, c3, epoch, save_folder):
    # create 4 plots of the ae and dmd loss.
    # save to folder named in line 30.
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 21))

    rect = fig.patch
    rect.set_facecolor("white")

    ax[0].plot(np.arange(len(dmd_loss_vec_train)), np.log10(dmd_loss_vec_train), label="training batch")
    ax[0].plot(np.arange(len(dmd_loss_vec_test)), np.log10(dmd_loss_vec_test), label="testing batch")
    ax[0].set_xlabel("# epoch")
    ax[0].set_ylabel("log10(loss)")
    ax[0].set_title("DMD loss training data, weight = " + str(c2.numpy()))

    ax[1].plot(np.arange(len(ae_loss_vec_train)), np.log10(ae_loss_vec_train), label="AE loss")
    ax[1].plot(np.arange(len(ae_loss_vec_test)), np.log10(ae_loss_vec_test), label="AE loss")
    ax[1].set_xlabel("# epoch")
    ax[1].set_ylabel("log10(loss)")
    ax[1].set_title("AE loss training data, weight = " + str(c1.numpy()))

    ax[2].plot(np.arange(len(pred_loss_vec_train)), np.log10(pred_loss_vec_train), label="Pred loss")
    ax[2].plot(np.arange(len(pred_loss_vec_test)), np.log10(pred_loss_vec_test), label="Pred loss")
    ax[2].set_xlabel("# epoch")
    ax[2].set_ylabel("log10(loss)")
    ax[2].set_title("Pred loss training data, weight = " + str(c3.numpy()))

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.suptitle("Loss curve for epoch = " + str(epoch))

    train_title = "loss_curve_at_" + str(epoch) + "epoch"
    plt.savefig('./Results/' + str(save_folder) + '/Loss/' + train_title + '.png', facecolor=fig.get_facecolor())
    plt.close()
