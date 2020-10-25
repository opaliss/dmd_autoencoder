"""" This module will train dmd autoencoder on pendulum dataset. """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import SimpleMachine as sm
import SimpleLoss as sl
from data import Data as dt
import pickle
from tensorflow.keras.models import model_from_json
from return_stats import print_status_bar
from create_plots import *
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['figure.facecolor'] = 'white'


new_data = False  # if False, it will read from pickle file instead of building the data.
if new_data:
    training_data = dt.Data(x_lower1=-1.8, x_upper1=1.8, x_lower2=-1.2, x_upper2=1.2, n_side=100, dt=0.3, tf=15,
                            data_type="ex2")
    pickle.dump(training_data, open('example_2_training_data_51_traj.pkl', 'wb'))
else:
    training_data = pickle.load(open('example_2_training_data_51_traj.pkl', 'rb'))

data = training_data.data_val

# Network Hyper Parameters
hyp_params = dict()
hyp_params['num_t_steps'] = training_data.params['num_time_steps']
hyp_params['num_phys_dims'] = training_data.params["num_physical_dim"]
hyp_params['num_init_conds'] = training_data.params['num_initial_conditions']
hyp_params['batch_size'] = 256  # MAJOR PARAMETER CHOICE
hyp_params['num_epochs'] = 200  # MAJOR PARAMETER CHOICE

# Encoding/Decoding Layer Parameters
hyp_params['num_en_layers'] = 2  # MAJOR PARAMETER CHOICE
hyp_params['num_en_neurons'] = 80  # MAJOR PARAMETER CHOICE
hyp_params['latent_dim'] = 2

hyp_params['activation'] = 'elu'
hyp_params['weight_initializer'] = 'he_uniform'
hyp_params['bias_initializer'] = 'he_uniform'
hyp_params['regfac'] = 3e-3

hyp_params['c1'] = 1  # coefficient autoencoder loss.
hyp_params['c2'] = 1  # coefficient of dmd loss.
hyp_params['c3'] = 1  # coefficient of pred loss.

save_folder = "AeEx2_oct12"  # save results in the folder " Results/save_folder"-
                                # including loss curves and plot latent data.

# Build the AutoEncoder with affiliated loss function and optimizer.
input_data = training_data.data_val
all_data = tf.data.Dataset.from_tensor_slices(input_data)

# shuffle the dataset and then divide to training vs testing data sets. 80% training .20% testing.
all_data_shuffle = all_data.shuffle(hyp_params['num_init_conds'], seed=42)

data_train = all_data_shuffle.take(int(0.8 * hyp_params['num_init_conds']))
data_test = all_data_shuffle.skip(int(0.8 * hyp_params['num_init_conds']))

hyp_params['num_init_conds_training'] = int(0.8 * hyp_params['num_init_conds'])
hyp_params['num_init_conds_test'] = hyp_params['num_init_conds'] - hyp_params['num_init_conds_training']

myMachine = sm.SimpleMachine(hyp_params)
myLoss = sl.SimpleLossFunction(hyp_params)

# Learning rate initialization
lr0 = 3e-3  # MAJOR PARAMETER CHOICE
cnt = 0
estep = 30  # MAJOR PARAMETER CHOICE

# clear previous run session.
tf.keras.backend.clear_session()

# save loss results.
train_loss_results = []
test_loss_results = []

train_dmd_loss = []
test_dmd_loss = []

train_ae_loss = []
test_ae_loss = []

train_pred_loss = []
test_pred_loss = []

epoch = 0

while epoch < (hyp_params['num_epochs']):
    # save the total loss of the training data and testing data.
    epoch_loss_avg_train = tf.keras.metrics.Mean()
    epoch_loss_avg_test = tf.keras.metrics.Mean()

    # keep track of individual losses as well, aka dmd loss and ae loss.
    epoch_loss_dmd_train = tf.keras.metrics.Mean()
    epoch_loss_dmd_test = tf.keras.metrics.Mean()

    epoch_loss_ae_train = tf.keras.metrics.Mean()
    epoch_loss_ae_test = tf.keras.metrics.Mean()

    epoch_loss_pred_train = tf.keras.metrics.Mean()
    epoch_loss_pred_test = tf.keras.metrics.Mean()

    # Build out the batches within a given epoch.
    train_batch = data_train.shuffle(hyp_params['num_init_conds_training'], seed=42).batch(hyp_params["batch_size"],
                                                                                           drop_remainder=True)

    test_batch = data_test.shuffle(hyp_params['num_init_conds_test'], seed=42).batch(hyp_params["batch_size"],
                                                                                     drop_remainder=True)

    # Learning rate scheduling plan.  See Ch. 11 of O'Reilly.
    if epoch % estep == 0:
        hyp_params['lr'] = (.2 ** (cnt)) * lr0
        cnt += 1
        myoptimizer = tf.keras.optimizers.Adam(hyp_params['lr'])

    # Iterate through all the batches within an epoch.
    for batch_training_data in train_batch:
        # normalize batch

        # Build terms that we differentiate (i.e. loss) and that we differentiate with respect to.
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions_train = myMachine(batch_training_data)
            ae_loss = predictions_train[3]
            dmd_loss = predictions_train[2]
            pred_loss = predictions_train[6]

            loss_train = myLoss(batch_training_data, predictions_train)

            if epoch % 5 == 0:
                if abs((tf.reduce_min(predictions_train[1][:, 0, :]) - tf.reduce_max(
                        predictions_train[1][:, 0, :])).numpy()) < 0.3 \
                        or abs((tf.reduce_min(predictions_train[1][:, 1, :]) - tf.reduce_max(
                    predictions_train[1][:, 1, :])).numpy()) < 0.3 \
                        or abs((tf.reduce_min(predictions_train[0][:, 0, :]) - tf.reduce_max(
                    predictions_train[0][:, 0, :])).numpy()) < 0.3 \
                        or abs((tf.reduce_min(predictions_train[0][:, 1, :]) - tf.reduce_max(
                    predictions_train[0][:, 1, :])).numpy()) < 0.3:
                    print("restart initializers. ")
                    epoch = -1
                    hyp_params['c2'] = 1
                    tf.keras.backend.clear_session()
                    myMachine = sm.SimpleMachine(hyp_params)
                    myLoss = sl.SimpleLossFunction(hyp_params)

        # Compute gradients and then apply them to update weights within the Neural Network
        gradients = tape.gradient(loss_train, myMachine.trainable_variables)
        myoptimizer.apply_gradients([
            (grad, var)
            for (grad, var) in zip(gradients, myMachine.trainable_variables)
            if grad is not None
        ])

        # Keep track of the loss after each batch.
        epoch_loss_avg_train.update_state(loss_train)
        epoch_loss_ae_train.update_state(ae_loss)
        epoch_loss_dmd_train.update_state(dmd_loss)
        epoch_loss_pred_train.update_state(pred_loss)

    for batch_test_data in test_batch:
        predictions_test = myMachine(batch_test_data)
        dmd_test = predictions_test[2]
        ae_test = predictions_test[3]
        pred_test = predictions_test[6]

        loss_test = myLoss(batch_test_data, predictions_test)

        epoch_loss_avg_test.update_state(loss_test)
        epoch_loss_ae_test.update_state(ae_test)
        epoch_loss_dmd_test.update_state(dmd_test)
        epoch_loss_pred_test.update_state(pred_test)

    train_loss_results.append(epoch_loss_avg_train.result())
    test_loss_results.append(epoch_loss_avg_test.result())

    train_dmd_loss.append(epoch_loss_dmd_train.result())
    train_ae_loss.append(epoch_loss_ae_train.result())
    train_pred_loss.append(epoch_loss_pred_train.result())

    test_dmd_loss.append(epoch_loss_dmd_test.result())
    test_ae_loss.append(epoch_loss_ae_test.result())
    test_pred_loss.append(epoch_loss_pred_test.result())

    if epoch % 15 == 0:
        # save plots in results folder. Plot the latent space, ae_reconstruction, and input_batch.
        create_plots(batch_training_data, predictions_train, hyp_params, epoch, train_loss_results, save_folder, \
                     "train")
        create_plots(batch_test_data, predictions_test, hyp_params, epoch, test_loss_results, save_folder, "test")

        plot_dmd_eigs(predictions_train[5], epoch, save_folder, "train")
        plot_dmd_eigs(predictions_test[5], epoch, save_folder, "test")

    if epoch % 10 == 0:
        # plot latent, input and reconstructed ae latest batch data.
        try:
            print_status_bar(epoch, hyp_params["num_epochs"], epoch_loss_avg_train.result(), \
                             epoch_loss_avg_test.result())
        except:
            print("print status failed.")

    if epoch % 50 == 0 and epoch != 0:
        # plot loss curves.
        create_plots_of_loss(train_dmd_loss, train_ae_loss, test_dmd_loss, test_ae_loss, train_pred_loss, \
                             test_pred_loss, myLoss.c1, myLoss.c2, myLoss.c3, epoch, save_folder)

        # save current machine.
        try:
            myMachine.autoencoder.save("my_model_Ex1_oct23", save_format='save_weights')
        except Exception:
            print("was not able to save weights. ")

    epoch += 1

# final summary of the network, again for diagnostic purposes.
myMachine.summary()