""" This module is a collection of functions which return statistics about each iteration during training. """
import numpy as np
import json
import os
import pickle


def print_status_bar(iteration, total, loss_train, loss_test, run_time, log_file_path=None):
    """
    Prints on screen and can be written in a log file.
    :param run_time: time it took to run the latest epoch.
    :param log_file_path: str with log file path. Example: "Results/oct21_ex2/log.txt"
    :param iteration: epoch number.
    :param total: total number of epoches.
    :param loss_train: training cumulative loss of the last batch passed.
    :param loss_test: testing cumulative loss of the last batch passed.
    :return: print statement with current statistics.
    """
    string = str(iteration) + "/" + str(total) + " ,loss_train: " + str(loss_train.numpy()) + " , loss_test: " + \
             str(loss_test.numpy()) + ", run time = " + str(run_time) + " sec."

    if log_file_path is not None:
        with open(log_file_path, "a") as log:
            log.write(string)
            log.write("\n")

    print(string)


def save_hyp_params_in_json(hyp_params, json_file_path):
    """
    save the machine hyper parameters in a json file.
    :param hyp_params: dict with dataset and machine free parameters.
    :param json_file_path: string - path of to file.
    :return: None
    """
    with open(json_file_path, 'w') as file:
        json.dump(hyp_params, file, indent=4)


def save_loss_curves(train_loss_results, test_loss_results, train_dmd_loss, test_dmd_loss, train_ae_loss,
                     test_ae_loss, train_pred_loss, test_pred_loss, file_path):
    """
    Save loss curves in individual pickle files which can be loaded and analyzed after training.
    :param file_path: string- path to save all the following files.
    :param train_loss_results: list
    :param test_loss_results: list
    :param train_dmd_loss: list
    :param test_dmd_loss: list
    :param train_ae_loss: list
    :param test_ae_loss: list
    :param train_pred_loss: list
    :param test_pred_loss: list
    :return: pickle files in Results folder.
    """
    # cumulative loss with each epoch.
    pickle.dump(train_loss_results, open(os.path.join(file_path, "train_loss_results.pkl"), "wb"))
    pickle.dump(test_loss_results, open(os.path.join(file_path, "test_loss_results.pkl"), "wb"))

    # dmd loss with each epoch.
    pickle.dump(train_dmd_loss, open(os.path.join(file_path, "train_dmd_loss.pkl"), "wb"))
    pickle.dump(test_dmd_loss, open(os.path.join(file_path, "test_dmd_loss.pkl"), "wb"))

    # auto-encoder loss with each epoch.
    pickle.dump(train_ae_loss, open(os.path.join(file_path, "train_ae_loss.pkl"), "wb"))
    pickle.dump(test_ae_loss, open(os.path.join(file_path, "test_ae_loss.pkl"), "wb"))

    # prediction loss with each epoch.
    pickle.dump(train_pred_loss, open(os.path.join(file_path, "train_pred_loss.pkl"), "wb"))
    pickle.dump(test_pred_loss, open(os.path.join(file_path, "test_pred_loss.pkl"), "wb"))


def create_folder(directory):
    """ a general function to create a new folder given its directory path."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            return True
    except OSError:
        return 'Error: Creating directory. ' + directory


def create_new_folders(folder_name):
    """
    create a new folder to save results located in "Results", including "Loss", "Test" and "Train" folder.
    :param folder_name: string - name of new folder.
    """
    parent_path = os.path.join("results")
    current_path = os.path.join(parent_path, folder_name)

    # make folder path.
    if create_folder(current_path):
        # make Loss+Test+Train folder.
        create_folder(os.path.join(current_path, "Loss"))
        create_folder(os.path.join(current_path, "Test"))
        create_folder(os.path.join(current_path, "Train"))

