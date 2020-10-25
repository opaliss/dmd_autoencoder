""" This module is a collection of functions which return statistics about each iteration during training. """
import numpy as np


def print_status_bar(iteration, total, loss_train, loss_test, log_file_path=None):
    """
    Prints on screen and can be written in a log file.
    :param log_file_path: str with log file path. Example: "Results/oct21_ex2/log.txt"
    :param iteration: epoch number.
    :param total: total number of epoches.
    :param loss_train: training cumulative loss of the last batch passed.
    :param loss_test: testing cumulative loss of the last batch passed.
    :return: print statement with current statistics.
    """
    string = str(iteration) + "/" + str(total) + " ,loss_train: " + str(loss_train.numpy()) + " , loss_test: " + \
             str(loss_test.numpy())

    if log_file_path is not None:
        with open(log_file_path, "w") as log:
            log.write(string)
            log.write("\n")
    print(string)
