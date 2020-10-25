from data.data_maker import time_stepper


class Data(object):

    def __init__(self, x_lower1, x_upper1, x_lower2, x_upper2, n_side, dt, tf, data_type):
        # settings related to dataset
        self.params = dict()
        self.data_val = time_stepper(x_lower1=x_lower1, x_upper1=x_upper1, x_lower2=x_lower2, x_upper2=x_upper2,
                                     n_side=n_side, dt=dt, tf=tf, datatype=data_type)
        n_ic, n_r, n_t = self.data_val.shape
        self.params['data_type'] = data_type
        self.params['num_time_steps'] = n_t
        self.params["num_physical_dim"] = n_r
        self.params['num_initial_conditions'] = n_ic


if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    create_data_1 = False
    create_data_2 = False

    if create_data_1:
        # create the dataset, ex1.
        training_data = Data(x_lower1=-0.5, x_upper1=0.5, x_lower2=-0.5, x_upper2=0.5, n_side=100, dt=0.3, tf=15,
                             data_type="ex1")
        # save the dataset in a pickle file.
        pickle.dump(training_data, open('data/example_1_dataset.pkl', 'wb'))

        # plot the dataset for visualization.
        data = training_data.data_val
        plt.figure(figsize=(15, 8))
        for ii in np.arange(0, 10000, 15):
            x1 = training_data.data_val[ii, 0, :]
            x2 = training_data.data_val[ii, 1, :]
            plt.plot(x1, x2, '-')
        plt.xlabel("x1", fontsize=20)
        plt.ylabel("X2", fontsize=20)
        plt.title("Ex1 dataset", fontsize=20)
        plt.savefig('./data/example1.png')
        plt.show()

    if create_data_2:
        # create the dataset, ex2.
        training_data = Data(x_lower1=-1.8, x_upper1=1.8, x_lower2=-1.2, x_upper2=1.2, n_side=100, dt=0.3, tf=15,
                             data_type="ex2")
        # save the dataset in a pickle file.
        pickle.dump(training_data, open('data/example_2_dataset.pkl', 'wb'))

        # plot the dataset for visualization.
        data = training_data.data_val
        plt.figure(figsize=(15, 8))
        for ii in np.arange(0, 10000, 15):
            x1 = training_data.data_val[ii, 0, :]
            x2 = training_data.data_val[ii, 1, :]
            plt.plot(x1, x2, '-')
        plt.xlabel("x1", fontsize=20)
        plt.ylabel("X2", fontsize=20)
        plt.title("Ex2 dataset", fontsize=20)
        plt.savefig('./data/example2.png')
        plt.show()
