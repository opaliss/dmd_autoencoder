from data_maker import time_stepper


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
