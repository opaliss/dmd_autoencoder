import pickle
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================
class DataMaker(object):

    def __init__(self, x_lower1, x_upper1, x_lower2, x_upper2, n_ic, dt, tf, data_type):
        # settings related to dataset
        self.params = dict()
        if data_type == 'discrete':
            self.data_val = time_stepper_discrete(x_lower1=x_lower1, x_upper1=x_upper1,
                                                  x_lower2=x_lower2, x_upper2=x_upper2,
                                                  n_ic=n_ic, dt=dt, tf=tf)
        elif data_type == 'pendulum':
            self.data_val = time_stepper_pendulum(x_lower1=x_lower1, x_upper1=x_upper1,
                                                  x_lower2=x_lower2, x_upper2=x_upper2,
                                                  n_ic=n_ic, dt=dt, tf=tf)

        elif data_type == 'fluid_flow':
            self.data_val = time_stepper_fluid_flow(r_lower=x_lower1, r_upper=x_upper1,
                                                    t_lower=x_lower2, t_upper=x_upper2,
                                                    n_ic=n_ic, dt=dt, tf=tf)
        n_i, n_r, n_t = self.data_val.shape
        self.params['data_type'] = data_type
        self.params['num_time_steps'] = n_t
        self.params["num_physical_dim"] = n_r
        self.params['num_initial_conditions'] = n_i


# ==============================================================================
# SUPPORTING FUNCTION IMPLEMENTATIONS
# ==============================================================================
def dyn_sys_discrete(lhs, mu=-0.05, lam=-1):
    """ example 1:
    ODE =>
    dx1/dt = mu*x1
    dx2/dt = lam*(x2-x1^2)

    By default: mu =-0.05, and lambda = -1.
    """
    rhs = np.zeros(2)
    rhs[0] = mu * lhs[0]
    rhs[1] = lam * (lhs[1] - (lhs[0]) ** 2.)
    return rhs


def dyn_sys_pendulum(lhs):
    """ pendulum example:
    ODE =>
    dx1/dt = x2
    dx2/dt = -sin(x1)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -np.sin(lhs[0])
    return rhs


def dyn_sys_fluid(lhs, mu=0.1, omega=1, A=-0.1, lam=10):
    """fluid flow example:
    ODE =>
    dx1/dt = mu*x1 - omega*x2 + A*x1*x3
    dx2/dt = omega*x1 + mu*x2 + A*x2*x3
    dx3/dt = -lam(x3 - x1^2 - x2^2)
    """
    rhs = np.zeros(3)
    rhs[0] = mu * lhs[0] - omega * lhs[1] + A * lhs[0] * lhs[2]
    rhs[1] = omega * lhs[0] + mu * lhs[1] + A * lhs[1] * lhs[2]
    rhs[2] = -lam * (lhs[2] - lhs[0] ** 2 - lhs[1] ** 2)
    return rhs


def rk4(lhs, dt, function):
    """
    :param lhs: previous step state.
    :param dt: delta t.
    :param data_type: "ex1" or "ex2".
    :return:  Runge–Kutta 4th order method.
    """
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs


def time_stepper_discrete(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=1e4, dt=0.02, tf=1.0):
    """
    :param tf: final time. default is 15.
    :param dt: delta t.
    :param x_lower1: lower bound of x1, initial condition.
    :param x_upper1: upper bound of x1, initial condition.
    :param x_upper2: lower bound of x2, initial condition.
    :param x_lower2: upper bound of x1, initial condition.
    :param n_ic: number of initial conditions on each axis. default is 100.
    :return: csv file "ex1.csv" or "ex2.csv"
    """
    # dim - time steps, default is 51.
    nsteps = np.int(tf / dt)

    # number of initial conditions.
    n_ic = np.int(n_ic)

    # create initial condition grid.
    icond1 = np.random.uniform(x_lower1, x_upper1, n_ic)
    icond2 = np.random.uniform(x_lower2, x_upper2, n_ic)

    # solve the system using Runge–Kutta 4th order method, see rk4 function above.
    data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float32)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float32)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_discrete)

    # process data for export.
    save_data_mat = data_mat.reshape((-1, nsteps + 1), order='C')

    # save the dataset generated in a csv file.
    np.savetxt('discrete_dataset.csv', save_data_mat, delimiter=',')

    return data_mat


def time_stepper_pendulum(x_lower1, x_upper1, x_lower2, x_upper2, n_ic=1e4, dt=0.02, tf=1.0):
    """
    :param tf: final time. default is 15.
    :param dt: delta t.
    :param x_lower1: lower bound of x1, initial condition.
    :param x_upper1: upper bound of x1, initial condition.
    :param x_upper2: lower bound of x2, initial condition.
    :param x_lower2: upper bound of x1, initial condition.
    :param n_ic: number of initial conditions
    :return: csv file
    """
    # dim - time steps, default is 51.
    nsteps = np.int(tf / dt)

    # number of initial conditions.
    n_ic = np.int(n_ic)

    # create initial condition grid
    max_potential = 0.99
    potential = lambda x, y: (1 / 2) * y ** 2 - np.cos(x)
    iconds = np.asarray([[x, y] for x, y in zip(np.random.uniform(x_lower1, x_upper1, 100 * n_ic),
                                                np.random.uniform(x_lower2, x_upper2, 100 * n_ic))
                         if potential(x, y) <= max_potential])[:n_ic, :]

    # solve the system using Runge–Kutta 4th order method, see rk4 function above.
    data_mat = np.zeros((n_ic, 2, nsteps + 1), dtype=np.float32)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1]], dtype=np.float32)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_pendulum)

    # process data for export.
    save_data_mat = data_mat.reshape((-1, nsteps + 1), order='C')

    # save the dataset generated in a csv file.
    np.savetxt('pendulum.csv', save_data_mat, delimiter=',')

    return data_mat


def time_stepper_fluid_flow(r_lower=0, r_upper=1.1, t_lower=0, t_upper=2 * np.pi, n_ic=1e4, dt=0.05, tf=6):
    """
    :param r_lower: lower bound for r. Default is 0.
    :param r_upper: Upper bound for r. Default is 1.
    :param t_lower: Lower bound for theta. Default is 0.
    :param t_upper: Upper bound for theta. Default is 2pi.
    :param n_ic: number of initial conditions. Default is 10000.
    :param dt: time step size. Default is 0.05.
    :param tf: final time. default is 6.
    :return: csv file
    """
    # dim - time steps, default is 51.
    nsteps = np.int(tf / dt)

    # number of initial conditions.
    n_ic = np.int(n_ic)

    # number of initial conditions for slow manifold.
    n_ic_slow = np.int(0.8*n_ic)

    # number of initial conditions for fast manifold.
    n_ic_fast = n_ic - n_ic_slow

    # create initial condition grid.
    r = np.random.uniform(r_lower, r_upper, n_ic_slow)
    theta = np.random.uniform(t_lower, t_upper, n_ic_slow)

    # compute x1, x2, and x3, based on theta and r.
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    x3 = np.power(x1, 2) + np.power(x2, 2)

    # initialize initial conditions matrix.
    iconds = np.zeros((n_ic, 3))

    # initial conditions for slow manifold.
    iconds[:n_ic_slow] = np.asarray([[x, y, z] for x, y, z in zip(x1, x2, x3) if z <= 2.5])[:n_ic_slow, :]

    # initial conditions for fast manifold.
    iconds[n_ic_slow:] = np.asarray([[x, y, z] for x, y, z in zip(np.random.uniform(-1.1, 1.1, 100 * n_ic_fast),
                                                                  np.random.uniform(-1.1, 1.1, 100 * n_ic_fast),
                                                                  np.random.uniform(0, 2.42, 100 * n_ic_fast))
                                     if z <= 2.5])[:n_ic_fast, :]

    # solve the system using Runge–Kutta 4th order method, see rk4 function above.
    data_mat = np.zeros((n_ic, 3, nsteps + 1), dtype=np.float32)
    for ii in range(n_ic):
        data_mat[ii, :, 0] = np.array([iconds[ii, 0], iconds[ii, 1], iconds[ii, 2]], dtype=np.float32)
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, dyn_sys_fluid)

    # process data for export.
    save_data_mat = data_mat.reshape((-1, nsteps + 1), order='C')

    # save the dataset generated in a csv file.
    np.savetxt('fluid_flow_dataset.csv', save_data_mat, delimiter=',')

    return data_mat


# ==============================================================================
# UNIT TEST
# ==============================================================================
if __name__ == "__main__":

    create_discrete = False
    create_pendulum = False
    create_fluid_flow = True

    if create_discrete:
        # create the dataset
        training_data = DataMaker(x_lower1=-0.5, x_upper1=0.5, x_lower2=-0.5, x_upper2=0.5,
                                  n_ic=10000, dt=0.02, tf=5, data_type="discrete")
        # save the dataset in a pickle file.
        pickle.dump(training_data, open('../../dmd_autoencoder - Copy/dmd_autoencoder_2/data/dataset_discrete.pkl', 'wb'))

        # plot the dataset for visualization.
        data = training_data.data_val
        plt.figure(figsize=(15, 8))
        for ii in np.arange(0, 10000, 10):
            x1 = training_data.data_val[ii, 0, :]
            x2 = training_data.data_val[ii, 1, :]
            plt.plot(x1, x2, '-')
        plt.xlabel("$x_{1}$", fontsize=20)
        plt.ylabel("$x_{2}$", fontsize=20)
        plt.title("Discrete dataset", fontsize=20)
        plt.savefig('discrete.png')
        plt.show()

    if create_pendulum:
        # create the dataset
        training_data = DataMaker(x_lower1=-3.1, x_upper1=3.1, x_lower2=-2, x_upper2=2,
                                  n_ic=10000, dt=0.3, tf=15, data_type="pendulum")
        # save the dataset in a pickle file.
        pickle.dump(training_data, open('../../dmd_autoencoder - Copy/dmd_autoencoder_2/data/dataset_pendulum.pkl', 'wb'))

        # plot the dataset for visualization.
        data = training_data.data_val
        plt.figure(figsize=(15, 8))
        for ii in np.arange(0, 10000, 10):
            x1 = training_data.data_val[ii, 0, :]
            x2 = training_data.data_val[ii, 1, :]
            plt.plot(x1, x2, '-')
        plt.xlabel("$x_{1}$", fontsize=20)
        plt.ylabel("$x_{2}$", fontsize=20)
        plt.title("Pendulum dataset", fontsize=20)
        plt.savefig('pendulum.png')
        plt.show()

    if create_fluid_flow:
        # create the dataset
        training_data = DataMaker(x_lower1=0, x_upper1=1.1, x_lower2=0, x_upper2=2 * np.pi,
                                  n_ic=1e4, dt=0.2, tf=50, data_type="fluid_flow")
        # save the dataset in a pickle file.
        pickle.dump(training_data, open('../../dmd_autoencoder - Copy/dmd_autoencoder_2/data/dataset_fluid.pkl', 'wb'))

        # plot the dataset for visualization.
        data = training_data.data_val
        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection='3d')

        for ii in np.arange(0, 1e4, 1e3):
            ii = int(ii)
            x1 = training_data.data_val[ii, 0, :]
            x2 = training_data.data_val[ii, 1, :]
            x3 = training_data.data_val[ii, 2, :]
            ax.plot3D(x1, x2, x3, 'gray')

        ax.set_xlabel("$x_{1}$", fontsize=15)
        ax.set_ylabel("$x_{2}$", fontsize=15)
        ax.set_zlabel("$x_{3}$", fontsize=15)
        ax.text2D(0.05, 0.95, "Fluid Flow dataset", transform=ax.transAxes, fontsize=20)
        # plt.savefig('fluid_flow.png')
        plt.show()
