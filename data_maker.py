import numpy as np


def function_ex1(lhs, mu=-0.05, lam=-1):
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


def function_ex2(lhs):
    """ pendulum example:
    ODE =>
    dx1/dt = x2
    dx2/dt = -sin(x1)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -np.sin(lhs[0])
    return rhs


def rk4(lhs, dt, data_type="ex1"):
    """
    :param lhs: previous step state.
    :param dt: delta t.
    :param data_type: "ex1" or "ex2".
    :return:  Runge–Kutta 4th order method.
    """
    if data_type == "ex1":
        k1 = dt * function_ex1(lhs)
        k2 = dt * function_ex1(lhs + k1 / 2.)
        k3 = dt * function_ex1(lhs + k2 / 2.)
        k4 = dt * function_ex1(lhs + k3)

    elif data_type == "ex2":
        k1 = dt * function_ex2(lhs)
        k2 = dt * function_ex2(lhs + k1 / 2.)
        k3 = dt * function_ex2(lhs + k2 / 2.)
        k4 = dt * function_ex2(lhs + k3)

    rhs = lhs + 1. / 6. * (k1 + 2. * (k2 + k3) + k4)
    return rhs


def time_stepper(x_lower1, x_upper1, x_lower2, x_upper2, dt=0.3, tf=15., n_side=100, datatype="ex1"):
    """
    :param tf: final time. default is 15.
    :param dt: delta t.
    :param x_lower1: lower bound of x1, initial condition.
    :param x_upper1: upper bound of x1, initial condition.
    :param x_upper2: lower bound of x2, initial condition.
    :param x_lower2: upper bound of x1, initial condition.
    :param n_side: number of initial conditions on each axis. default is 100.
    :param datatype: "ex1" or "ex2"
    :return: csv file "ex1.csv" or "ex2.csv"
    """
    # dim - time steps, default is 51.
    nsteps = np.int(tf / dt)

    # number of initial conditions.
    nicond = np.int(n_side ** 2)

    # create initial condition grid.
    icond1 = np.linspace(x_lower1, x_upper1, n_side, dtype=np.float32)
    icond2 = np.linspace(x_lower2, x_upper2, n_side, dtype=np.float32)

    # solve the system using Runge–Kutta 4th order method, see rk4 function above.
    data_mat = np.zeros((nicond, 2, nsteps + 1), dtype=np.float32)
    for jj in range(n_side):
        for kk in range(n_side):
            cind = n_side * jj + kk
            data_mat[cind, :, 0] = np.array([icond1[jj], icond2[kk]], dtype=np.float32)
            for ll in range(nsteps):
                data_mat[cind, :, ll + 1] = rk4(data_mat[cind, :, ll], dt, data_type=datatype)

    # process data for export.
    sav_data_mat = data_mat.reshape((-1, nsteps + 1), order='C')

    # save the dataset generated in a csv file.
    if datatype == "ex1":
        np.savetxt('ex1.csv', sav_data_mat, delimiter=',')
    elif datatype == "ex2":
        np.savetxt('ex2.csv', sav_data_mat, delimiter=',')

    return data_mat
