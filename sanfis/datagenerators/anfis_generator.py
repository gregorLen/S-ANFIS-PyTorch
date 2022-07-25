import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
plt.rcParams['axes.xmargin'] = 0  # remove margins from all plots
##############################################################################


def gen_data(data_id: str = 'mackey', n_obs: int = 1000, n_input: int = 2, lag: int = 1):

    # Mackey
    if data_id == 'mackey':
        y = mackey(124 + n_obs + n_input)[124:]
        X, y = gen_X_from_y(y, n_input, lag)

    # Nonlin sinc equation
    elif data_id == 'sinc':
        X, y = sinc_data(n_obs)
        assert n_input == 2, 'Nonlin sinc equation data set requires n_input==2. Please chhange to 2.'

    # Nonlin three-input equation
    elif data_id == '3in':
        X, y = nonlin_data(n_obs)
        assert n_input == 3, 'Nonlin Three-Input Equation required n_input==3. Please switch to 3.'
    else:
        raise ValueError(
            f'data_id must be either "mackey", "sinc" or "3in". Given was {data_id}')

    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    # to torch
    X, y = torch.Tensor(X), torch.Tensor(y)

    # split data into test and train set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False)

    return X, X_train, X_test, y, y_train, y_test

##############################################################################


# Mackey-Glass series computation
def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x


# Modelling a two-Input Nonlinear Function (Sinc Equation)
def sinc_equation(x1, x2):
    return ((np.sin(x1) / x1) * (np.sin(x2) / x2))


def sinc_data(n_obs, multiplier=2, noise=False):
    X = (np.random.rand(n_obs, 2) - .5) * multiplier
    y = sinc_equation(X[:, 0], X[:, 1]).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs) * 0.1
    return X.astype('float32'), y.astype('float32')


# Modelling a Three-Input Nonlinear Function (Sinc Equation)
def nonlin_equation(x, y, z):
    return ((1 + x**0.5 + 1 / y + z**(-1.5))**2)


def nonlin_data(n_obs, multiplier=1, noise=False):
    X = np.random.rand(n_obs, 3) * multiplier + 1
    y = nonlin_equation(X[:, 0], X[:, 1], X[:, 2]).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs)
    return X.astype('float32'), y.astype('float32')


# Generate a input matrix X from time series y
def gen_X_from_y(x, n_input=1, lag=1):
    n_obs = len(x) - n_input * lag

    data = np.zeros((n_obs, n_input + 1))
    for t in range(n_input * lag, n_obs + n_input * lag):
        data[t - n_input * lag, :] = [x[t - i * lag]
                                      for i in range(n_input + 1)]
    X = data[:, 1:].reshape(n_obs, -1)
    y = data[:, 0].reshape(n_obs, 1)

    return X.astype('float32'), y.astype('float32')
