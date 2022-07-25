import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from typing import Optional, Union
from sklearn.model_selection import train_test_split
import torch

# supress warnings
warnings.filterwarnings('ignore')


class SANFIS_Process:
    def __init__(self,
                 mu_params=np.array([[0.05, 0.05]]),

                 sigma_params=np.array([[0.25, 0.25]]),

                 AR_params=np.array([[0.50, 0],
                                     [-0.50, 0]]),
                 gamma=float('inf'),
                 c=0.0,
                 n_statevars=1,
                 n_memb=2,
                 memb_func='sigmoid',
                 upper_limit=10.0,
                 lower_limit=-10.0
                 ):
        """[summary]
        Args:
            mu_params ([type], optional): [description]. Defaults to np.array([[0.05, 0.05]]).
            sigma_params ([type], optional): [description]. Defaults to np.array([[0.25, 0.25]]).
            AR_params ([type], optional): [description]. Defaults to np.array([[0.50, 0], [-0.50, 0]]).
            gamma ([type], optional): [description]. Defaults to float('inf').
            c (float, optional): [description]. Defaults to 0.0.
            n_statevars (int, optional): [description]. Defaults to 1.
            n_memb (int, optional): [description]. Defaults to 2.
            memb_func (str, optional): [description]. Defaults to 'sigmoid'.
        """
        assert len(sigma_params.shape) == len(
            mu_params.shape) == 2, f'mu_params and sigma_params must be of dimension (n_staes x 2) but they are {sigma_params.shape} and {sigma_params.shape}'
        assert sigma_params.shape == mu_params.shape, f'mu_params and sigma_params must be of same shape but they are {sigma_params.shape} and {mu_params.shape}'
        assert mu_params.shape[
            1] == n_memb ** n_statevars, f'number of rules (mu_params.shape[1]) should equal n_memb**n_statevars. But n_memb == {n_memb}, n_statevars == {n_statevars} and n_rules = {mu_params.shape[1]}'

        self.gamma, self.c, self.mu_params, self.sigma_params, self.AR_params = gamma, c, mu_params, sigma_params, AR_params
        self.n_statevars = n_statevars
        self.m = n_memb   # of memberships
        self.memb_func = memb_func
        self.r = None
        self.y = None
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def cumsum_with_limits(self, values):
        n_obs = len(values)
        new_values = np.empty([n_obs, self.n_statevars]).astype(np.float32)
        sum_val = np.zeros([1, self.n_statevars])
        for i in range(n_obs):
            x = np.maximum(np.minimum(
                sum_val + values[i], self.upper_limit), self.lower_limit) - sum_val
            new_values[i] = x
            sum_val += x
        res = np.cumsum(new_values, axis=0)
        return res

    def sim_statevars(self, N=750, threshold=0):
        # np.random.seed(1)
        wn = np.random.randn(N, self.n_statevars).astype(np.float32)
        statevars = self.cumsum_with_limits(
            wn)

        # firing strength
        fire = []
        for s in range(self.n_statevars):
            aux_ = self.G(statevars[:, [s]], c=self.c[:, [
                          s]].T, gamma=self.gamma[:, [s]].T)
            fire.append(aux_)

        #####################

        # fire = 1 / (1 + np.exp(-self.gamma*(np.repeat(statevars, self.m, axis=1).reshape(-1, self.m, self.n_statevars))-self.c))

        # #
        # CP = []
        # # for every observation in batch...
        # for observation in fire:
        #     cp = observation[:, 0]
        #     # ...get cartesian product over each dimension
        #     for d in range(1, fire.shape[2]):
        #         xd = observation[:, d].unsqueeze(0)
        #         cp = np.matmul(cp.unsqueeze(-1), xd)

        #     flat_cp = cp.reshape(1, -1)
        #     CP.append(flat_cp)

        # output = np.reshape(np.stack(
        #     CP), (fire.shape[0], -1))

        ################
        states = []
        if self.n_statevars == 1:
            states = np.hstack(fire)
        elif self.n_statevars > 1:
            for i in range(self.m):
                for j in range(self.m):
                    aux_ = fire[0][:, [i]] * fire[1][:, [j]]
                    states.append(aux_)
                    # TODO: WONT WORK FOR MORE THANT 2 STATES! REVIEW!

            states = np.hstack(states)

        norm_states = states / np.sum(states, axis=1, keepdims=True)

        return norm_states, statevars

    def G(self, statevars, c=0.0, gamma=float('inf')):
        """
         Transition Function G:
         statevars = state variables
         c = threshold parameter
         gamma = steepness parameter
        """
        if self.memb_func == 'sigmoid':
            output = (1 + np.exp(-gamma * (statevars - c)))**(-1)
        elif self.memb_func == 'gaussian':
            output = np.exp(-((statevars - c) / gamma)**2)
        else:
            raise Exception(
                'Membership function fo the generator must be either "gaussian" or "sigmoid"')

        return output

    def sim(self, N=750, state_0=0, dt=1 / 250, t_burn=100):

        # sim regime states (and state variables)
        self.states, self.statevars = self.sim_statevars(N + t_burn)

        # sim shocks
        e = np.random.randn(N + t_burn).astype(np.float32)   # sim shocks

        # set first and second observations
        e[0:2] = e[0:2] * self.sigma_params[0][state_0] * np.sqrt(dt)
        r = e.copy()
        r[0:2] = r[0:2] + self.mu_params[0][state_0] * dt

        # Simulate:
        for t in np.arange(2, N + t_burn):
            # determine weights
            state = self.states[[t]]  # TODO review t or t-1 (information!)

            # calculate returns
            e[t] = np.matmul(self.sigma_params * e[t] * np.sqrt(dt), state.T)
            mu = np.matmul(self.mu_params, state.T) * dt
            r[t] = mu + e[t] + \
                np.matmul(
                    np.sum(self.AR_params * r[np.array([[t - 1], [t - 2]])], axis=0), state.T)

        # remove burn in period
        self.states = self.states[t_burn:, ]
        self.statevars = self.statevars[t_burn:, ]
        self.r = r[t_burn:]

        # calc prices
        self.y = 100 * np.exp(np.cumsum(self.r))

    def plot(self, colored=True, type='stock', save_path: Optional[str] = None):
        """Plots the time series of this process generator.
        Args:
            colored (bool, optional): adding colors. Defaults to True.
            type (str, optional): 'stock' for plotting stock prices, 'general' for plotting a general process. Defaults to 'stock'.
        """
        plt.style.use('ggplot')
        if self.n_statevars > 1 and colored == True:
            colored = False

        r = self.r
        states = self.states
        statevars = self.statevars
        y = self.y

        if type == 'stock':
            fig, axes = plt.subplots(3, figsize=(10, 12))
            fig.subplots_adjust(hspace=0.4)

            # ax 0
            ax = axes[0]
            ax.plot(statevars)  # , 'k', linewidth=.7)
            ax.margins(x=0)
            # ax.hlines(self.c,0,len(r), linestyles='dashed')
            if colored == True:
                ax.fill_between(np.arange(len(r)), np.min(statevars), np.max(statevars),
                                facecolor='green', alpha=.3)
                ax.fill_between(np.arange(len(r)), np.min(statevars), np.max(statevars), where=states[:, 0] > 0.5,
                                facecolor='blue', alpha=.3)
            ax.set(title='State Variables')

            # ax 1
            ax = axes[1]
            ax.plot(r, 'k', linewidth=.7)
            ax.margins(x=0)
            ax.hlines(0, 0, len(r), linestyles='dashed')
            if colored == True:
                ax.fill_between(np.arange(len(r)), min(r), max(r),
                                facecolor='green', alpha=.3)
                ax.fill_between(np.arange(len(r)), min(r), max(r), where=states[:, 0] > 0.5,
                                facecolor='blue', alpha=.3)
            ax.set(title='Simulated Returns')

            # ax 2
            ax = axes[2]
            ax.plot(y, 'k', linewidth=.7)
            ax.margins(x=0)
            if colored == True:
                ax.fill_between(np.arange(len(r)), min(y), max(y),
                                facecolor='green', alpha=.3)
                ax.fill_between(np.arange(len(r)), min(y), max(y), where=states[:, 0] > 0.5,
                                facecolor='blue', alpha=.3)

            ax.set(title='Simulated Prices')
            # ax.set_yscale('log')
            plt.show()

        elif type == 'general':
            fig, axes = plt.subplots(2, figsize=(8, 6))
            fig.subplots_adjust(hspace=0.1)

            # ax 0
            ax = axes[0]
            ax.plot(statevars)  # , 'k', linewidth=.7)
            for tick in ax.xaxis.get_major_ticks():
                # 'hack' to remove xtick labels but keep the grid
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            ax.tick_params(axis='y', labelsize=14)
            ax.margins(x=0)
            # ax.hlines(self.c,0,len(r), linestyles='dashed')
            if colored == True:
                ax.fill_between(np.arange(len(r)), np.min(statevars), np.max(statevars),
                                facecolor='green', alpha=.3)
                ax.fill_between(np.arange(len(r)), np.min(statevars), np.max(statevars), where=states[:, 0] > 0.5,
                                facecolor='blue', alpha=.3)
            # ax.set(title=r'State Variable($s_1$, $s_2$)')
            # ax.legend([r'$s_1$', r'$s_2$'], prop={
            #           'size': 15}, loc='upper center', bbox_to_anchor=(0.8, 1.05))

            # ax 1
            ax = axes[1]
            ax.plot(r, 'k', linewidth=.7)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.margins(x=0)
            ax.hlines(0, 0, len(r), linestyles='dashed')
            if colored == True:
                ax.fill_between(np.arange(len(r)), min(r), max(r),
                                facecolor='green', alpha=.3)
                ax.fill_between(np.arange(len(r)), min(r), max(r), where=states[:, 0] > 0.5,
                                facecolor='blue', alpha=.3)
            #ax.set(title='Simulated process ' + r'$x_t$')
            # ax.legend([f'$x_t$'], prop={'size': 15})
            fig.legend([r'$s_1$', r'$s_1$', r'$x_t$'], fontsize=10,
                       ncol=5, loc='upper center', prop={'size': 20})

            plt.show()

        else:
            raise ValueError(f'type must be either "stock" or "general". {type} was given')

        # save plot
        if save_path != None:
            fig.savefig(os.path.join(save_path, 'sanfis_dgp_process.pdf'),
                        bbox_inches='tight', pad_inches=0)

    def plotmfs(self, show_equation: bool = False, save_path: Optional[str] = None):
        plt.style.use('ggplot')
        gamma, c = self.gamma, self.c
        gamma, c = gamma.reshape((self.m, self.n_statevars, 1)), c.reshape(
            self.m, self.n_statevars, 1)
        # gamma, c = gamma.reshape((self.n_statevars, 1, self.m)), c.reshape(
        #    (self.n_statevars, 1, self.m))

        xn = np.linspace(self.lower_limit, self.upper_limit, 999)
        xn = np.tile(xn, (self.m, self.n_statevars, 1))

        # broadcast all curves in one array
        memb_curves = self.G(xn, c, gamma)

        fig, axs = plt.subplots(nrows=self.n_statevars,
                                ncols=1, figsize=(8, self.n_statevars * 3))
        fig.subplots_adjust(hspace=0.4)
        # fig.suptitle('Membership functions (DGP)', size=16)

        if self.n_statevars == 1:
            axs.grid(True)
            axs.set_title(f'State Variable {1}')
            for m in range(self.m):
                axs.plot(xn[m, 0, :], memb_curves[m, 0, :])
        elif self.n_statevars > 1:
            for s in range(self.n_statevars):
                axs[s].grid(True)
                axs[s].set_title(f'State Variable {s+1}', fontsize=19)
                for m in range(self.m):
                    axs[s].plot(xn[m, s, :], memb_curves[m, s, :])
                axs[s].tick_params(axis='x', labelsize=14)
                axs[s].tick_params(axis='y', labelsize=14)

        if (self.n_statevars == 2) & (show_equation):
            axs[0].text(5, 0.9, r'$F(s_{1,t};\gamma_{1,1},c_{1,1})$', size=16)
            axs[0].text(5, 0.1, r'$F(s_{1,t};\gamma_{1,2},c_{1,2})$', size=16)
            axs[1].text(5, 0.9, r'$G(s_{2,t};\gamma_{2,1},c_{2,1})$', size=16)
            axs[1].text(5, 0.1, r'$G(s_{2,t};\gamma_{2,2},c_{2,2})$', size=16)

        print('Membership functions (DGP):')
        plt.show()

        # save plot
        if save_path != None:
            fig.savefig(os.path.join(save_path, 'dgp_mfs.pdf'),
                        bbox_inches='tight', pad_inches=0)


def gen_data_ts(n_obs: int, dgp_params: Optional[dict] = None, lag: int = 1, test_size: float = 0.5, shuffle_data: bool = False, plot_dgp: bool = False, as_tensor: bool = True, save_path: Optional[str] = None):
    """Datagenerator method.
    Args:
        n_obs (int): Number of observations
        n_statevars (int): Number of state variables
        n_input (int): Number of input variables
        memb_func (str, optional): Membership function for the state variables (sigmoid/gaussian). Default to sigmoid.
        batch_size (int, optional): Batch size. Defaults to 16.
        lag (int, optional): Lag parameter. Defaults to 1.
        plot_dgp (bool, optional): Plot data generating process. Defaults to False.
        as_tensor (bool, optional): Data as torch.tensor(). Defaults to True.
    Returns:
        S, S_train, S_valid, X, X_train, X_valid, y, y_train, y_valid: Raw/Train/valid Samples for S, X, and y.
    """
    if dgp_params == None:
        # dgp_params = {"mu_params": np.array([[0.2, -0.25, 0.1, 0.0]]),
                      # "sigma_params": np.array([[0.05, 0.05, 0.05, 0.05]]),
                      # "AR_params": np.array([[0.2, 0.5, -0.3, 0.1],
                                             # [0.0, 0.0, 0.0, 0.0]]),
                      # "a_params": np.array([[10.5, 10.5],   # gauss: standard deviation  // sigmoid: steepness (gamma)
                                            # [-10.5, -10.5]]),
                      # "c_params": np.array([[0.0, 0.0],  # center of the curve or bell
                                            # [0.0, 0.0]]),
                      # "memb_func": "gaussian"}
        dgp_params = {"mu_params": np.array([[0.4, 0.0, 0.2, -0.4]]),
                        "sigma_params": np.array([[0.1, 0.1, 0.1, 0.1]]),
                    "AR_params": np.array([[0.2, 0.5, -0.3, 0.1],
                                            [0.1, 0.1, 0.2, -0.1]]),
                    "memb_func": 'sigmoid',
                    "a_params": np.array([[1.5, 15.5],   # gauss: standard deviation  // sigmoid: steepness (gamma)
                                            [-15.5, -2.5]]),
                    "c_params": np.array([[-3.0, -5.0],  # center of the curve or bell
                                            [3.0, 1.0]]),
                            "memb_func": "sigmoid"}

    n_input = dgp_params['AR_params'].shape[0]

    every_state_in_data = False
    while every_state_in_data == False:
        process = SANFIS_Process(dgp_params['mu_params'], dgp_params['sigma_params'], dgp_params['AR_params'], dgp_params['a_params'],
                                 dgp_params['c_params'], dgp_params['c_params'].shape[0], dgp_params['c_params'].shape[1], dgp_params["memb_func"])
        process.sim(n_obs + n_input, dt=1)

        X, y = gen_X_from_y(process.r, n_input, lag)
        S = process.statevars[n_input * lag:, :]

        # split train/valid set
        S_train, S_valid, X_train, X_valid, y_train, y_valid = train_test_split(
            S, X, y, test_size=test_size, shuffle=shuffle_data)

        # check if every state is in train data
        every_state_in_data = check_traindata(
            S_train, dgp_params, threshold=0.05)

    if plot_dgp:
        process.plot(colored=True, type='general', save_path=save_path)
        process.plotmfs(save_path=save_path)

    # Return
    if as_tensor:
        return tuple(torch.from_numpy(data) for data in (S, S_train, S_valid, X, X_train, X_valid, y, y_train, y_valid))
    else:
        return S, S_train, S_valid, X, X_train, X_valid, y, y_train, y_valid


def gen_X_from_y(y, n_input=1, lag=1, return_embedding: bool = False):
    """Generate an input matrix X from time series y
    Args:
        y (array): Time series array (1 dimensional)
        n_input (int, optional): Number of columns for the X-matrix. Defaults to 1. The number of lags to include in the model if an integer or the list of lag indices to include. For example, [1, 4] will only include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
        lag (int or list, optional): Lag parameter if n_input is int. Defaults to 1.    
        return_embedding (boolean): if True, returns the whole data embedding
    Returns:
        X: (lagged) Input matrix X
        y: Time series
    """
    if type(n_input) == int:
        n_obs = len(y) - n_input * lag
        data = np.zeros((n_obs, n_input + 1)).astype(np.float32)
        for t in range(n_input * lag, n_obs + n_input * lag):
            data[t - n_input * lag, :] = [y[t - i * lag]
                                          for i in range(n_input + 1)]

    elif type(n_input) == list:
        inputs = [0, *n_input]   # add t=0
        n_obs = len(y) - n_input[-1]
        data = np.zeros((n_obs, len(inputs))).astype(np.float32)
        for t in range(inputs[-1], n_obs + n_input[-1]):
            data[t - inputs[-1], :] = [y[t - n_input] for n_input in inputs]
    else:
        raise TypeError(
            f'Variable n_input must either be int or list, but {n_input} was given.')

    if return_embedding:
        return data

    X = data[:, 1:].reshape(n_obs, -1)
    y = data[:, 0].reshape(n_obs, 1)

    return X, y


def check_traindata(S_train: Union[torch.Tensor, np.ndarray], dgp_params: dict, threshold: float = 0.05) -> bool:
    """ensures that each state is given in the data at least with *threshold* percentage

    Args:
        S_train (Union[torch.Tensor, np.ndarray]): Data of state variables.
        dgp_params (dict): dgp parameters.
        threshold (float, optional): Defaults to 0.05.

    Returns:
        bool: Indicates if every state is given in the data.
    """

    # Ensure data that includes every state
    sv1 = S_train[:, 0]
    sv2 = S_train[:, 1]
    threshold = threshold * len(sv1)

    sv1min = min(dgp_params['c_params'][0][0], dgp_params['c_params'][1][0])
    sv1max = max(dgp_params['c_params'][0][0], dgp_params['c_params'][1][0])
    sv2min = min(dgp_params['c_params'][0][1], dgp_params['c_params'][1][1])
    sv2max = max(dgp_params['c_params'][0][1], dgp_params['c_params'][1][1])

    if (sv1max - sv1min) < 2 and (sv2max - sv2min) < 2:
        check = (
            np.sum(np.logical_and(sv1 > sv1max, sv2 > sv2max)) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1max, sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1min, sv2 > sv2max)) >= threshold) & (
            np.sum(np.logical_and(sv1 > sv1max, sv2 < sv2min)) >= threshold)

    elif (sv1max - sv1min) > 2 and (sv2max - sv2min) > 2:
        check = (
            # cases when sv1 < sv1min
            np.sum(np.logical_and(sv1 < sv1min, sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1min, np.logical_and(sv2 > sv2min, sv2 < sv2max))) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1min, sv2 > sv2max)) >= threshold) & (

            # cases when sv1min < sv1 < sv1max
            np.sum(np.logical_and(np.logical_and(sv1 > sv1min, sv1 < sv1max), sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(np.logical_and(sv1 > sv1min, sv1 < sv1max), np.logical_and(sv2 > sv2min, sv2 < sv2max))) >= threshold) & (
            np.sum(np.logical_and(np.logical_and(sv1 > sv1min, sv1 < sv1max), sv2 > sv2max)) >= threshold) & (

            # cases when sv1max < sv1
            np.sum(np.logical_and(sv1 > sv1max, sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(sv1 > sv1max, np.logical_and(sv2 > sv2min, sv2 < sv2max))) >= threshold) & (
            np.sum(np.logical_and(sv1 > sv1max, sv2 > sv2max)) >= threshold)

    elif (sv1max - sv1min) < 2 and (sv2max - sv2min) > 2:
        check = (
            # cases when sv1 < sv1max
            np.sum(np.logical_and(sv1 < sv1max, sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1max, np.logical_and(sv2 > sv2min, sv2 < sv2max))) >= threshold) & (
            np.sum(np.logical_and(sv1 < sv1max, sv2 > sv2max)) >= threshold) & (

            # cases when sv1 > sv1min
            np.sum(np.logical_and(sv1 > sv1min, sv2 < sv2min)) >= threshold) & (
            np.sum(np.logical_and(sv1 > sv1min, np.logical_and(sv2 > sv2min, sv2 < sv2max))) >= threshold) & (
            np.sum(np.logical_and(sv1 > sv1min, sv2 > sv2max)) >= threshold)

    elif (sv1max - sv1min) > 2 and (sv2max - sv2min) < 2:
        check = (
            # cases when sv2 < sv2max
            np.sum(np.logical_and(sv2 < sv2max, sv1 < sv1min)) >= threshold) & (
            np.sum(np.logical_and(sv2 < sv2max, np.logical_and(sv1 > sv1min, sv1 < sv1max))) >= threshold) & (
            np.sum(np.logical_and(sv2 < sv2max, sv1 > sv1max)) >= threshold) & (

            # cases when sv2 > sv2min
            np.sum(np.logical_and(sv2 > sv2min, sv1 < sv1min)) >= threshold) & (
            np.sum(np.logical_and(sv2 > sv2min, np.logical_and(sv1 > sv1min, sv1 < sv1max))) >= threshold) & (
            np.sum(np.logical_and(sv2 > sv2min, sv1 > sv1max)) >= threshold)

    return check


if __name__ == "__main__":
    N = 300
    burn_in = 100  # burn-in period
    dt = 1  # 1/250
    n_statevars = 2         # 1 / 2
    n_memb = 2
    setting = 5
    memb_func = 'sigmoid'   # gaussian / sigmoid

    if n_statevars == 1:
        mu_params = np.array([[0.05, -0.15]])
        sigma_params = np.array([[0.2, 0.5]])
        AR_params = np.array([[0.0, 0],
                              [0.0, 0]])
        gamma = np.array([[3],
                          [-3]])
        c = np.array([[5.0],
                      [5.0]])

    elif n_statevars == 2:

        # SETTING 1
        if setting == 1:
            mu_params = np.array([[0.35, 0.05, 0.0, -0.5]])
            sigma_params = np.array([[0.0, 0.0, 0.0, 0.0]])
            AR_params = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[float('inf'), float('inf')],
                              [-float('inf'), -float('inf')], ])
            c = np.array([[0.0, 0.0],
                          [0.0, 0.0]])

        # SETTING 2
        elif setting == 2:
            mu_params = np.array([[0.10, 0.05, 0.0, -0.25]])
            sigma_params = np.array([[0.0, 0.0, 0.0, 0.0]])
            AR_params = np.array([[0.2, -0.5, 0.1, 0.3],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[float('inf'), float('inf')],
                              [-float('inf'), -float('inf')], ])
            c = np.array([[-3.0, -3.0],
                          [-3.0, -3.0]])

        # SETTING 3
        elif setting == 3:
            mu_params = np.array([[0.2, -0.25, 0.1, 0.0]])
            sigma_params = np.array([[0.05, 0.05, 0.05, 0.05]])
            AR_params = np.array([[0.2, 0.5, -0.3, 0.1],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[10.5, 10.5],
                              [-10.5, -10.5]])
            c = np.array([[0.0, 0.0],
                          [0.0, 0]])

        # SETTING 4
        elif setting == 4:
            mu_params = np.array([[0.2, -0.25, 0.1, 0.0]])
            sigma_params = np.array([[0.00, 0.00, 0.00, 0.00]])
            AR_params = np.array([[0.0, 0.0, -0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[100.5, 100.5],
                              [-100.5, -100.5]])
            c = np.array([[0.0, 0.0],
                          [0.0, 0]])

        # SETTING 5
        elif setting == 5:
            mu_params = np.array([[0.2, -0.25, 0.1, 0.0]])
            sigma_params = np.array([[0.05, 0.05, 0.05, 0.05]])
            AR_params = np.array([[0.2, 0.5, -0.3, 0.1],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[1.5, 15.5],   # gauss: standard deviation  // sigmoid: steepness (gamma)
                              [-15.5, -2.5]])
            c = np.array([[0.0, -1.5],  # center of the curve or bell
                          [1.5, 1.5]])

        # SETTING 6
        elif setting == 6:
            mu_params = np.array([[0.2, -0.25, 0.1, 0.0]])
            sigma_params = np.array([[0.0, 0.0, 0.0, 0.0]])
            AR_params = np.array([[0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[1000000, 1000000],
                              [-1000000, -1000000]])
            c = np.array([[-2, -2],  # center of the curve or bell
                          [2, 2]])

        else:
            mu_params = np.array([[0.2, -0.25, 0.1, 0.0]])
            sigma_params = np.array([[0.05, 0.05, 0.05, 0.05]])
            AR_params = np.array([[0.2, 0.5, -0.3, 0.1],
                                  [0.0, 0.0, 0.0, 0.0]])
            gamma = np.array([[1.5, 15.5],   # gauss: standard deviation  // sigmoid: steepness (gamma)
                              [-15.5, -2.5]])
            c = np.array([[0.0, -1.5],  # center of the curve or bell
                          [1.5, 1.5]])

        # ...WORK IN PROGRESS: 3 STATE VARIABLES
    else:
        raise ValueError(f'n_statevars must be 1 or 2. {n_statevars} was given.')

    # sim data
    model = SANFIS_Process(mu_params, sigma_params,
                           AR_params, gamma, c, n_statevars, n_memb, memb_func)
    model.sim(N, dt=dt, t_burn=burn_in)

    # plot data
    model.plotmfs()
    model.plot(colored=True, type='general')

    # extract information
    states = model.states
    statevars = model.statevars

    # # test error
    # import torch
    # y = torch.tensor(model.r)
    # y_pred = torch.zeros_like(y)
    # loss_function = torch.nn.MSELoss(reduction='mean')
    # loss = loss_function(y, y_pred)
    # print(f'loss is {torch.sqrt(loss)}')

    