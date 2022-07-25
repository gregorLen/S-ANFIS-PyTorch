# %% ## Load modules ##
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sanfis import SANFIS, plottingtools
from sanfis.datagenerators import sanfis_generator

##  Set Parameters ##
# Model parameters
param = {"n_statevars": 2,            # no. of state variables
         "n_input": 2,                # no. of Regressor
         "n_memb": 2,                 # no. of fuzzy memberships
         "batch_size": 100,            # 16 / 32 / 64 / ...
         "memb_func": 'sigmoid',      # 'gaussian' / 'gbellmf' / 'sigmoid'
         "scaler": 'Std',           # 'Std', 'Median', None
         "n_epochs": 1000,               # 10 / 25 / 50 / 100 / ...
         "lr": 0.005,               # learning rate of the optimizer
         "patience": 100,            # patience parameter of the optimizer
         # delta param of the optimizer  (see paper for details)
         "delta": 1e-6,             # paper: setting1=3e-6 -- setting 2=1e-6
         "sigma": 0.1
         }

membfuncs = [
    {'function': 'sigmoid',
     'n_memb': 2,
     'params': {'c': {'value': [0.0, 0.0],
                      'trainable': True},
                'gamma': {'value': [-2.5, 2.5],
                          'trainable': True}}},

    {'function': 'sigmoid',
     'n_memb': 2,
     'params': {'c': {'value': [0.0, 0.0],
                      'trainable': True},
                'gamma': {'value': [-2.5, 2.5],
                          'trainable': True}}}
]


# DGP Parameters
n_obs = 1000
n_statevars = 2
lag = 1
shuffle_data = False
shuffle_batches = True
test_size = 0.4

dgp_params = {"mu_params": np.array([[0.4, 0.0, 0.2, -0.4]]),
              "sigma_params": np.array([[param['sigma'], param['sigma'], param['sigma'], param['sigma']]]),
              "AR_params": np.array([[0.2, 0.5, -0.3, 0.1],
                                     [0.1, 0.1, 0.2, -0.1]]),
              "memb_func": 'sigmoid',
              # Setting 1
              "a_params": np.array([[1.5, 15.5],   # gauss: standard deviation  // sigmoid: steepness (gamma)
                                    [-15.5, -2.5]]),
              "c_params": np.array([[-3.0, -5.0],  # center of the curve or bell
                                    [3.0, 1.0]])}

# General Parameters
seed = 3
# int / None
logdir = 'logs/runs/'
use_tensorboard = False                # True / False
plot_dgp = False                        # True / False
plot_prediction = False                  # True / False
plot_mfs = True                         # True / False
plot_learningcurves = True              # True / False
show_initial_weights = True             # True / False
device = None  # 'cpu' / 'cuda' / None(=automatic detection)
##############################################################################
# Set seed for reproducibility
if seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
## Generate Data ##
S, S_train, S_valid, X, X_train, X_valid, y, y_train, y_valid, = sanfis_generator.gen_data_ts(
    n_obs, dgp_params, lag, test_size, shuffle_data, plot_dgp, save_path=logdir)

# hyperparameters to log (in addition)
hparams_dict = {'seed': str(seed),
                'bs': str(param['batch_size']),
                'shuffle_data': str(shuffle_data),
                'shuffle_batches': str(shuffle_batches),
                'epochs': str(param['n_epochs'])}
#%% ## make model / set loss function and optimizer##
fis = SANFIS(
    membfuncs, param['n_input'], device, scale=param['scaler'])  # .to(device)
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(fis.parameters(), lr=param['lr'])
# optimizer = torch.optim.SGD(fis.parameters(), lr=param['lr'])
# %% ## fit model ##
history = fis.fit([S_train, X_train, y_train], [S_valid, X_valid, y_valid],
                  optimizer, loss_function, param['batch_size'],
                  shuffle_batches, param['n_epochs'], param['patience'],
                  param['delta'], use_tensorboard, logdir, hparams_dict)
# %% eval model
y_pred = fis.predict(X)
y_pred_train = fis.predict([S_train, X_train])
y_pred_valid = fis.predict([S_valid, X_valid])
train_loss = loss_function(y_pred_train, y_train).item()
valid_loss = loss_function(y_pred_valid, y_valid).item()
premise_parameters = fis.premise
consequence_parameters = fis.consequence
# %% ## Plots
if plot_prediction:
    plottingtools.plt_prediction(y, y_pred, save_path=logdir)
if plot_mfs:
    fis.plotmfs(show_initial_weights, bounds=[
        [-10, 10], [-10, 10]], save_path='img/prediction_mfs.pdf', show_title=False)
if plot_learningcurves:
    plottingtools.plt_learningcurves(history)

# print(fis.premise)
# print(fis.consequence)  # print model weights
print(
    f'\nfinal rmse train loss: {np.sqrt(train_loss):.4f} \nfinal rmse valid loss: {np.sqrt(valid_loss):.4f}')

