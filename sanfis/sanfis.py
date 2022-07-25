# -----------------------------------------------------------
# Implementation of S-ANFIS
# (C) 2020 Gregor Lenhard
# email gregor.lenhard@unibas.ch
# -----------------------------------------------------------
# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# general modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Callable, Optional, Union
# sanfis modules
from .helpers import _RunManager, _FastTensorDataLoader, DataScaler


class SANFIS(nn.Module):
    def __init__(self, membfuncs: list, n_input: int, to_device: Optional[str] = None, scale: str = 'None', name: str = 'S-ANFIS'):
        """State-Adaptie Neuro-Fuzzy Inference System (S-ANFIS)

        Parameters
        ----------
        membfuncs : list
            List of membership functions. Supported functions are: ``'gaussian'``, ``'bell'``, and ``'sigmoid'``.

            Examples:
                {'function': 'gaussian',
                'n_memb': 3,
                'params': {'mu': {'value': [-2.0, 0.0, 1.5],
                                'trainable': True},
                            'sigma': {'value': [1.0, 0.5, 1.0],
                                    'trainable': True}}},

                {'function': 'bell',
                'n_memb': 2,
                'params': {'c': {'value': [-1.5, 1.5],
                                'trainable': True},
                            'a': {'value': [3.0, 1.0],
                                'trainable': False},
                            'b': {'value': [1.0, 3.0],
                                'trainable': False}}},

                {'function': 'sigmoid',
                'n_memb': 2,
                'params': {'c': {'value': [0.0, 0.0],
                                'trainable': True},
                            'gamma': {'value': [-2.5, 2.5],
                                    'trainable': True}}},

        n_input : int
            Number of Input values for the S-ANFIS system. Typically equivalent to ``X.shape[1]``.
        to_device : str
            On which device to run the model. ``'gpu'`` or ``'cpu'``.
        scale : str
            Type of scaling to be performed. Possible values are ``'Std'`` or ``'None'``.
        name : str
            Name of the model.
        """
        super(SANFIS, self).__init__()
        self._membfuncs = membfuncs
        self._memberships = [memb['n_memb'] for memb in membfuncs]
        self._rules = int(np.prod(self._memberships))
        self._s = len(membfuncs)
        self._n = n_input
        self.scaler = DataScaler(scale)
        self.name = name

        # build model
        self.layers = nn.ModuleDict({
            # Layer 1 - fuzzyfication
            'fuzzylayer': _FuzzyLayer(membfuncs),

            # Layer 2 - rule layer
            'rules': _RuleLayer(),

            # Layer 3 - normalization - is a simple function --> see forward pass

            # Layer 4 - consequence layer
            'consequence': _ConsequenceLayer(self._n, self._rules),

            # Layer 5 - weighted-sum - is a simple function
        })

        # save initial fuzzy weights
        self._initial_premise = self.premise

        # determine device (cuda / cpu) if not specifically given
        if to_device == None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = to_device

        self.to(self.device)

    # Network architecture is defined in terms of class properties
    # You shall not switch the architecture after creating an object of SANFIS
    def forward(self, S_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        """forward pass through the model.

        Args:
            S_batch (torch.Tensor): Tensor of state variables S.
            X_batch (torch.Tensor): Tensor of explanatory (independent) input variables X.

        Returns:
            torch.Tensor: output / prediction of dependent variable (y_hat)
        """
        # Layer 1 - fuzzyfication
        output1 = self.layers['fuzzylayer'](S_batch)

        # Layer 2 - rule layer
        output2 = self.layers['rules'](output1)

        # Layer 3 - normalization layer // output3 == wnorm
        output3 = F.normalize(output2, p=1, dim=1)

        # Layer 4 - consequence layer
        output4 = self.layers['consequence'](
            X_batch, output3)

        # Layer 5 - summation
        output5 = output4.sum(axis=1).reshape(-1, 1)

        return output5

    def _reset_model_parameters(self):
        """reset model parameters (for early stopping procedure)
        """
        optlcass = self.optimizer.__class__
        self.optimizer = optlcass(self.parameters(), lr=self.optimizer.__dict__[
            'param_groups'][0]['lr'])

        # reset parameters
        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def fit(self, train_data: List[torch.Tensor], valid_data: List[torch.Tensor], optimizer: Callable, loss_function: Callable, batch_size: int = 16, shuffle_batches: bool = True, epochs: int = 100, patience: int = 10, delta: float = 1e-5, use_tensorboard: bool = False, logdir: Optional[str] = None, hparams_dict: dict = {}, disable_output: bool = False) -> pd.DataFrame:
        """Model fitting function.

        Args:
            train_data (List[torch.Tensor]): Training data, e.g. [S_train, X_train, y_train] or [X_train, y_train]
            valid_data (List[torch.Tensor]): Validation data, e.g. [S_valid, X_valid, y_valid] or [X_valid, y_valid]
            optimizer (Callable): optimizer, e.g. torch.optim.Adam(model.paramters())
            loss_function (Callable): loss function from torch, e.g. nn.torch.MSELoss()
            batch_size (int, optional): batch size. Defaults to 16.
            shuffle_batches (bool, optional): controls of batches schall be shuffled. Defaults to True.
            epochs (int, optional): number of epochs. Defaults to 100.
            patience (int, optional): patience paramter for early stopping. How many consecutive deterioration of the loss are allowed. Defaults to 10.
            delta (float, optional): delta for the loss deterioration. Defaults to 1e-5.
            use_tensorboard (bool, optional): Wether to use tensorboard. Defaults to False.
            logdir (Optional[str], optional): Directory for tensorboard logs. Defaults to None.
            hparams_dict (dict, optional): Additional Hyperparamters to be stored in tensorboard. Defaults to {}.
            disable_output (bool, optional): Supress model progress print. Defaults to False.

        Returns:
            pd.DataFrame: Loss history.
        """
        assert len(train_data) == len(valid_data), \
            f'train_data and valid data must be both either list of 2 or 3 torch tensors. train_data is len {len(train_data)} and valid_data is len {len(valid_data)}.'
        assert len(train_data) == 2 or len(train_data) == 3, \
            f'List of training data must contain 2 or 3 tensors. For ANFIS [X_train, y_train]. For S-ANFIS [S_train, X_train, y_train]'

        # transform data to list if necessary
        if len(train_data) == 2:
            train_data = [train_data[0], train_data[0], train_data[1]]
            valid_data = [valid_data[0], valid_data[0], valid_data[1]]

            # store optimizer
        self.optimizer = optimizer

        # transform data to list

        # get dataloader
        train_dl = _FastTensorDataLoader(train_data,
                                         batch_size, shuffle_batches)
        valid_dl = _FastTensorDataLoader(valid_data,
                                         batch_size, shuffle_batches)

        # scale data
        train_dl_scaled = self.scaler.fit_transform(train_dl)
        valid_dl_scaled = self.scaler.transform(valid_dl)

        # set up run manager
        run_manager = _RunManager(epochs, hparams_dict, self.name, self.n_statevars, self.num_rules,
                                  self.n_input, patience, delta)
        # set up tensorboard
        if use_tensorboard:
            run_manager.get_writer(logdir)

        # print
        if not disable_output:
            print(
                f"Train s-anfis on {len(train_dl_scaled.dataset[0])} samples, validate on {len(valid_dl_scaled.dataset[0])} samples")

        # main training loop (via tqdm progress bar)
        with tqdm(total=epochs, ascii=True, desc='Training Loop', unit='epochs', disable=disable_output) as pbar:
            for epoch in range(epochs):
                # Training
                train_loss = []
                self.train()

                for sb_train, xb_train, yb_train in train_dl_scaled:
                    # send tensors to device (cpu/gpu)
                    sb_train = sb_train.to(self.device)
                    xb_train = xb_train.to(self.device)
                    yb_train = yb_train.to(self.device)
                    # forward pass & loss calculation
                    train_pred = self(sb_train, xb_train)
                    loss = loss_function(train_pred, yb_train)
                    train_loss.append(loss.detach())
                    # perform backward, update weights, zero gradients
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Validation
                with torch.no_grad():
                    self.eval()
                    valid_loss = []
                    sb_valid, xb_valid, yb_valid = valid_dl_scaled.dataset
                    y_pred_valid = self(sb_valid, xb_valid)
                    # TODO: should not be a list.
                    valid_loss = [
                        loss_function(yb_valid, y_pred_valid)]

                # end epoch (track losses & update pbar)
                run_manager(self.state_dict(), epoch,
                            train_loss, valid_loss, pbar)

                # check early stop and safe weights
                if run_manager.early_stop == True:
                    self._reset_model_parameters()
                    run_manager.reset_earlystopper()

        # load best model weights
        best_weights = run_manager.load_checkpoint()
        self.load_state_dict(best_weights)

        # end training / get reports
        run_manager.end_training()
        self.report, history = run_manager.get_report_history()

        return history

    def predict(self, input: Union[List[torch.Tensor], torch.Tensor], return_scaled: bool = False) -> Union[torch.Tensor, tuple]:
        """Predict output y from a given input [S, X]

        Args:
            input (List[torch.Tensor]): Input data, e.g. [S_data, X_data]
            return_scaled (bool, optional): Wether to return scaled data of y. Defaults to False.

        Returns:
            Union[torch.Tensor, list]: Output / prediction of y
        """
        if type(input) == torch.Tensor:
            input = [input, input]
        elif type(input) == list and len(input) == 1:
            input = [input[0], input[0]]
        elif type(input) == list and len(input) == 2:
            pass
        else:
            raise ValueError(
                f'input must be either a torch tensor (for ANFIS), a list of 1 torch tensor (for ANFIS) or a list of two torch tensors (for S-ANFIS).')

        # get dataloader
        dataloader = _FastTensorDataLoader(input, batch_size=1000)

        # scale data
        dataloader_scaled = self.scaler.transform(dataloader)
        dataloader_scaled.shuffle = False

        # predict
        with torch.no_grad():
            self.eval()
            y_pred_scaled = torch.tensor([])
            if len(input) == 2:
                for sb, xb in dataloader_scaled:
                    pred = self(sb.to(self.device), xb.to(self.device))
                    y_pred_scaled = torch.cat([
                        y_pred_scaled, pred]).reshape(-1, 1)
            else:
                for sb, xb, yb in dataloader_scaled:
                    pred = self(sb.to(self.device), xb.to(self.device))
                    y_pred_scaled = torch.cat([
                        y_pred_scaled, pred]).reshape(-1, 1)

        # rescale y_pred
        y_pred = self.scaler.transform_y(y_pred_scaled, inverse=True)

        # return data
        if return_scaled:
            y_scaled = dataloader_scaled.dataset[2]
            return y_pred, y_pred_scaled, y_scaled
        else:
            return y_pred

    def plotmfs(self, show_initial_weights=True, show_firingstrength: bool = True, bounds: Optional[list] = None, names: Optional[list] = None, title: Optional[str] = None, show_title: bool = True, save_path: Optional[str] = None):
        """Plots the membership functions.

        Args:
            show_initial_weights (bool, optional): Defaults to True.
            show_firingstrength (bool, optional): Show (normalized) firing strength as area plot. Defaults to True.
            bounds (Optional[list], optional): Bounds of the respective membership function. Defaults to None.
            names (Optional[list], optional): Names of the respective (state) variable. Defaults to None.
            title (str, optional): Title of the plot.
            show_title (bool, optional): Defaults to True.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        """
        # plot bounds
        if not bounds:
            lower_s = self.scaler.lower_s
            higher_s = self.scaler.higher_s
        else:
            lower_s, higher_s = list(zip(*bounds))

        # (scaled) state variables
        SN = torch.empty((1000, self._s))
        for i, (smin, smax) in enumerate(zip(lower_s, higher_s)):
            SN[:, i] = torch.linspace(smin, smax, 1000)
        SN_scaled = self.scaler.transform_S(SN)

        # membership curves
        with torch.no_grad():
            membership_curves = []
            for i, layer in enumerate(self.layers.fuzzylayer.fuzzyfication):
                membership_curves.append(
                    layer(SN_scaled[:, [i]]).detach().numpy())

        # initial membership curves
        with torch.no_grad():
            init_membership_curves = []
            for i, layer in enumerate(_FuzzyLayer(self._initial_premise).fuzzyfication):
                init_membership_curves.append(
                    layer(SN_scaled[:, [i]]).detach().numpy())

        # set plot names
        if names == None:
            plot_names = [
                f'State Variable {s+1} ({self.premise[s]["function"]})' for s in range(self.n_statevars)]
        else:
            plot_names = names

        # setup plot
        plt.style.use('seaborn')
        fig, ax = plt.subplots(
            nrows=self.n_statevars, ncols=1, figsize=(8, self.n_statevars * 3))
        if show_title:
            if title == None:
                fig.suptitle(f'Membership functions {self.name}', size=16)
            else:
                fig.suptitle(title, size=16)
        fig.subplots_adjust(hspace=0.4)

        # plot curves
        for s, curve in enumerate(membership_curves):
            ax[s].grid(True)
            ax[s].set_title(
                plot_names[s], size=19)
            # prepare colors
            colors = []
            for m in range(curve.shape[1]):
                colors.append(next(ax[s]._get_lines.prop_cycler)['color']
                              )
            # plot membfuncs for each statevar
            for m in range(curve.shape[1]):
                # color = next(ax[s]._get_lines.prop_cycler)['color']
                ax[s].plot(SN[:, s], curve[:, m], color=colors[m])
                if show_initial_weights:
                    ax[s].plot(SN[:, s], init_membership_curves[s][:, m],
                               '--', color=colors[m], alpha=.5)
            # show normalized memb_funcs
            if show_firingstrength:
                norm_curve = curve / curve.sum(1).reshape(-1, 1)
                ax[s].stackplot(
                    SN[:, s], [col for col in norm_curve.T], alpha=0.3, colors=colors)

            # ax[s].set_xticklabels(SN[:, s].detach().tolist(), fontsize=16)
            # ax[s].set_yticklabels(fontsize=16)
            ax[s].tick_params(axis='x', labelsize=14)
            ax[s].tick_params(axis='y', labelsize=14)

        plt.show()

        if save_path != None:
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    @property
    def n_statevars(self):
        return self._s

    @property
    def n_input(self):
        return self._n

    @property
    def memberships(self):
        return self._memberships

    @property
    def num_rules(self):
        return self._rules

    @property
    def premise(self):
        return [level.coeffs for level in self.layers.fuzzylayer.fuzzyfication]

    @premise.setter  # TODO: REFRESH
    def premise(self, new_memberships: list):
        self.layers.fuzzylayer = _FuzzyLayer(new_memberships)
        self._initial_premise = self.premise

    @property
    def consequence(self):
        return self.layers['consequence'].coeffs

    @consequence.setter
    def consequence(self, new_consequence: dict):
        self.layers['consequence'].coeffs = new_consequence

    @property
    def scaling_params(self):
        return self.scaler.scaler.__dict__


class _FuzzyLayer(nn.Module):
    def __init__(self, membfuncs):
        """Represents the fuzzy layer (layer 1) of s-anfis. Inputs will be fuzzyfied
        """
        super(_FuzzyLayer, self).__init__()
        self.n_statevars = len(membfuncs)

        fuzzyfication = nn.ModuleList()

        for membfunc in membfuncs:
            if membfunc['function'] == 'gaussian':
                MembershipLayer = _GaussianFuzzyLayer(
                    membfunc['params'], membfunc['n_memb'])
            elif membfunc['function'] == 'bell':
                MembershipLayer = _BellFuzzyLayer(
                    membfunc['params'], membfunc['n_memb'])
            elif membfunc['function'] == 'sigmoid':
                MembershipLayer = _SigmoidFuzzyLayer(
                    membfunc['params'], membfunc['n_memb'])
            else:
                raise Exception(
                    'Membership function must be either "gaussian", "bell", or "sigmoid".')

            fuzzyfication.append(MembershipLayer)

        self.fuzzyfication = fuzzyfication

    def reset_parameters(self):
        [layer.reset_parameters() for layer in self.fuzzyfication]

    def forward(self, input_):
        assert input_.shape[1] == self.n_statevars,\
            f'Number of State Variables in the input ({input_.shape[1]}) and network membershinputfunctions ({self.n_statevars}) dont match.'
        output = [Layer(input_[:, [i]])
                  for i, Layer in enumerate(self.fuzzyfication)]

        return output


class _GaussianFuzzyLayer(nn.Module):
    def __init__(self, params: dict, n_memb: int):
        """Represents the gaussian fuzzy layer (layer 1) of s-anfis. Inputs will be fuzzyfied
        """
        super(_GaussianFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._mu = torch.tensor([params['mu']['value']])
        self._sigma = torch.tensor([params['sigma']['value']])

        if params['mu']['trainable'] == True:
            self._mu = nn.Parameter(self._mu)

        if params['sigma']['trainable'] == True:
            self._sigma = nn.Parameter(self._sigma)

    @property
    def coeffs(self):
        return {'function': 'gaussian',
                'n_memb': self.m,
                'params': {'mu': {'value': self._mu.data.clone().flatten().tolist(),
                                  'trainable': isinstance(self._mu, nn.Parameter)},
                           'sigma': {'value': self._sigma.data.clone().flatten().tolist(),
                                     'trainable': isinstance(self._sigma, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._mu[:] = torch.tensor([self.params['mu']['value']])
            self._sigma[:] = torch.tensor([self.params['sigma']['value']])

    def forward(self, input_):
        output = torch.exp(
            - torch.square(
                (input_.repeat(
                    1, self.m).reshape(-1, self.m) - self._mu)
                / self._sigma.square()
            )
        )
        return output


class _BellFuzzyLayer(nn.Module):
    def __init__(self, params: dict, n_memb: int):
        """Represents the bell-shaped fuzzy layer (layer 1) of S-ANFIS. Inputs will be fuzzyfied
        """
        super(_BellFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._c = torch.tensor([params['c']['value']])
        self._a = torch.tensor([params['a']['value']])
        self._b = torch.tensor([params['b']['value']])

        if params['a']['trainable'] == True:
            self._a = nn.Parameter(self._a)

        if params['b']['trainable'] == True:
            self._b = nn.Parameter(self._b)

        if params['c']['trainable'] == True:
            self._c = nn.Parameter(self._c)

    @property
    def coeffs(self):
        return {'function': 'bell',
                'n_memb': self.m,
                'params': {'c': {'value': self._c.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._c, nn.Parameter)},
                           'a': {'value': self._a.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._a, nn.Parameter)},

                           'b': {'value': self._b.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._b, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._c[:] = torch.tensor([self.params['c']['value']])
            self._a[:] = torch.tensor([self.params['a']['value']])
            self._b[:] = torch.tensor([self.params['b']['value']])

    def forward(self, input_):

        output = 1 / (1 + torch.pow(((input_.repeat(1,
                                                    self.m).view(-1, self.m) - self._c).square() / self._a), self._b))

        return output


class _SigmoidFuzzyLayer(nn.Module):
    """Represents the sigmoid fuzzy layer (layer 1) of s-anfis. Inputs will be fuzzyfied
    """

    def __init__(self, params: dict, n_memb: int):
        super(_SigmoidFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._c = torch.tensor([params['c']['value']])
        self._gamma = torch.tensor([params['gamma']['value']])

        if params['c']['trainable'] == True:
            self._c = nn.Parameter(self._c)
        if params['gamma']['trainable'] == True:
            self._gamma = nn.Parameter(self._gamma)

    @property
    def coeffs(self):
        return {'function': 'sigmoid',
                'n_memb': self.m,
                'params': {'c': {'value': self._c.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._c, nn.Parameter)},
                           'gamma': {'value': self._gamma.data.clone().flatten().tolist(),
                                     'trainable': isinstance(self._gamma, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._c[:] = torch.tensor([self.params['c']['value']])
            self._gamma[:] = torch.tensor([self.params['gamma']['value']])

    def forward(self, input_):

        # = 1 / (1 + e^(- input_))
        output = torch.sigmoid(
            self._gamma * (input_.repeat(1, self.m).view(-1, self.m) - self._c))

        return output


class _RuleLayer(nn.Module):
    def __init__(self):
        """Rule layer / layer 2 of the S-ANFIS network
        """
        super(_RuleLayer, self).__init__()

    def forward(self, input_):
        batch_size = input_[0].shape[0]
        n_in = len(input_)

        if n_in == 2:
            output = input_[0].view(batch_size, -1, 1) * \
                input_[1].view(batch_size, 1, -1)

        elif n_in == 3:
            output = input_[0].view(batch_size, -1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1) * \
                input_[2].view(batch_size, 1, 1, -1)

        elif n_in == 4:
            output = input_[0].view(batch_size, -1, 1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1, 1) * \
                input_[2].view(batch_size, 1, 1, -1, 1) * \
                input_[3].view(batch_size, 1, 1, 1, -1)

        elif n_in == 5:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1, 1, 1) * \
                input_[2].view(batch_size, 1, 1, -1, 1, 1) * \
                input_[3].view(batch_size, 1, 1, 1, -1, 1) * \
                input_[4].view(batch_size, 1, 1, 1, 1, -1)

        elif n_in == 6:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1, 1, 1, 1) * \
                input_[2].view(batch_size, 1, 1, -1, 1, 1, 1) * \
                input_[3].view(batch_size, 1, 1, 1, -1, 1, 1) * \
                input_[4].view(batch_size, 1, 1, 1, 1, -1, 1) * \
                input_[5].view(batch_size, 1, 1, 1, 1, 1, -1)
        elif n_in == 7:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1) * \
                input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1) * \
                input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1) * \
                input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1) * \
                input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1) * \
                input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1)
        elif n_in == 8:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1, 1) * \
                input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1, 1) * \
                input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1, 1) * \
                input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1, 1) * \
                input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1, 1) * \
                input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1, 1) * \
                input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1, 1) * \
                input_[7].view(batch_size, 1, 1, 1, 1, 1, 1, 1, -1)
        else:
            raise Exception(
                f"Model Supports only 2,3,4,5,6,7 or 8 input variables but {n_in} were given.")

        output = output.reshape(batch_size, -1)

        return output


class _ConsequenceLayer(nn.Module):
    def __init__(self, n_input, n_rules):
        """Consequence layer / layer 4 of the S-ANFIS network
        """
        super(_ConsequenceLayer, self).__init__()
        self.n = n_input
        self.rules = n_rules

        # weights
        self._weight = nn.Parameter(torch.Tensor(self.n, n_rules))
        self._bias = nn.Parameter(torch.Tensor(1, n_rules))
        self.reset_parameters()

    @property
    def coeffs(self):
        return {'bias': self._bias,
                'weight': self._weight}

    @coeffs.setter
    def coeffs(self, new_coeffs: dict):
        assert type(
            new_coeffs) is dict, f'new coeffs should be dict filled with torch parameters, but {type(new_coeffs)} was given.'
        assert self._bias.shape == new_coeffs['bias'].shape and self._weight.shape == new_coeffs['weight'].shape, \
            f"New coeff 'bias' should be of shape {self._bias.shape}, but is instead {new_coeffs['bias'].shape} \n" \
            f"New coeff 'weight' should be of shape {self._weight.shape}, but is instead {new_coeffs['weight'].shape}"

        # transform to torch Parameter if any coeff is of type numpy array:
        if any(type(coeff) == np.ndarray for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(torch.from_numpy(
                new_coeffs[key]).float()) for key in new_coeffs}

        # transform to torch Parameter if any coeff is of type torch.Tensor:
        if any(type(coeff) == torch.Tensor for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(
                new_coeffs[key].float()) for key in new_coeffs}

        self._bias = new_coeffs['bias']
        self._weight = new_coeffs['weight']

    def reset_parameters(self):
        with torch.no_grad():
            self._weight[:] = torch.rand(
                self.n, self.rules) - 0.5

            self._bias[:] = torch.rand(1, self.rules) - 0.5

    def forward(self, input_, wnorm):
        output = wnorm * (torch.matmul(input_, self._weight) + self._bias)
        return output
