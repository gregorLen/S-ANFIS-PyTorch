# Torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# general modules
import pandas as pd
import numpy as np
import copy
from sklearn.exceptions import NotFittedError
import copy
import time
from sklearn.exceptions import NotFittedError


def equally_spaced_initializer(n_memb: int, n_statevars: int, minval: float = -1.5, maxval: float = 1.5) -> torch.Tensor:
    """Custom weight initializer: equally spaced weights along an operating range of [minval, maxval].

    Args:
        n_memb (int): no. of membership functions.
        n_statevars (int): no. of state variables.
        minval (float, optional): minimum value. Defaults to -1.5.
        maxval (float, optional): maximum value. Defaults to 1.5.

    Returns:
        torch.Tensor: Initialized parameters.
    """
    linspace = torch.reshape(torch.linspace(minval, maxval, n_memb),
                             (-1, 1))
    return nn.Parameter(linspace.repeat(1, n_statevars))


class _RunManager():
    def __init__(self, epochs: int, hparams_dict: dict, name: str, n_statevars: int, n_rules: int, n_input: int, patience: int = 10, delta: int = 0.0001):
        """Run Manager keeps track of epochs, (best) losses and prints the progress bars. Also controls the tensorboard.

        Args:
            epochs (int): No. of epochs.
            hparams_dict (dict): (Additional) hyperparameters to store in tensorboard.
            name (str)
            n_statevars (int): No. of state variables.
            n_rules (int): No. of rules.
            n_input (int): No. of inputs.
            patience (int, optional): Patience parameter. Defaults to 10.
            delta (int, optional): Delta parameter. Defaults to 0.0001.
        """

        # sanfis parameters
        self.epochs = epochs
        self.name = name
        self.hparams_dict = hparams_dict
        self.n_statevars = n_statevars
        self.n_input = n_input
        self.n_rules = n_rules

        # early stopping criteria
        self.patience = patience
        self.iter = 0
        self.counter = 0
        self.best_loss = float('inf')
        self.global_best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

        # train and valid curve
        self.train_curve_iter = []     # train loss per iteration
        self.train_curve = []    # train loss per epoch
        self.valid_curve = []    # valid loss per epoch

        # epoch counter
        self.epoch = 0

        # progress bar
        self.start_time = time.time()
        self.pbar_step = self.epochs / 100
        self.tbwriter = None

    def __call__(self, model_weights, epoch, train_loss, valid_loss, pbar):
        self.epoch += 1

        # track losses
        self._track_losses(train_loss, valid_loss)

        loss = self.valid_curve[-1].item()

        # check early stop criteria
        if loss + self.delta < self.best_loss:
            self.best_loss = loss
            self.epoch = epoch
            self.counter = 0
            if loss < self.global_best_loss:
                self.global_best_loss = loss
                self.save_checkpoint(model_weights)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # update progress bar
        if self.epoch % self.pbar_step == 0:
            pbar.update(self.pbar_step)
            pbar.set_postfix(
                train_loss=round(self.train_curve[-1].item(), 5), valid_loss=round(self.valid_curve[-1].item(), 5))

    def get_writer(self, logdir):
        if logdir == None:
            logdir = 'logs/runs/'

        logDATE = __import__("datetime").datetime.now().strftime(
            '%Y_%m_%d_%H%M%S')

        logHPARAM = ''.join(
            [f'_{d}{self.hparams_dict[d]}' for d in self.hparams_dict])
        logNAME = f'-S{self.n_statevars}_N{self.n_input}_R{self.n_rules}{logHPARAM}'

        writer = SummaryWriter(logdir + logDATE + logNAME, comment=logNAME)
        # writer = SummaryWriter(comment=logNAME) # alternative

        self.tbwriter = writer

    def save_checkpoint(self, weights):
        self.best_weights = copy.deepcopy(weights)

    def load_checkpoint(self):
        return self.best_weights

    def reset_earlystopper(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def _track_losses(self, train_loss, valid_loss):
        self.train_curve.append(sum(train_loss) / len(train_loss))
        self.train_curve_iter.extend(train_loss)
        self.valid_curve.append(sum(valid_loss) / len(valid_loss))

        if self.tbwriter:
            self.tbwriter.add_scalar(
                'Loss/train', self.train_curve[-1], self.epoch)
            self.tbwriter.add_scalar(
                'Loss/valid', self.valid_curve[-1], self.epoch)
            # #  add histograms of weights
            # for name, weight in self.named_parameters():
            #     writer.add_histogram(name, weight, epoch)

    def end_training(self):
        self.run_time = time.time() - self.start_time

        if self.tbwriter:
            # log hparams
            HPARAMS = {"model": self.name,
                       "n_statevars": self.n_statevars,
                       "n_input": self.n_input,
                       "n_rules": self.n_rules,
                       **self.hparams_dict}
            self.tbwriter.add_hparams(HPARAMS,
                                      {
                                          "train_loss": self.train_curve[-1],
                                          "valid_loss": self.global_best_loss,
                                      },
                                      )
            # close writer
            self.tbwriter.flush()
            self.tbwriter.close()

    def get_report_history(self):

        report = {**self.hparams_dict,
                  "model": self.name,
                  "n_statevars": self.n_statevars,
                  "n_input": self.n_input,
                  "n_rules": self.n_rules,
                  "run_time": self.run_time}

        history = pd.DataFrame({'train_curve': np.array(
            self.train_curve), 'valid_curve': np.array(self.valid_curve)}).rename_axis('epoch')

        return report, history


class _FastTensorDataLoader():
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, tensors, batch_size=32, shuffle=False):
        """
        Initialize a _FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A _FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.dataset = tensors

        self.dataset_len = self.dataset[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices)
                          for t in self.dataset)
        else:
            batch = tuple(t[self.i:self.i + self.batch_size]
                          for t in self.dataset)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class StandardScaler(object):
    def __init__(self):
        self.fitted = False

    def fit(self, train_dl: _FastTensorDataLoader):
        self.S_mean, self.X_mean, self.y_mean = [
            data.mean(axis=0) for data in train_dl.dataset]
        self.S_std, self.X_std, self.y_std = [
            data.std(axis=0) for data in train_dl.dataset]
        self.fitted = True

    def transform(self, dataloader: _FastTensorDataLoader) -> _FastTensorDataLoader:
        if not self.fitted:
            raise NotFittedError(
                'Error: The StandardScaler instance is not yet fitted. Call "fit" with appropriate arguments before using this estimator.')

        transformed_dataloder = copy.deepcopy(dataloader)

        transformed_dataloder.dataset[0] = (
            transformed_dataloder.dataset[0] - self.S_mean) / self.S_std
        transformed_dataloder.dataset[1] = (
            transformed_dataloder.dataset[1] - self.X_mean) / self.X_std

        # correct for zero division
        transformed_dataloder.dataset[1][transformed_dataloder.dataset[1].isnan(
        )] = 0.0

        # if real output required
        if len(transformed_dataloder.dataset) == 3:
            transformed_dataloder.dataset[2] = (
                transformed_dataloder.dataset[2] - self.y_mean) / self.y_std

        return transformed_dataloder

    def transform_X(self, X, inverse=False):
        if inverse:
            return X * self.X_std + self.X_mean
        return (X - self.X_mean) / self.X_std

    def transform_S(self, S, inverse=False):
        if inverse:
            return S * self.S_std + self.S_mean
        return (S - self.S_mean) / self.S_std

    def transform_y(self, y, inverse=False):
        if inverse:
            return y * self.y_std + self.y_mean
        return (y - self.y_mean) / self.y_std

    def fit_transform(self, dataloader: _FastTensorDataLoader):
        self.fit(dataloader)
        return self.transform(dataloader)


class NoneScaler(object):
    """
    This class does nothing.
    """

    def fit(self, train_dl: _FastTensorDataLoader):
        pass

    def transform(self, dataloader: _FastTensorDataLoader):
        return dataloader

    def transform_X(self, X, inverse=False):
        return X

    def transform_S(self, S, inverse=False):
        return S

    def transform_y(self, y, inverse=False):
        return y

    def fit_transform(self, dataloader: _FastTensorDataLoader):
        self.fit(dataloader)
        return self.transform(dataloader)


class DataScaler(object):

    """Class to perform data scaling operations
    The scaling technique is defined by the ``scaler`` parameter which takes one of the 
    following values: 
    - ``'Std'`` for standarizing the data to follow a normal distribution. 
    - ``'None'`` No transformation at all. 

    ----------
    normalize : str
        Type of scaling to be performed. Possible values are ``'Std'`` or  ``None``.
    """

    def __init__(self, scaler: str = 'Std'):

        if scaler == 'Std':
            self.scaler = StandardScaler()
        elif scaler == 'None':
            self.scaler = NoneScaler()
        else:
            raise ValueError(
                f"Scaler can normalize via 'Std' or 'None', but {scaler} was given.")

    def fit_transform(self, train_dl: _FastTensorDataLoader):
        """Method that estimates an scaler object using the data in ``dataset`` and scales the data in  ``dataset``

        """

        return self.scaler.fit_transform(train_dl)

    def transform(self, dataloader: _FastTensorDataLoader) -> _FastTensorDataLoader:
        """Method that scales the data in ``dataloader``
        """
        # store information from the data for plotting purposes
        # see plotmfs() from sanfis class
        self.lower_s = [np.percentile(s, 5) for s in dataloader.dataset[0].T]
        self.higher_s = [np.percentile(s, 95) for s in dataloader.dataset[0].T]
        # self.max_s =
        return self.scaler.transform(dataloader)

    def transform_X(self, X, inverse: bool = False):
        return self.scaler.transform_X(X, inverse)

    def transform_S(self, S, inverse: bool = False):
        return self.scaler.transform_S(S, inverse)

    def transform_y(self, y, inverse: bool = False):
        return self.scaler.transform_y(y, inverse)
