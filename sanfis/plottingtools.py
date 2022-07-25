import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pandas.core.frame import DataFrame
import os

plt.style.use('seaborn')                # default / ggplot / seaborn


def plt_prediction(y, y_pred, title='', save_path: Optional[str] = None):
    f, axs = plt.subplots(2, 1, figsize=(8, 6))
    f.subplots_adjust(hspace=0.3)
    f.suptitle(f'{title}', size=16)
    y, y_pred = np.array(y), np.array(y_pred)  # both to np array
    ymin, ymax = min(min(y), min(y_pred)), max(max(y), max(y_pred))
    max(max(y), max(y_pred))
    axs[0].plot(y, color='k')
    axs[0].plot(y_pred, alpha=1.0, color='orange')
    # axs[0].legend([r'$x_t$', r'$\hat{x}_t$'], prop={'size': 17})
    axs[0].grid(True)
    axs[0].set_ylim((ymin, ymax))
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    #axs[0].set_title('Data vs. Predicted values')
    # axs[1].plot(np.arange(y.shape[0]), y - y_pred)
    n_bins = int(len(y) / 20)
    limit = max(abs(y - y_pred))[0]
    axs[1].hist(y - y_pred, bins=n_bins, align='mid',
                range=[-limit, limit], facecolor='gray')
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    # axs[1].legend([r'$x_t - \hat{x}_t$'], prop={'size': 17})
    axs[1].grid(True)
    # axs[1].set_ylim((ymin, ymax))
    #axs[1].set_title('Prediction Error')
    f.legend([r'$y_t$', r'$\hat{y}_t$', r'$y_t - \hat{y}_t$'],
             fontsize=10, ncol=5, loc='upper center', prop={'size': 20})

    plt.show()

    if save_path != None:
        f.savefig(os.path.join(save_path),
                  bbox_inches='tight', pad_inches=0)


def plt_learningcurves(history: DataFrame, title: Optional[str] = None, save_path: Optional[str] = None):
    plot = history.plot(figsize=(8, 5), logy=True)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

    if save_path != None:
        fig = plot.get_figure()
        fig.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
