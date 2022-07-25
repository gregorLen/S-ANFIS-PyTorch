from sanfis import SANFIS
from sanfis.datagenerators import anfis_generator
from sanfis import plottingtools
import torch
# plain Vanilla ANFIS
# Set 4 input variables with 3 gaussian membership functions each
MEMBFUNCS = [
    {'function': 'gaussian',
     'n_memb': 3,
     'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0, 1.0],
                          'trainable': True}}},

    {'function': 'gaussian',
     'n_memb': 3,
     'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0, 1.0],
                          'trainable': True}}},

    {'function': 'gaussian',
     'n_memb': 3,
     'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0, 1.0],
                          'trainable': True}}},

    {'function': 'gaussian',
     'n_memb': 3,
     'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0, 1.0],
                          'trainable': True}}},

]

# generate some data (mackey chaotic time series)
X, X_train, X_valid, y, y_train, y_valid = anfis_generator.gen_data(data_id='mackey',
                                                                  n_obs=2080, n_input=4)

# create model
model = SANFIS(membfuncs=MEMBFUNCS,
               n_input=4,
               scale='Std')
optimizer = torch.optim.Adam(params=model.parameters())
loss_functions = torch.nn.MSELoss(reduction='mean')

# fit model
history = model.fit(train_data=[X_train, y_train],
                    valid_data=[X_valid, y_valid],
                    optimizer=optimizer,
                    loss_function=loss_functions,
                    epochs=200,
                    )

# predict data
y_pred = model.predict(X)

# plot learning curves
plottingtools.plt_learningcurves(history, save_path='img/learning_curves.pdf')

# plot prediction
plottingtools.plt_prediction(y, y_pred, save_path='img/mackey_prediction.pdf')

# %%
