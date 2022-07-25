# sanfis

This is a PyTorch-based implementation of my project S-ANFIS: [State-ANFIS: A Generalized Regime-Switching Model for Financial Modeling (2022)](https://ieeexplore.ieee.org/abstract/document/9776208). S-ANFIS is an generalization of Jang's [ANFIS: adaptive-network-based fuzzy inference system (1993)](https://ieeexplore.ieee.org/abstract/document/256541). The implemenation can easliy be used to fit an ANFIS network.

## 1. What is S-ANFIS

S-ANFIS is a simple generalization of the ANFIS network, where the input to the premise and the consequence part of the model can be controlled separately. As general notation, I call the input the premise part "state" variables ``s`` and the input of the consequence part "input" or "explanatory" variables ``x``. 

![S-ANFIS architecture](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/sanfis_architecture.png)

For an in-depth explaination, check out [our paper](https://ieeexplore.ieee.org/abstract/document/9776208).

## 2. Installation
This package is intended to be installed on top of PyTorch, so you need to do that first.
### Step 1: Install PyTorch
Make sure to consider the correct operating system: Windows, macOS (Intel / Apple Silicon) or Linux. Everything is explained on the [developer's website](https://pytorch.org/get-started/locally/). 

To ensure that PyTorch was installed correctly, verify the installation by running sample PyTorch code:

```python
import torch
x = torch.rand(5, 3)
print(x)
```
### Step 2: Install sanfis
sanfis can be installed via pip:

```bash
pip install sanfis
```

## 3. Quick start
First let's generate some data! The given example is an [AR(2)-process](https://en.wikipedia.org/wiki/Autoregressive_model) whoose AR-parameters depend on the regime of two independent state variables:

```python
# Load modules
import numpy as np
import torch
from sanfis import SANFIS, plottingtools
from sanfis.datagenerators import sanfis_generator

# seed for reproducibility
np.random.seed(3)
torch.manual_seed(3)
## Generate Data ##
S, S_train, S_valid, X, X_train, X_valid, y, y_train, y_valid, = sanfis_generator.gen_data_ts(
    n_obs=1000, test_size=0.33, plot_dgp=True)
```

![s-anfis data generating process](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/sanfis_dgp_process.png)

Set a list of membership functions for each of the state variables that enter the model:

```python
# list of membership functions
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
```
The given example uses two sigmoid functions for each state variable.

Now create the model, fit and evaluate:

```python
# make model / set loss function and optimizer
fis = SANFIS(membfuncs=membfuncs, n_input=2, scale='Std')
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(fis.parameters(), lr=0.005)

# fit model
history = fis.fit([S_train, X_train, y_train], [S_valid, X_valid, y_valid],
                  optimizer, loss_function, epochs=1000)
# eval model
y_pred = fis.predict([S, X])
plottingtools.plt_prediction(y, y_pred,
                             save_path='img/sanfis_prediction.pdf')
# plottingtools.plt_learningcurves(history)
```

![s-anfis prediction](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/sanfis_prediction.png)

## 4. Features
### 4.1 Membership functions
The implementation allows a very flexible usage of membership functions. For each input variable that enters the premise-part of the model, the type and number of membership functions can be flexibly chosen. As of today, three possible membership functions are implemented:

#### Gaussian
The Gaussian is described by 2 parameters, `mu` for the location and `sigma` for the wideness.

```python
# Example
gaussian_membfunc = {'function': 'gaussian',
			 'n_memb': 3,	 # 3 membership functions
			 'params': {'mu': {'value': [-2.0, 0.0, 1.5], 
			                'trainable': True},
			           'sigma': {'value': [1.0, 0.5, 1.0],
			               'trainable': True}}
			}
```

In this example, three membership functions are considered.

#### General bell-shaped
The general bell-shaped function is described by three parameters, `a` (wideness), `b` (shape) and `c` (location).

```python
bell_membfunc = {'function': 'bell',
			'n_memb': 2,
			'params': {'c': {'value': [-1.5, 1.5],
			                'trainable': True},
			            'a': {'value': [3.0, 1.0],
			                'trainable': False},
			            'b': {'value': [1.0, 3.0],
			                'trainable': False}}
					}
```
#### Sigmoid
The sigmoid is described by two parameters: `c` (location) and `gamma` (steepness).

```python
sigmoid_membfunc = {'function': 'sigmoid',
			'n_memb': 2,
			'params': {'c': {'value': [0.0, 0.0],
			                'trainable': True},
			            'gamma': {'value': [-2.5, 2.5],
			                    'trainable': True}}
}
```

Remember to add a list of membership functions as `membfunc` argument when creating the ```SANFIS``` oject, e.g.:

```python
MEMBFUNCS = [gaussian_membfunc, bell_membfunc, sigmoid_membfunc]
model = SANFIS(MEMBFUNCS, n_input=2)
model.plotmfs(bounds=[[-2.0, 2.0],  # plot bounds for first membfunc
                      [-4.0, 2.0],  # plot bounds for second membfunc
                      [-5.0, 5.0]],  # plot bounds fo third membfunc
              save_path='img/membfuncs.pdf')
```

![membership functions](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/membfuncs.png)

### 4.2 Tensorboard
Tensorboard provides visualization needed for machine learning experimentation. Further information can be found [here](https://www.tensorflow.org/tensorboard)

#### Step 1: Install tensorboard
```bash
pip install tensorboard
```

#### Step 2: enable tensorboard usage during training

Tensorboard functionality can be added via arguments in the `fit()` function, e.g.

```python
history = model.fit( ...
                    use_tensorboard=True,
                    logdir='logs/tb',
                    hparams_dict={}
                   )
```
Note that `hparams_dict` is an optional argument where you can store additional hyperparameters of you model, e.g. `hparams_dict={'n_input':2}`.

#### Step 3: Open tensorboard
```bash
tensorboard --logdir=logs/tb
```


## 5. Using the plain vanilla ANFIS network
![ANFIS architecture](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/anfis_architecture.png)

To use the plain vanilla ANFIS network, simply remove the state variables `s` from the training (`fit()`). This automatically sets the same input for premise and consequence part of the model.

```python
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
X, X_train, X_valid, y, y_train, y_valid = datagenerator.gen_data(data_id='mackey',
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

```


![learning curves](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/learning_curves.png)

![prediction mackey time series](https://github.com/gregorLen/sanfis-pytorch/blob/main/img/mackey_prediction.png)

## 6. Related work

- [AnfisTensorflow2.0](https://github.com/gregorLen/AnfisTensorflow2.0) by me
- [bare-bones implementation of ANFIS](https://github.com/twmeggs/anfis) (manual derivatives) by [twmeggs](https://github.com/twmeggs) 
- [PyTorch implementation](https://github.com/jfpower/anfis-pytorch) by [James Power](http://www.cs.nuim.ie/~jpower/)
- [simple ANFIS based on Tensorflow 1.15.2](https://github.com/tiagoCuervo/TensorANFIS) by [Santiago Cuervo](https://github.com/tiagoCuervo)

## Contact
I am very thankful for feedback. Also, if you have questions, please contact gregor.lenhard92@gmail.com

## References
If you use my work, please cite it appropriately:


G. Lenhard and D. Maringer, "State-ANFIS: A Generalized Regime-Switching Model for Financial Modeling," 2022 IEEE Symposium on Computational Intelligence for Financial Engineering and Economics (CIFEr), 2022, pp. 1-8, doi: 10.1109/CIFEr52523.2022.9776208.

BibTex:


```
@INPROCEEDINGS{lenhard2022sanfis,
  author={Lenhard, Gregor and Maringer, Dietmar},
  booktitle={2022 IEEE Symposium on Computational Intelligence for Financial Engineering and Economics ({CIFEr})}, 
  title={State-{ANFIS}: A Generalized Regime-Switching Model for Financial Modeling}, 
  year={2022},
  pages={1--8},
  doi={10.1109/CIFEr52523.2022.9776208},
  organization={IEEE}
  }
```