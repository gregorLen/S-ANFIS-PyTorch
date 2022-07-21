# sanfis

This is a PyTorch-based implementation of my project S-ANFIS: [State-ANFIS: A Generalized Regime-Switching Model for Financial Modeling (2022)](https://ieeexplore.ieee.org/abstract/document/9776208). S-ANFIS is an generalization of Jang's [ANFIS: adaptive-network-based fuzzy inference system (1993)](https://ieeexplore.ieee.org/abstract/document/256541). The implemenation can easliy be used to fit an ANFIS network.

## 1. What is S-ANFIS

S-ANFIS is a simple generalization of the ANFIS network, where the input to the premise and the consequence part of the model can be controlled separately. As general notation, I call the input the premise part "state" variables ``s`` and the input of the consequence part "input" or "explanatory" variables ``x``. 

![S-ANFIS architecture](https://github.com/gregorLen/sanfis-pytorch/blob/main/sanfis_architecture.pdf?raw=true)

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

```python
from sanfis import SANFIS
import torch

# a list of membership functions
MEMBFUNCS = [
       {'function': 'gaussian',
     'n_memb': 2,
     'params': {'mu': {'value': [-0.5, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0],
                          'trainable': True}}},

    {'function': 'gaussian',
     'n_memb': 2,
     'params': {'mu': {'value': [-0.5, 0.5],
                       'trainable': True},
                'sigma': {'value': [1.0, 1.0],
                          'trainable': True}}},
]

# Create model
model = SANFIS(membfuncs=MEMBFUNCS,
               n_input=2,
               scale='Std'
               )
optimizer = torch.optim.Adam(params=model.parameters())
loss_functions = torch.nn.MSELoss(reduction='mean')

# Generate some sample data
S_train, S_valid = torch.randn(100, 2), torch.randn(10, 2)
X_train, X_valid = torch.randn(100, 2), torch.randn(10, 2)
y_train, y_valid = torch.randn(100, 1), torch.randn(10, 1)

# Fit the model
history = model.fit(train_data=[S_train, X_train, y_train],
                    valid_data=[S_valid, X_valid, y_valid],
                    optimizer=optimizer,
                    loss_function=loss_functions,
                    batch_size=100,
                    epochs=10
                    )

# Plot resultsing membership functions 
model.plotmfs(
    bounds=[[-2.0, 2.0], [-2.0, 2.0]],
)
```

## 4. Features
### 4.1 Membership functions
The implementation allows a very flexible usage of membership functions. For each input variable that enters the premise-part of the model, the type and number of membership functions can be flexibly chosen. As of today, three possible membership functions are implemented:

#### Gaussian
```python
# Example
gaussian_membfunc = {'function': 'gaussian',
						 'n_memb': 3,
						 'params': {'mu': {'value': [-2.0, 0.0, 1.5],
						                'trainable': True},
						           'sigma': {'value': [1.0, 0.5, 1.0],
						               'trainable': True}}
						}
```

#### General bell-shaped
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
model = SANFIS(membfuncs=MEMBFUNCS,
			   n_input = ...
			   scale = ...
			   )
```

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
To use the plain vanilla ANFIS network, simply remove the state variables `s` from the training:

```python
model = SANFIS(membfuncs=MEMBFUNCS,
               n_input=2,
               scale='Std')
optimizer = torch.optim.Adam(params=model.parameters())
loss_functions = torch.nn.MSELoss(reduction='mean')
history = model.fit(train_data=[X_train, y_train],
                    valid_data=[X_valid, y_valid],
                    optimizer=optimizer,
                    loss_function=loss_functions,
                    )
```

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
