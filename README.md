# sanfis

This is a PyTorch-based implementation of my project S-ANFIS: [State-ANFIS: A Generalized Regime-Switching Model for Financial Modeling (2022)](https://ieeexplore.ieee.org/abstract/document/9776208). S-ANFIS is an generalization of Jang's [ANFIS: adaptive-network-based fuzzy inference system (1993)](https://ieeexplore.ieee.org/abstract/document/256541). 

## Installation
### Step 1: Install PyTorch
Install PyTorch on you device. Make sure to consider the correct operating system: Windows, macOS (Intel / Apple Silicon) or Linux. Everything is explained on the [developer's website](https://pytorch.org/get-started/locally/).

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
##Features

## Quick start


## Using the plain vanilla ANFIS network

## Using tensorboard

## Related work

## Contact

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