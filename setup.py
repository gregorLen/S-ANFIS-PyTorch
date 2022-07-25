import pathlib
from setuptools import setup, find_packages
from os import path

HERE = pathlib.Path(__file__).parent

PACKAGE_NAME = 'sanfis'
VERSION = '0.1.0'
AUTHOR = 'Gregor Lenhard'
AUTHOR_EMAIL = 'gregor.lenhard92@gmail.com'
URL = 'https://github.com/gregorLen/S-ANFIS-PyTorch'

LICENSE = 'MIT License'
DESCRIPTION = 'Implementation to the State-Adaptive Neurofuzzy Inference System (S-ANFIS) network'
LONG_DESC_TYPE = "text/markdown"

CLASSIFIERS = [
    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
],

INSTALL_REQUIRES = [
    'numpy',
    'seaborn',
    'pandas',
    'matplotlib',
    'tqdm',
    'scikit-learn',
    'tensorboard'
]

DEV_REQUIRES = {
    "dev": ['pytest',
            ]
}

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      extras_require=DEV_REQUIRES,
      packages=find_packages(),
      py_modules=['sanfis', 'sanfis/datagenerators'],
      # package_dir={'': 'sanfis'},
      )
