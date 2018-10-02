# Interpreting Machine Learning Algorithms with LIME
We have often found that Machine Learning (ML) algorithms capable of capturing structural non-linearities in training data - models that are sometimes referred to as 'black box' (e.g. Random Forests, Deep Neural Networks, etc.) - perform far better at prediction than their linear counterparts (e.g. Generalised Linear Models). They are, however, much harder to interpret - in fact, quite often it is not possible to gain any insight into why a particular prediction has been produced, when given an instance of input data (i.e. the model features). Consequently, it has not been possible to use 'black box' ML algorithms in situations where clients have sought cause-and-effect explanations for model predictions, with end-results being that sub-optimal predictive models have been used in their place, as their explanatory power has been more valuable, in relative terms.

This repository contains an example project for an alternative approach - we train a 'black box' ML model to the best of our abilities and then apply an ancillary algorithm to generate explanations for the predictions. More specifically, we will test the ability of the Local Interpretable Model-agnostic Explanations (LIME) algorithm, recently described by Ribiero et al (2016), to provide explanations for a Random Forest regressor trained on multiple-lot on-line auction data.

For reference, the paper that describes the LIME algorithm can be found here: https://arxiv.org/pdf/1602.04938v1.pdf (a version is also included as part of this repository); details of its implementation in Python (as used in this notebook), can be found here: https://github.com/marcotcr/lime/; while a more general discussion of ML algorithm interpretation (that includes LIME), can be found in the eBook by Christoph Molnar, which can be found here: https://christophm.github.io/interpretable-ml-book/.

## Project Dependencies

We use [pipenv](https://docs.pipenv.org) for managing project dependencies and Python environments (i.e. virtual environments). All of the direct packages dependencies required to run the code (e.g. NumPy for arrays/tensors and Pandas for DataFrames), as well as all the packages used during development (e.g. IPython and JupyterLab as the chosen development environment), are described in the `Pipfile`. Their **precise** downstream dependencies are described in `Pipfile.lock`.

### Installing Pipenv

To get started with Pipenv, first of all download it - assuming that there is a global version of Python available on your system and on the PATH, then this can be achieved by running the following command,

```bash
pip3 install pipenv
```

Pipenv is also available to install from many non-Python package managers. For example, on OS X it can be installed using the [Homebrew](https://brew.sh) package manager, with the following terminal command,

```bash
brew install pipenv
```

For more information, including advanced configuration options, see the [official pipenv documentation](https://docs.pipenv.org).

### Installing this Projects' Dependencies

Make sure that you're in the project's root directory (the same one in which the `Pipfile` resides), and then run,

```bash
pipenv install --dev
```

This will install all of the direct project dependencies as well as the development dependencies (the latter a consequence of the `--dev` flag).

### Running Python, IPython and JupyterLab from the Project's Virtual Environment

In order to continue development in a Python environment that precisely mimics the one the project was initially developed with, use Pipenv from the command line as follows,

```bash
pipenv run python3
```

The `python3` command could just as well be `ipython3` or the JupterLab, for example,

```bash
pipenv run jupyter lab
```

This will fire-up JupyterLab *where the default Python 3 kernel includes all of the direct and development project dependencies*. This is how we advise that the notebooks within this project are used.

### Pipenv Shells

Prepending `pipenv` to every command you want to run within the context of your Pipenv-managed virtual environment, can get very tedious. This can be avoided by entering into a Pipenv-managed shell,

```bash
pipenv shell
```

which is equivalent to 'activating' the virtual environment. Any command will now be executed within the virtual environment. Use `exit` to leave the shell session.