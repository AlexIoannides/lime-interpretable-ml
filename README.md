# Interpreting Black-Box Machine Learning Algorithms with LIME
We have often found that Machine Learning (ML) algorithms capable of capturing structural non-linearities in training data - models that are sometimes referred to as 'black box' (e.g. Random Forests, Deep Neural Networks, etc.) - perform far better at prediction than their linear counterparts (e.g. Generalised Linear Models). They are, however, much harder to interpret - in fact, quite often it is not possible to gain any insight into why a particular prediction has been produced, when given an instance of input data (i.e. the model features). Consequently, it has not been possible to use 'black box' ML algorithms in situations where clients have sought cause-and-effect explanations for model predictions, with end-results being that sub-optimal predictive models have been used in their place, as their explanatory power has been more valuable, in relative terms.

This repository contains an example project for an alternative approach - we train a 'black box' ML model to the best of our abilities and then apply an ancillary algorithm to generate explanations for the predictions. More specifically, we will test the ability of the Local Interpretable Model-agnostic Explanations (LIME) algorithm, recently described by Ribiero et al (2016), to provide explanations for a Random Forest regressor trained on multiple-lot on-line auction data.

For reference, the paper that describes the LIME algorithm can be found here: https://arxiv.org/pdf/1602.04938v1.pdf; details of its implementation in Python (as used in this notebook), can be found here: https://github.com/marcotcr/lime/; while a more general discussion of ML algorithm interpretation (that includes LIME), can be found in the eBook by Christoph Molnar, which can be found here: https://christophm.github.io/interpretable-ml-book/.

## Project Dependencies
This project was assembled using an isolated virtual environment, created with `virtualenv`. The exact details (e.g. versions) of the packages used are contained in the requirements.txt file found the root directory. At a very high-level this is essentially:
- Lime
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib
- IPython

The steps below walk-through the process of installing these dependencies and making them available to the Jupyter template notebook via a dedicated Python 3 kernel.

### Creating a Virtual Environment
I am assuming that the reader will be using OS X, is familiar with using the terminal and that they have Python 3 and the Jupyter package installed and made generally available (i.e. they are on the PATH). From within the root directory, run the following commands to create a virtual environment (called `venv`) and activate it,

```bash
python3 -m venv venv
source venv/bin/activate
```

To deactivate the virtual environment at a later date (e.g. after you run the steps below), just use `deactivate` from the command line.

### Install Dependencies from requirements.txt
To install the dependencies for this project, run the following commands from the terminal next,

```bash
pip install -r requirements.txt
```

### Creating a Jupyter Kernel for the Virtual Environment
Now that we have an isolated virtual environment for this project we need to be able to access it via Jupyter kernel. This is achieved with the following terminal commands (while the virtual environment is activated),

```bash
venv/bin/ipython kernel install --user --name=lime_proj
```

The `pymc3_proj` should now be available from within Jupyter to associate with the notebook in this repo! Should you need to remove the kernel at a later date, then the location of the `kernels` directory (which will contain a dedicated directory for the `pymc3_proj` kernel), can be found by running,

```bash
jupyter kernel --data-dir
```

Then, it's simply a matter of running `rm -rf kernels/pymc3_proj` from within the directory returned from the above command.
