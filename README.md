# What is ARTM?

[BigARTM](https://github.com/bigartm/bigartm) is a powerful tool for [topic modeling](https://en.wikipedia.org/wiki/Topic_model) based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combination of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.

[python_artm](https://github.com/ilirhin/python_artm) is simple implementation of part of functionality of [BigARTM](https://github.com/bigartm/bigartm). It is written to simplify the process of the developing new features and checking new ideas.

The library is written primarily in python and designed for experiments over new functionality for [BigARTM](https://github.com/bigartm/bigartm).

# Install 
To develop clone the repository and call `make setup-venv` in the root of the repo. 

Usage of [virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) is highly recommended.

The repository contains three libraries:
1. `pyartm` - the implementation of ARTM, it's the main package with light dependencies.
2. `pyartm_datasets` - package to work with different datasets. It helps to do experiments. If you already have dataset you don't need this package.
3. `pyartm_experiments` - package with the different experiments using the previous packages and their results. 

For the moment the pypi installation is not supported, but installation from git is available:
```
env GIT_LFS_SKIP_SMUDGE=1 pip install "git+https://github.com/ilirhin/python_artm.git#egg=pyartm&subdirectory=pyartm"
env GIT_LFS_SKIP_SMUDGE=1 pip install "git+https://github.com/ilirhin/python_artm.git#egg=pyartm_datasets&subdirectory=pyartm_datasets"
env GIT_LFS_SKIP_SMUDGE=1 pip install "git+https://github.com/ilirhin/python_artm.git#egg=pyartm_experiments&subdirectory=pyartm_experiments"
```

`env GIT_LFS_SKIP_SMUDGE=1` is recommended if you want to speed up the installation.

# Datasets
You can get the datasets for the experiments by the [link](https://yadi.sk/d/BWPx6v-iYb_xuw). Download this directory. To pass the path to this directory to `artm` set the environment variable `PYARTM_DATASETS_PATH` (add it to `.bashrc` file):
```
export PYARTM_DATASETS_PATH=<path to the dowloaded unziped directory>
``` 
By default the path `~/pyartm-datasets` is used.

When installing `pyartm_datasets` you may have problems with Pattern library (it's a gensim's dependency required for lemmatization). See [here](https://bit.ly/2RMEC0W) for possible solutions on Mac. 

# Experiments

[The archive with the results of experiments](https://yadi.sk/d/yJqS9DDEvEJdaA)
