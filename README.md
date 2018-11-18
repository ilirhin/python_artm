# What is ARTM?

[BigARTM](https://github.com/bigartm/bigartm) is a powerful tool for [topic modeling](https://en.wikipedia.org/wiki/Topic_model) based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combination of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.

[python_artm](https://github.com/ilirhin/python_artm) is simple implementation of part of functionality of [BigARTM](https://github.com/bigartm/bigartm). It is written to simplify the process of the developing new features and checking new ideas.

The library is written primarily in python (except one function required for high performance) and designed for experiments over new functionality for [BigARTM](https://github.com/bigartm/bigartm).

# Install 
To develop clone the repository and call `pip install -e .` in the root of the repo. 

Usage of [virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) is highly recommended.

For the moment the pypi installation is not supported, but installation from git is available:
```
pip install git+https://github.com/ilirhin/python_artm.git
```

# Datasets
You can get the datasets for the experiments by the [link](https://yadi.sk/d/BWPx6v-iYb_xuw). Download this directory. To pass the path to this directory to `artm` set the environment variable `ARTM_DATASETS_PATH`:
```
export ARTM_DATASETS_PATH=<path to the dowloaded unziped directory>
``` 
By default the path `~/artm-datasets` is used.
