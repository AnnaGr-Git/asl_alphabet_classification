ASL Alphabet Classification
==============================

In this project, we will classify images of ASL-hand-signs into the corresponding letters.

## Reproduce using the newest build with Docker image:
The newest image created from the latest build from the repo can be pulled from the Google Cloud Container with the following command:
```bash
docker pull gcr.io/aslalphabet-374510/testing1:latest  
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Project Description
------------

Anna Grillenberger - s213637
Kristof Drexler - s220279
Victor Lyra - s220285
Daniel Schober - s212599


### Overall goal of the project
Wee want to translate the hand signs of the American Sign Language (ASL) to letters. To do this, we will develop a deep-learning model for detecting the handsigns from images.
### What framework are you going to use (PyTorch Image Models - timm)
We will use timm for developing our models. As the project matures, and we add new frameworks, we will update the readme.
### How to you intend to include the framework into your project
We plan on utilizing one of the strengths of the Transformers framework which is that it provides thousands of pretrained models to perform different tasks. As a starting point we intend to use some of the pretrained models on our data and then see how we can further improve from there.
### What data are you going to run on (initially, may change)
We are using the Kaggle dataset on [ASL alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The dataset contains a `train` and a `test` folder. The `test` folder has 1 image for each letter in the alphabet. The training data has 87 thousand images of 200x200 pixels. The dataset is labeled with one of 29 classes, these are the 26 letters of the alphabet, SPACE, DELETE and NOTHING. The images have  some variation, this includes different lighting conditions, hand sizes, distance of hand from camera.
### What deep learning models do you expect to use
We will use a pretrained version of resnet, and finetune it on our data [RESNET](https://huggingface.co/docs/timm/models/resnet). RESNET is a model that learns residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.
