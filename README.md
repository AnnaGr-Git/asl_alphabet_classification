ASL Alphabet Classification
==============================
![example workflow](https://github.com/AnnaGr-Git/asl_alphabet_classification/actions/workflows/tests.yml/badge.svg)
<br/>
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

In this project, we will classify images of ASL-hand-signs into the corresponding letters.

## Getting Started

### Python environment

To run the code, a conda environment is required.
Check the [Conda website](https://www.anaconda.com/) for how to install it.

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:AnnaGr-Git/asl_alphabet_classification.git
   ```
2. Create and activate a new conda environment
   ```sh
   conda env create -f environment.yml
   conda activate MLOPS_project
   ```
3. Install further pre-requisites
   ```sh
   pip install -r requirements.txt
   ```

# Usage


## Fetch dataset

Run the following command.
```shell
dvc pull
```

After running the command, the `data/` directory will be created and the data will be downloaded under the `data/raw/` directory.

### Preprocessing

Run the following command to preprocess the data.
```shell
python src/data/make_dataset.py preprocess
```

## Train and Evaluate
Default training (config/experiment/exp1.yaml will be used):
```shell
python src/models/train_model.py
```
Run a training with hyperparameters defined in config/experiment/exp2.yaml
```shell
python src/models/train_model.py experiment=exp2
```
Change parameters (e.g. trainsize) of config in command-line:
```shell
python src/models/train_model.py experiment=exp2 experiment.trainsize=0.8
```


### Parameters

TODO: explain the parameters

## Docker

### Reproduce using the newest build with Docker image
The newest image created from the latest build from the repo can be pulled from the Google Cloud Container with the following command:
```bash
docker pull gcr.io/aslalphabet-374510/training:latest
```

### Run a training in the cloud
We are using Vertex AI to run a training in Google Cloud. To do that run the following:
```bash
gcloud ai custom-jobs create \
   --region=europe-west1 \
   --display-name=trainingrun \
   --config=config_cloud.yaml
```

## Continuous integration

### Run pytests
```bash
coverage run -m pytest tests/
```
or
```bash
pytest tests/
```

### Make coverage report
```bash
coverage report -m -i
```

### Pre-commits

Run pre-commits to check code formating, type hints and Docstring coverage.
```shell
pre-commit run --all-files
```

You can enable the pre-commit hooks to run everytime you do a commit using
```shell
pre-commit install
```
In case you want to commit without running the pre-commit hooks, do:
```shell
git commit -m <message> --no-verify
```

## Logging

We are using [Weights and Biases](https://wandb.ai/) to log the model training.

To authenticate the wandb in a docker container do as the following example:
```shell
docker run -e WANDB_API_KEY=<your-api-key> wandb:latest
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
