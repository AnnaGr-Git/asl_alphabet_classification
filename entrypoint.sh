#!/bin/sh
dvc pull
python -u src/data/make_dataset.py preprocess --num_samples=10
python -u src/models/train_model.py

