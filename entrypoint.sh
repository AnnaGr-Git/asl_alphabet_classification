#!/bin/sh
dvc pull
python -u src/data/makedataset.py --num_samples=10
python -u src/models/train_model.py

