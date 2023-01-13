#!/bin/sh
dvc pull
python -u src/models/train_model.py