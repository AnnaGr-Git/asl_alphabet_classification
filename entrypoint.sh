#!/bin/sh
dvc pull
wandb login $WANDB_API_KEY
python -u src/data/make_dataset.py preprocess-command --num_samples=100
python -u src/models/train_model.py $EXP
