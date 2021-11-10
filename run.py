"""Kaggle competition: Santander customer transaction prediction"""

# Imports
import importlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from santander.auxiliary import get_data, get_submission
# from deeplearning.data.loader import Loader
from santander.model import NN2
# from deeplearning.model.init import init_normal, init_xavier
# from deeplearning.trainer.trainer import Trainer

import deeplearning.runner

from deeplearning.metric import RocAuc
from deeplearning.reporter import ConsoleReporter


# Hyperparameters
# BATCH_SIZE = 1024
BATCH_SIZE = 128


# Data
# train_ds, val_ds, test_ds, test_ids = get_data(
#     train="files/aug_train.csv",
#     test="files/aug_test.csv",
#     submission=True)
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/tiny_train.csv",
    test="files/tiny_test.csv",
    submission=True)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True)
val_loader = DataLoader(
    dataset=val_ds,
    batch_size=BATCH_SIZE)
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=BATCH_SIZE)

# Model, Optimizer
model = NN2(input_size=400, hidden_dim=100)

# Metric reporter
console_reporter = ConsoleReporter()
rocauc = RocAuc()
rocauc.subscribe(console_reporter)


# Trainer
TRAINER = deeplearning.trainer.Trainer(
    model=model,
    optimizer=optim.Adam(
        params=model.parameters(),
        lr=2e-3,
        weight_decay=1e-4
    ),
    train_loader=train_loader,
    loss_fn=nn.BCELoss(),
    metrics=rocauc,
)

TRAINER.train(2)

TRAINER.reset()


# Export
get_submission(
    model=TRAINER.model,
    loader=test_loader,
    test_ids=test_ids,
    device="cuda",
    filename="sub-20210927.csv")


del(TRAINER)
importlib.reload(deeplearning.trainer)
