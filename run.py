"""Kaggle competition: Santander customer transaction prediction"""

# Imports
import importlib

import torch.nn as nn
import torch.optim as optim
from deeplearning.metric import RocAuc, BinaryCrossentropy
from deeplearning.modelinit import init_xavier
from deeplearning.reporter import ConsoleReporter, TensorboardHparamReporter
from deeplearning.runner_mediator import Trainer, Validator, Runner
from torch.utils.data import DataLoader

from santander.auxiliary import get_data, get_submission
from santander.model import NN2

# Hyperparameters
# BATCH_SIZE = 1024
# BATCH_SIZE = 256
# BATCH_SIZE = 128
hparam = {
    "batchsize": 256,
    "lr": 2e-3
}


# Data
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/aug_train.csv",
    test="files/aug_test.csv",
    submission=True)
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/tiny_train.csv",
    test="files/tiny_test.csv",
    submission=True)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=hparam["batchsize"],
    shuffle=True)
val_loader = DataLoader(
    dataset=val_ds,
    batch_size=hparam["batchsize"])
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=hparam["batchsize"])

# Model, Optimizer
model = NN2(input_size=400, hidden_dim=100)
init_xavier(model)
optimizer = optim.Adam(
    params=model.parameters(),
    lr=hparam["lr"],
    weight_decay=1e-4
)

# Reporter
console_train = ConsoleReporter(name="Train")
tb_train = TensorboardHparamReporter(name="Train", hparam=hparam)
console_val = ConsoleReporter(name="Val")

# Metrics
rocauc_train = RocAuc()
rocauc_train.subscribe(console_train)
rocauc_train.subscribe(tb_train)
bce_train = BinaryCrossentropy()
bce_train.subscribe(console_train)
bce_train.subscribe(tb_train)

rocauc_val = RocAuc()
rocauc_val.subscribe(console_val)


loss_fn = nn.BCELoss()


# Trainer
TRAINER = Trainer(
    loader=train_loader,
    batch_metrics=bce_train,
    epoch_metrics=rocauc_train
)

VAL = Validator(
    loader=val_loader,
    metrics=rocauc_val
)

RUNNER = Runner(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    trainer=TRAINER,
    validator=VAL
)

RUNNER = Runner(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    trainer=TRAINER
)

RUNNER.train()
RUNNER.validate()
RUNNER.run(10)


# Export
get_submission(
    model=RUNNER.model,
    loader=test_loader,
    test_ids=test_ids,
    # device="cuda",
    device="cpu",
    filename="sub-20211122.csv")


del(TRAINER)
importlib.reload(deeplearning.runner_strategy)
