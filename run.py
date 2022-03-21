"""Kaggle competition: Santander customer transaction prediction"""

# Imports
import importlib

import torch.nn as nn
import torch.optim as optim
from tinydl.hyperparameter import Hyperparameter
from tinydl.metric import BinaryCrossentropy, RocAuc
from tinydl.metric2 import BinaryCrossentropy2, RocAuc2
from tinydl.modelinit import init_xavier
from tinydl.reporter import ConsoleReporter, TensorboardHparamReporter, TensorboardScalarReporter
from tinydl.reporter2 import ConsoleReporter2, TensorboardHparamReporter2, TensorboardScalarReporter2
from tinydl.runner import Runner, Trainer, Validator
from tinydl.stage import Stage
from torch.utils.data import DataLoader

from santander.auxiliary import get_data, get_submission
from santander.model import NN2


# Dataset
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/aug_train.csv",
    test="files/aug_test.csv",
    submission=True)
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/tiny_train.csv",
    test="files/tiny_test.csv",
    submission=True)

# Loss function
loss_fn = nn.BCELoss()


# Hyperparameters
hparam = {
    "batchsize": [128, 1024],
    "lr": [2e-3, 2e-4]
}
hparam = {
    "batchsize": [128],
    "lr": [2e-4]
}

hyper = Hyperparameter(hparam)

#################
experiment = next(hyper.get_experiment())

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=experiment["batchsize"],
    shuffle=True)
val_loader = DataLoader(
    dataset=val_ds,
    batch_size=experiment["batchsize"])
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=experiment["batchsize"])

model = NN2(input_size=400, hidden_dim=100)
init_xavier(model)
optimizer = optim.Adam(
    params=model.parameters(),
    lr=experiment["lr"],
    weight_decay=1e-4
)

console_train = ConsoleReporter(name="Train")
tb_train = TensorboardHparamReporter(name="Train", hparam=experiment)
tb_scalar_train = TensorboardScalarReporter(name="Train", hparam=experiment)

rocauc_train = RocAuc()
rocauc_train.subscribe(console_train)
rocauc_train.subscribe(tb_train)

bce_train = BinaryCrossentropy()
bce_train.subscribe(console_train)

bce_train_batch = BinaryCrossentropy()
bce_train_batch.subscribe(tb_scalar_train)
bce_train_batch.subscribe(tb_train)


console_val = ConsoleReporter(name="Valid")
bce_val = BinaryCrossentropy()
bce_val.subscribe(console_val)
# rocauc_val = RocAuc()
# rocauc_val.subscribe(console_val)

TRAINER = Trainer(
    loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_metrics=bce_train_batch,
    # epoch_metrics=rocauc_train
    epoch_metrics=bce_train,
)

VAL = Validator(
    loader=val_loader,
    batch_metrics=[],
    epoch_metrics=bce_val
)

RUNNER = Runner(
    model=model,
    trainer=TRAINER,
    validator=VAL
)

RUNNER.run(3)

#####################################
for num_experiment, experiment in enumerate(hyper.get_experiment()):
    print(f"Experiment: {num_experiment+1}, Config: {experiment}")

    # DataLoader
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=experiment["batchsize"],
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=experiment["batchsize"])
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=experiment["batchsize"])

    # Model, Optimizer
    model = NN2(input_size=400, hidden_dim=100)
    init_xavier(model)
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=experiment["lr"],
        weight_decay=1e-4
    )

    # Reporter
    console_reporter = ConsoleReporter()
    tensorboard_reporter = TensorboardHparamReporter(hparam=experiment)

    # Metrics
    bce_train = BinaryCrossentropy()
    bce_train.subscribe(console_reporter)
    bce_train.subscribe(tensorboard_reporter)

    rocauc_train = RocAuc()
    rocauc_train.subscribe(console_reporter)

    bce_val = BinaryCrossentropy()
    bce_val.subscribe(console_reporter)

    rocauc_val = RocAuc()
    rocauc_val.subscribe(console_reporter)

    # Trainer, Validator, Runner
    TRAINER = Trainer(
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_metrics=[],
        epoch_metrics=[bce_train, rocauc_train],
    )

    VAL = Validator(
        loader=val_loader,
        epoch_metrics=[bce_val, rocauc_val]
    )

    # RUNNER = Runner(
    #     model=model,
    #     trainer=TRAINER,
    #     validator=VAL
    # )

    # TRAINER = Trainer(
    #     loader=train_loader,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     batch_metrics=bce_train_batch,
    #     epoch_metrics=[bce_train, rocauc_train],
    # )

    # VAL = Validator(
    #     loader=val_loader,
    #     batch_metrics=[],
    #     epoch_metrics=rocauc_val
    # )

    RUNNER = Runner(
        model=model,
        trainer=TRAINER,
        validator=VAL
    )

    RUNNER.run(1)


# Export
get_submission(
    model=RUNNER.model,
    loader=test_loader,
    test_ids=test_ids,
    # device="cuda",
    device="cpu",
    filename="sub-20211122.csv")


del(TRAINER)
importlib.reload(tinydl.runner_mediator)


#############################
# metric2 and reporter2