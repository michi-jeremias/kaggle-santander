"""Kaggle competition: Santander customer transaction prediction"""

# Imports
import torch.nn as nn
import torch.optim as optim
from tinydl.hyperparameter import Hyperparameter
from tinydl.metric import BinaryCrossentropy, RocAuc
from tinydl.modelinit import init_xavier
from tinydl.reporter import ConsoleReporter, TensorboardHparamReporter, TensorboardScalarReporter
from tinydl.runner import Runner, Trainer, Validator
from torch.utils.data import DataLoader

from santander.auxiliary import get_data, get_submission
from santander.model import NN2


# Dataset
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/aug_train.csv",
    test="files/aug_test.csv",
    submission=False)  # True sets the ratio of train/test to 0.999, False = 0.8

# Loss function
loss_fn = nn.BCELoss()


# Hyperparameters
# Best: 256, 2e-5
hparam = {
    "batchsize": [128, 256, 512, 1024],
    "lr": [2e-3, 2e-4]
}


hyperparameter = Hyperparameter(hparam)

#################

for experiment in hyperparameter.get_experiments():
    print(f"NEW RUN: {experiment}")

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

    console_reporter = ConsoleReporter()
    console_reporter.add_metrics([RocAuc()])

    tbscalar_reporter = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter.add_metrics([BinaryCrossentropy(), RocAuc()])

    console_reporter_val = ConsoleReporter()
    console_reporter_val.add_metrics(RocAuc())

    tbscalar_reporter_val = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter_val.add_metrics([BinaryCrossentropy(), RocAuc()])

    tbhparam_reporter = TensorboardHparamReporter(hparam=experiment)
    tbhparam_reporter.add_metrics([BinaryCrossentropy(), RocAuc()])

    TRAINER = Trainer(
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_reporters=tbscalar_reporter,
        epoch_reporters=console_reporter
    )

    VALIDATOR = Validator(
        loader=val_loader,
        batch_reporters=tbscalar_reporter_val,
    )

    RUNNER = Runner(
        model=model,
        trainer=TRAINER,
        validator=VALIDATOR,
        run_reporters=tbhparam_reporter
    )

    RUNNER.run(3)


# Export
get_submission(
    model=RUNNER.model,
    loader=test_loader,
    test_ids=test_ids,
    # device="cuda",
    device="cpu",
    filename="sub-20211122.csv")
