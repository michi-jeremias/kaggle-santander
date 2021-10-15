"""Kaggle competition: Santander customer transaction prediction"""

# Imports
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

from santander.auxiliary import get_data, get_submission
from deeplearning.data.loader import Loader
from deeplearning.model.model import NN2, NN3
from deeplearning.model.init import init_normal, init_xavier
from deeplearning.trainer.trainer import Trainer


# Hyperparameters
BATCH_SIZE = 1024


# Data
train_ds, val_ds, test_ds, test_ids = get_data(
    train="files/aug_train.csv",
    test="files/aug_test.csv",
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
loader = Loader(train_loader, val_loader, test_loader)

# Model, Optimizer
model = NN2(input_size=400, hidden_dim=100)
optimizer = optim.Adam(
    params=model.parameters(),
    lr=2e-3,
    weight_decay=1e-4)
model.apply(init_normal)


# Loss
loss_fn = nn.BCELoss()


# Trainer
trainer = Trainer(
    model=model,
    optimizer=optim.Adam(
        params=model.parameters(),
        lr=2e-3,
        weight_decay=1e-4
    ),
    loader=loader,
    loss_fn=nn.BCELoss(),
    metrics_fn=metrics.roc_auc_score,
    init_fn=init_normal
)
trainer.train(20)


# Export
get_submission(
    model=trainer.model,
    loader=test_loader,
    test_ids=test_ids,
    device="cuda",
    filename="sub-20210927.csv")
