from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from stml import STML

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bac
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Generate data using make_classification()
X, y = make_classification()

# Split the generated data into train and test sets using train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the ResNet-18 model
batch_size = 8
num_epochs = 20
weights = None
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
device = torch.device("mps")
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

X_train_stml = STML(X_train, verbose=False, size=(50, 50))
X_train_stml = torch.from_numpy(np.moveaxis(X_train_stml, 3, 1)).float()
y_train_stml = torch.from_numpy(y_train).float()

stml_dataset = TensorDataset(X_train_stml, y_train_stml)
data_loader = DataLoader(stml_dataset, batch_size=batch_size, shuffle=True)

# TRAIN
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(data_loader, 0):
        inputs, labels = batch
        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()

    # TEST
    model.eval()

    X_test_stml = STML(X_test, verbose=False, size=(50, 50))
    X_test_stml = torch.from_numpy(np.moveaxis(X_test_stml, 3, 1)).float()
    logits = model(X_test_stml.to(device))
    preds = (logits.cpu().detach().numpy() > 0).astype(int)

    score = bac(y_test, preds)
    print(score)
