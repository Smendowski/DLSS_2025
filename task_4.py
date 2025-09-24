import strlearn as sl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score as bac
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from stml import STML
import numpy as np

# Generate the data stream
n_chunks = 30
stream = sl.streams.StreamGenerator(n_chunks=n_chunks, n_classes=2, n_features=10, n_informative=10, n_redundant=0,
                                    n_repeated=0, random_state=42, n_drifts=1)

# Define CLF
clf = GaussianNB()

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

# Test-Than-Train
scores = []
for i_chunk in range(n_chunks):
    X, y = stream.get_chunk()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_stml = STML(X_train, verbose=False, size=(50, 50))
    X_train_stml = torch.from_numpy(np.moveaxis(X_train_stml, 3, 1)).float()
    y_train_stml = torch.from_numpy(y_train).float()

    stml_dataset = TensorDataset(X_train_stml, y_train_stml)
    data_loader = DataLoader(stml_dataset, batch_size=batch_size, shuffle=True)

    if i_chunk == 0:
        print(f"CHUNK: {i_chunk} - training...")
        # TRAIN
        for epoch in range(1):
            model.train()
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch
                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.unsqueeze(1).to(device))
                loss.backward()
                optimizer.step()

    else:
        print(f"CHUNK: {i_chunk} - prediction...")
        # Predict
        model.eval()

        X_test_stml = STML(X_test, verbose=False, size=(50, 50))
        X_test_stml = torch.from_numpy(np.moveaxis(X_test_stml, 3, 1)).float()
        logits = model(X_test_stml.to(device))
        preds = (logits.cpu().detach().numpy() > 0).astype(int)

        score = bac(y_test, preds)
        scores.append(score)

        print(f"CHUNK: {i_chunk} - re-training...")
        # Re-TRAIN
        for epoch in range(1):
            model.train()
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch
                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.unsqueeze(1).to(device))
                loss.backward()
                optimizer.step()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(range(1, n_chunks), scores, label='GaussianNB')
ax.set_xlabel('Chunk')
ax.set_ylabel('Balanced Accuracy')
ax.set_ylim(0, 1)
ax.grid(ls=":", c=(.7, .7, .7))
ax.legend()

plt.tight_layout()
plt.savefig("ex4.png")
