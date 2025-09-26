import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch

X = np.load("X.npy")
y = np.load("y.npy")
X = torch.from_numpy(np.moveaxis(X, 3, 1)).float()
y = torch.from_numpy(y).float()

"""
TransRate
"""


def coding_rate(Z, eps=1 * 10 ^ -4):
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))

    return 0.5 * rate


def transrate(Z, y, eps=1 * 10 ^ -4):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.0
    K = int(y.max() + 1)
    for i in range(K):
        RZY += coding_rate(Z[(y == i).flatten()], eps)
    return RZ - RZY / K


"""
"""

# MODEL
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
device = torch.device("mps")
model = model.to(device)

print(model)

# TEST
model.eval()

# TransRate
print(get_graph_node_names(model))

return_nodes = {
    'flatten': 'flatten',
    'layer4.0.downsample.1': 'layer4.0.downsample.1',
    'layer3.1.bn2': 'layer3.1.bn2',
    'layer3.0.downsample.1': 'layer3.0.downsample.1',
}

extractor = create_feature_extractor(model, return_nodes=return_nodes)
X_extracted = extractor(X.to(device))

scores = []
for id, key in enumerate(return_nodes.keys()):
    print(key)
    if key == 'flatten':
        score = transrate(X_extracted[key].cpu().detach().numpy(), y)
    else:
        score = transrate(X_extracted[key].cpu().detach().numpy().reshape(X.shape[0], -1), y)

    print(score)
    scores.append(score)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(scores)
ax.set_xlabel("Layer")
ax.set_ylabel("TransRate")
ax.grid(ls=":", c=(.7, .7, .7))
ax.set_xticks(range(len(return_nodes.keys())))
plt.tight_layout()
plt.savefig("layer_selection")

# It means how many layers we should use???
# we can decide from which layers we can extract embeddings
