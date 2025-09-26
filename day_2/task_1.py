from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch

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

X = np.load("X.npy")
y = np.load("y.npy")
X = torch.from_numpy(np.moveaxis(X, 3, 1)).float()
y = torch.from_numpy(y).float()

# MODEL
weights = ResNet18_Weights.IMAGENET1K_V1
# weights = None
model = resnet18(weights=weights)
device = torch.device("mps")
model = model.to(device)

# TEST
model.eval()

# TransRate
print(get_graph_node_names(model))

return_nodes = {
    'flatten': 'flatten',
}
extractor = create_feature_extractor(model, return_nodes=return_nodes)
X_extracted = extractor(X.to(device))["flatten"].cpu().detach().numpy()

score = transrate(X_extracted, y)
print(score)
