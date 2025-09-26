from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from stml import STML

# Generate datasets using make_classification()
X, y = make_classification(n_samples=5)

# Use STML to encode generated tabular data into images
X_stml = STML(X, size=(224,224), verbose=True)

print(X_stml.shape[0])

# Plot 5 first samples and their labels – save the result as a .png file
fig, ax = plt.subplots(1, 5, figsize=(15, 5))

for i in range(X_stml.shape[0]):
    ax[i].imshow(X_stml[i], cmap='gray')
    ax[i].set_title("Class: %i" % y[i], fontsize=20)
    ax[i].axis('off')
plt.tight_layout()
plt.savefig("ex1")

