import strlearn as sl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score as bac

# Generate the data stream
n_chunks = 30
stream = sl.streams.StreamGenerator(n_chunks=n_chunks, n_classes=2, n_features=10, n_informative=10, n_redundant=0,
                                    n_repeated=0, random_state=42, n_drifts=1)

# Define CLF
clf = GaussianNB()

# Test-Than-Train
scores = []
for i in range(n_chunks):
    X, y = stream.get_chunk()
    print("*", X.shape, y.shape)
    # raise Exception(X.shape, y.shape) # ((200, 10), (200,))
    # clf.partial_fit(X, y.reshape(-1, 1), classes=[0, 1])
    # For the first data chunk we fit the model, but for every next data chunk
    # we first make a prediction and then we do a partial fit.

    if i == 0:
        clf.fit(X, y)
    else:
        # scores.append(clf.predict(X, y))
        y_pred = clf.predict(X)
        scores.append(bac(y_true=y, y_pred=y_pred))
        clf.partial_fit(X, y, classes=[0, 1])

print(scores)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(range(1, n_chunks), scores, label='GaussianNB')
ax.set_xlabel('Chunk')
ax.set_ylabel('Balanced Accuracy')
ax.set_ylim(0, 1)
ax.grid(ls=":", c=(.7, .7, .7))
ax.legend()

plt.tight_layout()
plt.savefig("ex3.png")
