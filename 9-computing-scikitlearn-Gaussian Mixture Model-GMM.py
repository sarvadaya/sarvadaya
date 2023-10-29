import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

n_components = 3
gmm = GaussianMixture(n_components=n_components)

gmm.fit(X)

labels = gmm.predict(X)

probs = gmm.predict_proba(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.title('GMM Clustering')
plt.xlabel('First Prindipal Component')
plt.ylabel('Second Principal Component')

plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=probs, cmap='viridis', marker='x')
plt.title('GMM Clustering (Probabilities)')
plt.xlabel('First Prindipal Component')
plt.ylabel('Second Principal Component')

plt.show()
