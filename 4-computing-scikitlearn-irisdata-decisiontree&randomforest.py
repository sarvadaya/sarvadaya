import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2]]):
    X = iris.data[:, pair]
    y = iris.target

    clf_tree = DecisionTreeClassifier().fit(X, y)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X,y)

    ax = sub[pairidx // 2, pairidx % 2]
    DecisionBoundaryDisplay.from_estimator(
        clf_tree,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )
    ax.set_title("Decision Tree")

    DecisionBoundaryDisplay.from_estimator(
        clf_rf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )
    ax.set_title("Random Forest")

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision Surface of Decision Trees and Random Forest on Pairs of Features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()
