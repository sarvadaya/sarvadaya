import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split

data_list = [
    datasets.load_iris(return_X_y=True),
    datasets.make_classification(n_samples=800, random_state=0),
    datasets.make_hastie_10_2(n_samples=2000, random_state=0),
]
names = ["Iris Data", "Classification Data", "Hastie  Data"]

n_gb = []
score_gb = []
time_gb = []
n_gbes = []
score_gbes = []
time_gbes = []

n_estimators = 200

for X, y in data_list:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    gbes = ensemble.GradientBoostingClassifier(
        n_estimators=n_estimators,
        validation_fraction=0.2,
        n_iter_no_change=5,
        tol=0.01,
        random_state=0,
    )
    gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)

    start = time.time()
    gb.fit(X_train, y_train)
    time_gb.append(time.time() - start)

    start = time.time()
    gbes.fit(X_train, y_train)
    time_gbes.append(time.time() - start)

    score_gb.append(gb.score(X_test, y_test))
    score_gbes.append(gbes.score(X_test, y_test))

    n_gb.append(gb.n_estimators_)
    n_gbes.append(gbes.n_estimators_)

bar_width =0.2
n = len(data_list)
index = np.arange(0,n *bar_width, bar_width) * 2.5
index = index[0:n]

plt.figure(figsize=(10, 6))
plt.bar(index, score_gb, bar_width, label ='Gradient Boosting', color='b', alpha=0.7)
plt.bar(index + bar_width, score_gbes, bar_width, label='Gradient Boosting with Early Stopping', color='g', alpha=0.7)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Accuracy of Gradient Boosing vs. Early Stopping Gradient Boosting')
plt.xticks(index + bar_width, names)
plt.legend(loc='best')
plt.tight_layout()
plt.show()