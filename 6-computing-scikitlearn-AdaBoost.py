import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.default_rng(42)
X = np.sort(5 * rng.random(80))[:, np.newaxis]
y = np.sin(X).ravel() + rng.normal(0, 0.1, X.shape[0])
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300,random_state=42)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

colors = sns.color_palette("colorblind")
plt.figure(figsize=(10,6))
plt.scatter(X, y, color=colors[0], label="training samples")
plt.plot(X, y_1, color=colors[1], label="Decision Tree", linewidth=2)
plt.plot(X, y_2,color=colors[2], label="AdaBoost", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree and AdaBoost Regression")

plt.legend()
plt.show()


