
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

boston = datasets.load_boston()

X = boston.data[:,np.newaxis,5]
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.scatter(X_test, y_test, color='black',label="Actual Data")

plt.plot(X_test, y_pred,color='blue',linewidth=3, label='Linear Regression Fit')

plt.title('Linear Regression Predicts House Prices')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Housing Prices')
plt.legend()
plt.show()

print(f"均方误差（Mean Squared Error): {mse:2f}")
print(f"决定系数（R^2):{r2:.2f}")
