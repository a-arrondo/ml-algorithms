
import numpy as np
import pandas as pd
from linear_regression import LinearRegression

data_url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Salary%20Data.csv"

salaries = pd.read_csv(data_url)

X = salaries.loc[:, "Experience Years"]
print(X.values)
y = salaries.loc[:, "Salary"]
print(y.values)

print(salaries.head(10))

lm = LinearRegression()
lm = lm.fit(X.values, y.values)

print("PREDICTION")
print(lm.predict(np.matrix(10)))

