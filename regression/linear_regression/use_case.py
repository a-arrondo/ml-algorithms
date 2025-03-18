
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
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

inp = np.matrix(np.linspace(np.min(X), np.max(X), num = salaries.shape[0])).T
output = lm.predict(inp)

sns.scatterplot(salaries, x="Experience Years", y="Salary")
plt.plot(inp, output.T, color = "red")

plt.title("Linear regression")
plt.xlabel("Experience Years")
plt.ylabel("Salary")

plt.show()

