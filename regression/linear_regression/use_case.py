
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# DATA COLLECTION
data_url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Salary%20Data.csv"
df = pd.read_csv(data_url)

# DATA EXPLORATION
print(f"# Dimensionality")
print(f"({df.shape[0]}, {df.shape[1]})")
print("\n# First 10 elements:")
print(df.head(10))
print("\n# Statistical summary")
print(df.describe())

YEAR_COL = df.columns[0]
SALARY_COL = df.columns[1]

X = df[YEAR_COL].values
y = df[SALARY_COL].values

# MODEL EVALUATION
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

lm = LinearRegression()
lm = lm.fit(X_train, y_train)
print(X_test)
y_pred = np.asarray(lm.predict(X_test))

rmse = mean_squared_error(y_test, y_pred, squared = False)
exp_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n# Prediction statistics")
print(f"\tRMSE: {rmse}")
print(f"\tExplained variance: {exp_var}")
print(f"\tR2 score: {r2}")
# ?
lm = LinearRegression()
lm = lm.fit(X, y)

# DATA VISUALIZATION
reg_inp = np.array(np.linspace(np.min(X), np.max(X),
    num = df.shape[0]))
reg_out = lm.predict(reg_inp)

sns.scatterplot(df, x=YEAR_COL, y=SALARY_COL)
plt.plot(reg_inp, reg_out.T, color = "red")

plt.title("Linear regression")
plt.xlabel(YEAR_COL)
plt.ylabel(SALARY_COL)

plt.show()

