
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
        explained_variance_score, r2_score

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
y_pred = np.asarray(lm.predict(X_test))

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = False)
exp_var = explained_variance_score(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("\n# Prediction statistics")
print(f"\tMAE: {mae}")
print(f"\tMSE: {mse}")
print(f"\tRMSE: {rmse}")
print(f"\tExplained variance: {exp_var}")
print(f"\tR2 score: {r_squared}")


# DATA VISUALIZATION
lm = LinearRegression()
lm = lm.fit(X, y)

reg_inp = np.array(np.linspace(np.min(X), np.max(X),
    num = df.shape[0]))
reg_out = lm.predict(reg_inp)

sns.scatterplot(df, x=YEAR_COL, y=SALARY_COL, alpha=0.8, s = 50)
plt.plot(reg_inp, reg_out.T, color = "red", alpha=0.8)

equation = f"Salary = {lm._coefs[1]:.3f} * Experience + {lm._coefs[0]:.3f}"
plt.text(1, 120000, equation, fontsize=14, color="red")

plt.title("Salary vs. Years of Experience:\n"
        "(Univariate Linear Regression)",
        fontweight="bold", fontsize=15)

plt.xlabel(YEAR_COL, fontsize=14)
plt.ylabel(SALARY_COL, fontsize=14)

plt.show()

