
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
TITLE_SIZE = 15
TITLE_FW = "bold"
LABEL_SIZE = 14
MAIN_COLOR = "royalblue"
SECOND_COLOR = "firebrick"

lm = LinearRegression()
lm = lm.fit(X, y)

reg_inp = np.array(np.linspace(np.min(X), np.max(X),
    num = df.shape[0]))
reg_out = lm.predict(reg_inp)

fig1, ax1 = plt.subplots()
ax1.scatter(df[YEAR_COL], df[SALARY_COL], alpha=0.8, s = 50)
ax1.plot(reg_inp, reg_out.T, color = SECOND_COLOR, alpha=0.8)

equation = f"Salary = {lm.get_coefs()[0]:.3f} * Experience + {lm.get_intercept():.3f}"
ax1.text(1, 120000, equation, fontsize=14, color=SECOND_COLOR)

ax1.set_title("Salary vs. Years of Experience:\n"
        "(Univariate Linear Regression)",
        fontweight="bold", fontsize=TITLE_SIZE)

ax1.set_xlabel(YEAR_COL, fontsize=LABEL_SIZE)
ax1.set_ylabel(SALARY_COL, fontsize=LABEL_SIZE)
print("# Data visualization")

# RIDGE REGRESSION
l2_penaltys = [10**i for i in range(-4, 7)]
data = data = np.zeros((len(l2_penaltys), 3))
for i, l2_penalty in enumerate(l2_penaltys):
    lm = LinearRegression().fit(X, y, l2_penalty=l2_penalty)
    y_pred = lm.predict(X)
    rmse = mean_squared_error(y, y_pred, squared = False)
    data[i] = [l2_penalty, lm.coefs_[0], rmse]

df = pd.DataFrame(data, columns=["l2_penalty", "coef", "r2"])

fig2, ax2 = plt.subplots(1, 1)
ax3 = ax2.twinx()
ax2.plot(df["l2_penalty"], df["coef"], "o-", color=MAIN_COLOR)
ax3.plot(df["l2_penalty"], df["r2"], "o-", color=SECOND_COLOR)
ax2.set_xscale("log")
ax2.set_title("Ridge Regression's influence on coefficients",
        fontweight=TITLE_FW, fontsize=TITLE_SIZE)
ax2.set_xlabel("L2 penalty parameter (log)",
        fontsize=LABEL_SIZE)
ax2.set_ylabel("Experience years coefficient",
        fontsize=LABEL_SIZE, color = MAIN_COLOR)
ax2.tick_params(axis="y", colors=MAIN_COLOR)

ax3.set_ylabel("RMSE measure",
        fontsize=LABEL_SIZE, color = SECOND_COLOR)
ax3.tick_params(axis="y", colors=SECOND_COLOR)
plt.show()
