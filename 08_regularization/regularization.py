import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam_Score": [42, 48, 53, 58, 65, 68, 74, 80, 85, 90]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create polynomial features
poly = PolynomialFeatures(degree=5)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Ordinary Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

linear_pred = linear_model.predict(X_test_poly)

# Ridge Regression with L2 regularization
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_poly, y_train)

ridge_pred = ridge_model.predict(X_test_poly)

# Evaluate Linear Regression
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\nLinear Regression:")
print("Mean Squared Error:", linear_mse)
print("R2 Score:", linear_r2)

print("\nRidge Regression:")
print("Mean Squared Error:", ridge_mse)
print("R2 Score:", ridge_r2)

print("\nRegularization helps control model complexity and reduce overfitting.")