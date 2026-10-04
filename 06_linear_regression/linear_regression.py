import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam_Score": [42, 48, 53, 58, 65, 68, 74, 80, 85, 90]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Features and target
X = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test values
y_pred = model.predict(X_test)

print("\nActual Scores:")
print(y_test.values)

print("\nPredicted Scores:")
print(y_pred)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict score for a new student
new_student = [[7]]
prediction = model.predict(new_student)

print("\nPrediction for New Student:")
print("Hours Studied = 7")
print("Predicted Exam Score:", prediction[0])