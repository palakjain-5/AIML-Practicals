import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Result": [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Features and target
X = df[["Hours_Studied", "Attendance"]]
y = df["Result"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

print("\nActual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Predict a new student
new_student = [[7, 82]]
prediction = model.predict(new_student)

print("\nPrediction for new student:")
print("Hours Studied = 7, Attendance = 82")
print("Result:", prediction[0])