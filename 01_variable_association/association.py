import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Create the dataset
# -------------------------------

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam_Score": [35, 40, 45, 50, 55, 60, 68, 72, 80, 88]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)


# -------------------------------
# 2. Calculate correlation
# -------------------------------

correlation = df["Hours_Studied"].corr(df["Exam_Score"])

print("\nCorrelation Coefficient:", round(correlation, 2))


# -------------------------------
# 3. Interpret the result
# -------------------------------

if correlation > 0:
    print("Result: Positive association between Hours Studied and Exam Score.")

elif correlation < 0:
    print("Result: Negative association between Hours Studied and Exam Score.")

else:
    print("Result: No linear association between the variables.")


# -------------------------------
# 4. Display correlation matrix
# -------------------------------

print("\nCorrelation Matrix:")
print(df.corr())


# -------------------------------
# 5. Create scatter plot
# -------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Hours_Studied",
    y="Exam_Score",
    data=df,
    s=100
)

plt.title("Association Between Hours Studied and Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.grid(True)
plt.show()