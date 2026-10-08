import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample multimodal dataset
data = {
    "text": [
        "Government announces new education policy",
        "Scientists discover water on a distant planet",
        "Celebrity claims that the moon is made of cheese",
        "Breaking news: aliens have landed in India",
        "New technology improves smartphone battery life",
        "Drinking a special potion makes humans live forever",
        "Researchers develop a new method to clean water",
        "A viral post claims humans can breathe underwater"
    ],
    
    "image_present": [
        1, 1, 1, 1, 1, 1, 1, 1
    ],
    
    "label": [
        "Real", "Real", "Fake", "Fake",
        "Real", "Fake", "Real", "Fake"
    ]
}

df = pd.DataFrame(data)

print("Multimodal Fake News Dataset:")
print(df)

# Convert text into numerical features
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df["text"])

# Add image information
image_features = df[["image_present"]].values

# Combine text and image features
from scipy.sparse import hstack

features = hstack([text_features, image_features])

# Target variable
target = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)

# Train classification model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

print("\nActual Results:")
print(y_test.values)

print("\nPredicted Results:")
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Test a new news article
new_news = ["Scientists announce a new breakthrough in renewable energy"]

new_text_features = vectorizer.transform(new_news)

# Assume image is present
new_image_features = [[1]]

new_features = hstack([
    new_text_features,
    new_image_features
])

prediction = model.predict(new_features)

print("\nPrediction for New News:")
print("News:", new_news[0])
print("Result:", prediction[0])