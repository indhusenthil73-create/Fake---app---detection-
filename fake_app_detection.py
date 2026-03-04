import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "review": [
        "This app is amazing",
        "Worst fake app",
        "Very useful and helpful",
        "Fake app waste of time"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully")
