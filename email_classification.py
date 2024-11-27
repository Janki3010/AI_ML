import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\janki.p\\PycharmProjects\\AI-ML\\spam.csv', usecols=[0, 1])
print(df.head())

df.columns = ['Label', 'Message']

df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print()

# Convert text data into numerical features using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Create and train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

new_email = input("Enter Email:")
new_email = [new_email]
new_email_test = tfidf.transform(new_email)
new_email_pred = model.predict(new_email_test)

print(f"Prediction for the {new_email[0]} is", 'Spam' if new_email_pred[0] == 1 else 'Ham')

# Visualizing the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.xticks(np.arange(2), ['ham', 'spam'])
# plt.yticks(np.arange(2), ['ham', 'spam'])
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()