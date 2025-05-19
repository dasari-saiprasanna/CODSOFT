# iris_classifier.py

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map(dict(enumerate(iris.target_names)))

# 2. Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# 3. Visualize the data using a pairplot
sns.pairplot(df, hue="species")
plt.suptitle("Iris Feature Pairplot", y=1.02)
plt.show()

# 4. Split the dataset into training and testing sets
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# 7. Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
predicted_species = model.predict(sample)
print("\nPredicted species for sample [5.1, 3.5, 1.4, 0.2]:", predicted_species[0])
