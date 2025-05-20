import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
print("Original dataset shape:", df.shape)
print("\nOriginal class distribution:\n", df['Class'].value_counts())

# ⚠️ Optional: use 10% of data to reduce processing time
df = df.sample(frac=0.1, random_state=42)
print("\nReduced dataset shape:", df.shape)

# Step 2: Preprocess the data
print("\nPreprocessing data...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)

# Step 3: Handle class imbalance with SMOTE
print("\nHandling class imbalance with SMOTE...")
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Resampled class distribution:\n", pd.Series(y_resampled).value_counts())

# Step 4: Split the data
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 5: Train models
print("\nTraining models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest (reduced number of trees)
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 6: Evaluate models
print("\n=== Logistic Regression Evaluation ===")
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest Evaluation ===")
print(classification_report(y_test, y_pred_rf))

# Step 7: Confusion Matrix for Random Forest
print("\nGenerating confusion matrix for Random Forest...")
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save and show the plot
image_path = "confusion_matrix.png"
plt.savefig(image_path)
plt.show(block=True)

# Open the saved image file in default viewer
try:
    webbrowser.open(image_path)
except Exception as e:
    print(f"Could not open image automatically: {e}")

print("\n✅ Task completed successfully!")
