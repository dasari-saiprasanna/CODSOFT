import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load CSV using local file path
df = pd.read_csv('Movies.csv', encoding='latin1')

# Clean column names
df.columns = df.columns.str.strip()

# Extract numeric duration (e.g., from "120 min" to 120)
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Extract year from strings like "(2020)"
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop rows with missing essential values
required_columns = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Year', 'Rating']
df = df.dropna(subset=required_columns)

# Combine all actors into one string for simplicity
df['Actors'] = df['Actor 1'] + ', ' + df['Actor 2'] + ', ' + df['Actor 3']

# Encode categorical variables
label_encoders = {}
for col in ['Genre', 'Director', 'Actors']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df[['Genre', 'Director', 'Actors', 'Duration', 'Year']]
y = df['Rating']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model to file
joblib.dump(model, 'movie_rating_model.pkl')
print("\n✅ Model saved as 'movie_rating_model.pkl'.")

# Plot actual vs predicted ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid(True)
plt.tight_layout()
plt.show()
