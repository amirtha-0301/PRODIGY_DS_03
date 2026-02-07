import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("bank.csv")

# Display basic info
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Encode categorical columns
encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = encoder.fit_transform(df[column])

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization: Target variable distribution
plt.figure(figsize=(6, 4))
df['y'].value_counts().plot(kind='bar')
plt.title("Target Variable Distribution (Subscribed / Not Subscribed)")
plt.xlabel("Outcome")
plt.ylabel("Number of Customers")
plt.show()
