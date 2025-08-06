# Bug-Prone Module Detector using Logistic Regression

# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load the dataset (assumes repo_metrics.csv is in the same folder)
df = pd.read_csv("repo_metrics.csv")

# 3. Show the first few rows (optional sanity check)
print("Sample of dataset:\n", df.head())

# 4. Define input features and target label
X = df[['loc', 'complexity', 'churn']]   # Features
y = df['buggy']                          # Target: 1 if buggy, 0 if clean

# 5. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 10. Predict bug status of a new file (optional demo)
sample_file = [[150, 4, 5]]  # loc=150, complexity=4, churn=5
bug_prob = model.predict_proba(sample_file)[0][1]
print(f"\nPredicted Bug Probability for sample file: {bug_prob:.2f}")
