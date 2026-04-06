import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Drop unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove missing values
df = df.dropna()

# 🔥 CREATE TARGET COLUMN (IMPORTANT)
# If credit amount is high → risky (0), else good (1)
df["Risk"] = df["Credit amount"].apply(lambda x: 0 if x > 5000 else 1)

# Convert text → numbers
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 🔹 Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Prediction
y_pred = model.predict(X_test)

# 🔹 Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score:", roc)

# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)