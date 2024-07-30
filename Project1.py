import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data_path = 'creditcard.csv'  
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(df.head())

X = df.drop(columns=['Class'])
y = df['Class']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_train)

y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
