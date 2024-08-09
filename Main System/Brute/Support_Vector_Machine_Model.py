import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and sample datasets
df1 = pd.read_csv('02-14-2018.csv').sample(n=10000, random_state=42)
df2 = pd.read_csv('02-15-2018.csv').sample(n=10000, random_state=42)

final_dataset = pd.concat([df1, df2])

# Map labels to numerical values
final_dataset['Label'] = final_dataset['Label'].map({'Benign': 1, 'Infilteration': 0, 'Bot': 0,
                                                     'DoS attacks-GoldenEye': 0, 'DoS attacks-Hulk': 0,
                                                     'DoS attacks-Slowloris': 0, 'SSH-Bruteforce': 0,
                                                     'FTP-BruteForce': 0, 'DDOS attack-HOIC': 0,
                                                     'DoS attacks-SlowHTTPTest': 0, 'DDOS attack-LOIC-UDP': 0,
                                                     'Brute Force -Web': 0, 'Brute Force -XSS': 0,
                                                     'SQL Injection': 0})

# Convert Timestamp to numerical format
final_dataset['Timestamp'] = pd.to_datetime(final_dataset['Timestamp']).astype(np.int64)

# Convert all columns to float
final_dataset = final_dataset.astype(float)

# Drop unnecessary columns
final_dataset.drop(['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
                    'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], axis=1, inplace=True)

# Drop duplicate rows
final_dataset = final_dataset.drop_duplicates(keep="first")

# Handle infinite and NaN values
final_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
final_dataset.fillna(0, inplace=True)

# Define features and target
X = final_dataset.drop('Label', axis=1)
y = final_dataset['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'Attack_Detection_Model.pkl')

# Evaluate the model
train_accuracy = svm_model.score(X_train, y_train)
test_accuracy = svm_model.score(X_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Predictions and Evaluation
y_pred = svm_model.predict(X_test)

print("\nSupport Vector Machine Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", classification_report(y_test, y_pred))

# Confusion Matrix
svm_cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(svm_cnf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Support Vector Machine Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
