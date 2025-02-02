import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample my ids datasets read
df1 = pd.read_csv('02-15-2018.csv')
df2 = pd.read_csv('02-20-2018.csv')
columns_to_drop = ['Src IP', 'Src Port', 'Dst IP', 'Flow ID']
df2.drop(columns=columns_to_drop, inplace=True)

# Sampling dataset the large datasets
Strat_df1 = df1.groupby('Label', group_keys=False).apply(lambda x: x.sample(10000))
Strat_df2 = df2.groupby('Label', group_keys=False).apply(lambda x: x.sample(10000))

final_dataset = pd.concat([Strat_df1, Strat_df2])
del Strat_df1, Strat_df2

# Replacing the text features with numerical values
final_dataset.replace(to_replace=['Infilteration', 'Bot', 'DoS attacks-GoldenEye','DDoS attacks-LOIC-HTTP','DoS attacks-Hulk', 'DoS attacks-Slowloris', 'SSH-Bruteforce', 'FTP-BruteForce', 'DDOS attack-HOIC', 'DoS attacks-SlowHTTPTest', 'DDOS attack-LOIC-UDP', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'], value=0, inplace=True)
final_dataset.replace(to_replace=['Benign'], value=1, inplace=True)

# Converting Timestamp to numerical format
final_dataset['Timestamp'] = pd.to_datetime(final_dataset['Timestamp']).astype(np.int64)

# Converting all columns to float
columns = final_dataset.columns
for i in columns:
    final_dataset[i] = final_dataset[i].astype(float)

# Dropping some columns with common zero values
final_dataset.drop(['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], axis=1, inplace=True)

# Data Preprocessing
final_dataset = final_dataset.drop_duplicates(keep="first")
final_dataset['Flow Byts/s'] = final_dataset['Flow Byts/s'].replace([np.inf, -np.inf], np.nan)
final_dataset['Flow Pkts/s'] = final_dataset['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan)
final_dataset = final_dataset.replace([np.inf, -np.inf], np.nan)
final_dataset = final_dataset.replace(np.nan, 0)

# Feature Engineering
P_Y = final_dataset['Label']
P_X = final_dataset.drop(['Label'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(P_X, P_Y, test_size=0.2)

# Applying the permutation Importance Feature selection technique
PX_train = np.asarray(X_train)
PX_test = np.asarray(X_test)
PY_train = np.asarray(Y_train)
PY_test = np.asarray(Y_test)

sel = SelectFromModel(PermutationImportance(RandomForestClassifier(), cv=5)).fit(PX_train, PY_train)
X_train2 = sel.transform(PX_train)
X_test2 = sel.transform(PX_test)

# Display some sample data
print("Sample data:")
print(final_dataset.head())

print("Selected important features:")
print(P_X.columns[sel.get_support()])

model = RandomForestClassifier()
model.fit(X_train2, Y_train)

print("RandomForestClassifier trained.")

columns = P_X.columns
joblib.dump(model, 'perm_imp')  # Dumping the trained model

coefficients = model.feature_importances_
absCoefficients = abs(coefficients)
Perm_imp = pd.concat((pd.DataFrame(columns, columns=['Variable']), pd.DataFrame(absCoefficients, columns=['absCoefficient'])), axis=1).sort_values(by='absCoefficient', ascending=False)
least_features = Perm_imp.iloc[50:, 0]  # Identify the least importance features in the dataset

print("Least important features:")
print(least_features)

data = least_features.tolist()
for i in data:
    final_dataset.drop(labels=[i], axis=1, inplace=True)

print("Final dataset shape after dropping least important features:")
print(final_dataset.shape)

# Applying the RandomForestClassifier Model
y = final_dataset['Label']
X = final_dataset.drop(['Label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier()

print("Training RandomForestClassifier...")
RF_clf.fit(X_train, y_train)
print("RandomForestClassifier trained.")

joblib.dump(RF_clf, 'RandomForestClassifier')

# Display some sample data before training RandomForestClassifier
print("Sample data before training RandomForestClassifier:")
print(X_train.head())

# Display some sample data after training RandomForestClassifier
print("Sample data after training RandomForestClassifier:")
print(X_train.head())

# Calculate accuracy on training set
train_accuracy = RF_clf.score(X_train, y_train)
print("Accuracy on training set:", train_accuracy)

# Calculate accuracy on test set
test_accuracy = RF_clf.score(X_test, y_test)
print("Accuracy on test set:", test_accuracy)

print("Random Forest Classifier Model training completed.")

# Predictions and Evaluation
y_pred = RF_clf.predict(X_test)

print("\nRandom Forest Classifier Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", classification_report(y_test, y_pred))

# Confusion Matrix
rf_cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 2)
sns.heatmap(rf_cnf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
