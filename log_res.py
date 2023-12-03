import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def normalize_feat(xTrain, xTest):
    scaler = StandardScaler()
    xTrain_norm = scaler.fit_transform(xTrain)
    xTest_norm = scaler.transform(xTest)
    return xTrain_norm, xTest_norm

def unreg_log(xTrain, yTrain, xTest, yTest):
    model = LogisticRegression(penalty='none', max_iter=10000)
    model.fit(xTrain, yTrain)
    yProbs = model.predict_proba(xTest)[:, 1]
    fpr, tpr, _ = roc_curve(yTest, yProbs)
    auc_score = roc_auc_score(yTest, yProbs)
    return fpr, tpr, auc_score, model

# Load the datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')['isFraud']
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['isFraud']

# Normalize the features
X_train_norm, X_test_norm = normalize_feat(X_train, X_test)

# Train the logistic regression model and get ROC metrics
fpr, tpr, auc_score, model = unreg_log(X_train_norm, y_train, X_test_norm, y_test)

# Predictions and Evaluation
predictions = model.predict(X_test_norm)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# AUPRC and Precision-Recall Curve
probabilities = model.predict_proba(X_test_norm)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, probabilities)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.2f}")

plt.figure()
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'2-class Precision-Recall curve: AUPRC={auprc:.2f}')
plt.show()