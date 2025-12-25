import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data  = pd.read_csv("C:\\Users\\astik\\Downloads\\loan.csv")

print(data.head())
print(data.shape)
print(data.columns)
print(data.info())

fill_ffill = ['Gender','Married','Dependents','Self_Employed','Credit_History']
for col in fill_ffill:
    data[col] = data[col].ffill()

fill_mean = ['LoanAmount','Loan_Amount_Term']
for col in fill_mean:
    data[col] = data[col].fillna(data[col].mean())

data = data.drop("Loan_ID", axis = 1)

le = LabelEncoder()
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

for col in cat_cols:
    data[col] = le.fit_transform(data[col])

scaler = StandardScaler()
num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
data[num_cols] = scaler.fit_transform(data[num_cols])

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,6))
data['Loan_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Loan Approval vs Rejection Distribution")
plt.ylabel("")
plt.show()

from sklearn.linear_model import LogisticRegression
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
y_proba_lr = lr.predict_proba(x_test)[:,1]


print("Logistic Regression Accuracy:",accuracy_score(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))
print("ROC AUC:",roc_auc_score(y_test,y_proba_lr))

fpr,tpr,_ = roc_curve(y_test,y_proba_lr)
plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label='LR ROC',linewidth=3)
plt.plot([0,1],[0,1],'k--')
plt.title('Logistic Regression ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# SVM
from sklearn.svm import SVC

svm_model = SVC(C=10, probability=True)
svm_model.fit(x_train,y_train)

y_pred_svm = svm_model.predict(x_test)
y_proba_svm = svm_model.predict_proba(x_test)[:,1]

print(" SVM Accuracy:",accuracy_score(y_test,y_pred_svm))
print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))
print("ROC AUC:",roc_auc_score(y_test,y_proba_svm))

fpr,tpr,_ = roc_curve(y_test,y_proba_svm)
plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label='SVM ROC',linewidth=3, color='orange')
plt.plot([0,1],[0,1],'k--')
plt.title('SVM ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()


# KNN

from sklearn.neighbors import KNeighborsClassifier

# Training KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


y_pred_knn = knn.predict(x_test)

y_proba_knn = knn.predict_proba(x_test)[:, 1]


print(" KNN Model Accuracy =", accuracy_score(y_test, y_pred_knn))
print("="*40)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("="*40)
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("="*40)
print("ROC AUC Score =", roc_auc_score(y_test, y_proba_knn))

#  ROC Curve
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
plt.figure(figsize=(6,5))
plt.plot(fpr_knn, tpr_knn, label='KNN ROC', linewidth=3, color='purple')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend()
plt.grid()
plt.show()

# random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

ran = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=5,
    max_features=2,
    random_state=42
)
ran.fit(x_train, y_train)

y_pred_rf = ran.predict(x_test)
y_proba_rf = ran.predict_proba(x_test)[:, 1]


print(" Random Forest Accuracy =", accuracy_score(y_test, y_pred_rf))
print("="*50)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("="*50)
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("="*50)
print("ROC AUC Score =", roc_auc_score(y_test, y_proba_rf))


fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(6,5))
plt.plot(fpr_rf, tpr_rf, label='Random Forest ROC', linewidth=3, color='green')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.grid()
plt.show()

# bar graph
models_name = [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Random Forest"
]

accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_svm),
    accuracy_score(y_test, y_pred_rf),
]

plt.figure(figsize=(12,6))
bars = plt.bar(models_name, accuracies, width=0.55,
               color=['blue','purple','brown','orange'])

plt.ylim(0.6,1)
plt.xlabel("Machine Learning Models", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.title("Loan Prediction - Model Performance Comparison", fontsize=14)
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.05, yval + 0.01, f"{yval:.2f}", fontsize=10)

plt.show()

# all in one 

plt.figure(figsize=(8, 6))

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)
plt.plot(fpr_lr, tpr_lr, linewidth=2,
         label=f"Logistic Regression (AUC = {auc_lr:.2f})")

# KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
auc_knn = roc_auc_score(y_test, y_proba_knn)
plt.plot(fpr_knn, tpr_knn, linewidth=2,
         label=f"KNN (AUC = {auc_knn:.2f})")

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
auc_svm = roc_auc_score(y_test, y_proba_svm)
plt.plot(fpr_svm, tpr_svm, linewidth=2,
         label=f"SVM (AUC = {auc_svm:.2f})")

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, linewidth=2,
         label=f"Random Forest (AUC = {auc_rf:.2f})")

# Random baseline
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of ML Models")
plt.legend(loc="lower right")
plt.grid()
plt.show()
