import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

# 1. Baseline model from notebook
df = pd.read_csv('SVM-based Loan Predictor.csv')
df_clean = df.drop(columns=['Loan_ID']).dropna(subset=['Loan_Status']).copy()

for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
for col in df_clean.select_dtypes(include=['object', 'str']).columns:
    if col != 'Loan_Status':
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
df_clean['Loan_Status'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})

CATEGORICAL_CATEGORIES = {
    'Gender': ['Female', 'Male'],
    'Married': ['No', 'Yes'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes'],
    'Property_Area': ['Rural', 'Semiurban', 'Urban']
}
for col, cats in CATEGORICAL_CATEGORIES.items():
    df_clean[col] = pd.Categorical(df_clean[col], categories=cats)

df_base = pd.get_dummies(df_clean, drop_first=True)
X_base = df_base.drop(columns=['Loan_Status'])
y = df_base['Loan_Status']

rf_base = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

res_base = cross_validate(rf_base, X_base, y, cv=cv, scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'])
print("--- Baseline Model (Current Notebook) ---")
print(f"Accuracy  : {res_base['test_accuracy'].mean():.4f}")
print(f"ROC-AUC   : {res_base['test_roc_auc'].mean():.4f}")
print(f"F1-Score  : {res_base['test_f1'].mean():.4f}")
print(f"Precision : {res_base['test_precision'].mean():.4f}")
print(f"Recall    : {res_base['test_recall'].mean():.4f}")

# 2. Strategy A: Engineer Non-Linear Capacity Ratios (Fixing near-zero linear correlation of raw amounts)
df_exp = df_clean.copy()
df_exp['Loan_to_Income'] = (df_exp['LoanAmount'] * 1000) / (df_exp['TotalIncome'] + 1)
df_exp['EMI'] = (df_exp['LoanAmount'] * 1000) / df_exp['Loan_Amount_Term']
df_exp['EMI_to_Income'] = df_exp['EMI'] / (df_exp['TotalIncome'] + 1)
df_exp['Balance_Income'] = df_exp['TotalIncome'] - df_exp['EMI']
df_exp['TotalIncome_Log'] = np.log1p(df_exp['TotalIncome'])
df_exp['LoanAmount_Log'] = np.log1p(df_exp['LoanAmount'])

# Strategy B: Remove severe multicollinearity (drop raw ApplicantIncome and CoapplicantIncome)
df_exp_no_multi = df_exp.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

X_exp = pd.get_dummies(df_exp_no_multi, drop_first=True).drop(columns=['Loan_Status'])

rf_opt = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
res_exp = cross_validate(rf_opt, X_exp, y, cv=cv, scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'])

print("\n--- Strategy A+B: Ratios + Removing Multicollinearity + Regularization ---")
print(f"Accuracy  : {res_exp['test_accuracy'].mean():.4f}  (Improvement: +{(res_exp['test_accuracy'].mean() - res_base['test_accuracy'].mean())*100:.2f}%)")
print(f"ROC-AUC   : {res_exp['test_roc_auc'].mean():.4f}")
print(f"F1-Score  : {res_exp['test_f1'].mean():.4f}")
print(f"Precision : {res_exp['test_precision'].mean():.4f}")
print(f"Recall    : {res_exp['test_recall'].mean():.4f}")

