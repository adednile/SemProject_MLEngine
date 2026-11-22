import json, io, base64, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

print("Starting end-to-end execution of improved notebook pipeline...")

# Load raw
df_raw = pd.read_csv("SVM-based Loan Predictor.csv")
df_clean = df_raw.copy()

# 1. Drop non-predictive Loan_ID
if 'Loan_ID' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Loan_ID'])

# 2. Filter missing target records
df_clean = df_clean.dropna(subset=['Loan_Status'])

# 3. Impute numerical missing values with median
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4. Impute categorical missing values with mode
for col in df_clean.select_dtypes(include=['object', 'str']).columns:
    if col != 'Loan_Status':
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 5. Correlation-Guided Financial Feature Engineering
df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']

# Relative debt leverage & burden ratios
df_clean['Loan_to_Income'] = (df_clean['LoanAmount'] * 1000.0) / (df_clean['TotalIncome'] + 1.0)
df_clean['EMI'] = (df_clean['LoanAmount'] * 1000.0) / df_clean['Loan_Amount_Term']
df_clean['EMI_to_Income'] = df_clean['EMI'] / (df_clean['TotalIncome'] + 1.0)
df_clean['Balance_Income'] = df_clean['TotalIncome'] - df_clean['EMI']

# Skew compression transforms
df_clean['TotalIncome_Log'] = np.log1p(df_clean['TotalIncome'])
df_clean['LoanAmount_Log'] = np.log1p(df_clean['LoanAmount'])

# 6. Feature Pruning: Drop collinear raw ApplicantIncome and CoapplicantIncome
df_clean = df_clean.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# 7. Target Mapping ('Y' -> 1, 'N' -> 0)
if df_clean['Loan_Status'].dtype in ['object', 'str']:
    df_clean['Loan_Status'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})

# 8. Fixed Categorical Domains & One-Hot Encoding
CATEGORICAL_CATEGORIES = {
    'Gender': ['Female', 'Male'],
    'Married': ['No', 'Yes'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes'],
    'Property_Area': ['Rural', 'Semiurban', 'Urban']
}
for col, cats in CATEGORICAL_CATEGORIES.items():
    if col in df_clean.columns:
        df_clean[col] = pd.Categorical(df_clean[col], categories=cats)

df_model = pd.get_dummies(df_clean, drop_first=True)

X = df_model.drop(columns=['Loan_Status'])
y = df_model['Loan_Status']

print(f"Features matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# Base Random Forest Classifier
base_rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=5, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    random_state=42
)

# 5-Fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_scores = cross_validate(base_rf, X, y, cv=cv, scoring=scoring)

print("\n--- 5-Fold Stratified Cross-Validation Results ---")
for metric in scoring:
    m = cv_scores[f'test_{metric}'].mean()
    s = cv_scores[f'test_{metric}'].std()
    print(f"• Mean {metric.capitalize():<12}: {m:.4f} (± {s:.4f})")

# 80/20 Stratified Partition
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_rf.fit(X_train, y_train)

y_pred = base_rf.predict(X_test)
y_prob = base_rf.predict_proba(X_test)[:, 1]

# Confusion Matrix & ROC Plot
fig_holdout, axes_holdout = plt.subplots(1, 2, figsize=(14, 5.5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_holdout[0], cbar=False,
            xticklabels=['Predicted: Rejected (0)', 'Predicted: Approved (1)'],
            yticklabels=['Actual: Rejected (0)', 'Actual: Approved (1)'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes_holdout[0].set_title("Test Confusion Matrix (Improved Model)", weight='bold')

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
axes_holdout[1].plot(fpr, tpr, color='#008080', lw=2.5, label=f'Random Forest (ROC-AUC = {roc_auc:.3f})')
axes_holdout[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
axes_holdout[1].set_title("Holdout Test ROC Curve", weight='bold')
axes_holdout[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes_holdout[1].set_ylabel("True Positive Rate (Recall / Sensitivity)")
axes_holdout[1].legend(loc="lower right", frameon=True)
plt.tight_layout()
holdout_b64 = fig_to_b64(fig_holdout)

# Multi-Scenario Live Borrower Evaluation
test_cases = [
    {
        "Scenario": "1. Prime Member",
        "Description": "Solid income, moderate loan, clean credit history (1.0).",
        "ApplicantIncome": 6000.0,
        "CoapplicantIncome": 2500.0,
        "LoanAmount": 120.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Semiurban"
    },
    {
        "Scenario": "2. High-Income Delinquent",
        "Description": "High income (KES 9,000) but past default history on record (Credit_History=0.0).",
        "ApplicantIncome": 9000.0,
        "CoapplicantIncome": 3000.0,
        "LoanAmount": 130.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 0.0,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban"
    },
    {
        "Scenario": "3. Low-Income with Guarantor",
        "Description": "Low applicant income (KES 2,000) rescued by strong coapplicant (KES 4,500), clean credit.",
        "ApplicantIncome": 2000.0,
        "CoapplicantIncome": 4500.0,
        "LoanAmount": 100.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Gender": "Female",
        "Married": "No",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Semiurban"
    },
    {
        "Scenario": "4. Overleveraged / Low-Income",
        "Description": "Low income (KES 2,200) seeking large loan (KES 200,000) without coapplicant.",
        "ApplicantIncome": 2200.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 200.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "2",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "Property_Area": "Rural"
    }
]

df_scenarios = pd.DataFrame(test_cases)
df_scenarios_clean = df_scenarios.copy()
df_scenarios_clean['TotalIncome'] = df_scenarios_clean['ApplicantIncome'] + df_scenarios_clean['CoapplicantIncome']
df_scenarios_clean['Loan_to_Income'] = (df_scenarios_clean['LoanAmount'] * 1000.0) / (df_scenarios_clean['TotalIncome'] + 1.0)
df_scenarios_clean['EMI'] = (df_scenarios_clean['LoanAmount'] * 1000.0) / df_scenarios_clean['Loan_Amount_Term']
df_scenarios_clean['EMI_to_Income'] = df_scenarios_clean['EMI'] / (df_scenarios_clean['TotalIncome'] + 1.0)
df_scenarios_clean['Balance_Income'] = df_scenarios_clean['TotalIncome'] - df_scenarios_clean['EMI']
df_scenarios_clean['TotalIncome_Log'] = np.log1p(df_scenarios_clean['TotalIncome'])
df_scenarios_clean['LoanAmount_Log'] = np.log1p(df_scenarios_clean['LoanAmount'])

df_scenarios_clean = df_scenarios_clean.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

for col, cats in CATEGORICAL_CATEGORIES.items():
    if col in df_scenarios_clean.columns:
        df_scenarios_clean[col] = pd.Categorical(df_scenarios_clean[col], categories=cats)

X_scenarios = pd.get_dummies(df_scenarios_clean.drop(columns=['Scenario', 'Description']), drop_first=True)

for col in X.columns:
    if col not in X_scenarios.columns:
        X_scenarios[col] = 0
X_scenarios = X_scenarios[X.columns]

scen_preds = base_rf.predict(X_scenarios)
scen_probs = base_rf.predict_proba(X_scenarios)[:, 1]

scenario_results = []
for i, sc in enumerate(test_cases):
    p = scen_probs[i]
    d = scen_preds[i]
    risk_tier = "Low Risk" if p >= 0.70 else "Medium Risk" if p >= 0.40 else "High Risk"
    scenario_results.append({
        "Scenario": sc["Scenario"],
        "Credit History": f"{sc['Credit_History']:.1f}",
        "Total Income (KES)": f"{sc['ApplicantIncome'] + sc['CoapplicantIncome']:,.0f}",
        "Loan (k KES)": f"{sc['LoanAmount']:.0f}",
        "Approval Prob (%)": f"{p * 100:.1f}%",
        "Model Decision": "Approved" if d == 1 else "Rejected",
        "Risk Tier": risk_tier
    })

scen_df = pd.DataFrame(scenario_results)
print("\n--- Multi-Scenario Live Borrower Evaluation ---")
print(scen_df.to_string(index=False))

fig_scen, ax_scen = plt.subplots(figsize=(9, 3.8))
bar_colors = ['#2ca02c' if p >= 0.7 else '#ff7f0e' if p >= 0.4 else '#d62728' for p in scen_probs]
bars = ax_scen.barh([s["Scenario"] for s in test_cases], scen_probs * 100, color=bar_colors, edgecolor='black', height=0.55)
ax_scen.axvline(50, color='gray', linestyle='--', linewidth=1.5, label='50% Decision Threshold')
ax_scen.set_xlim(0, 100)
ax_scen.set_xlabel("Approval Probability (%)", fontsize=11)
ax_scen.set_title("Base Random Forest: Predicted Approval Probability by Scenario", weight='bold', pad=12)
for bar in bars:
    w = bar.get_width()
    ax_scen.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', weight='bold')
ax_scen.invert_yaxis()
ax_scen.legend(loc='lower right', frameon=True)
plt.tight_layout()
scen_b64 = fig_to_b64(fig_scen)

# Comparative Model Benchmarking
candidate_models = {
    "Base Model: Random Forest": base_rf,
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Support Vector Machine (RBF)": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(random_state=42), ensemble=False))
}

comparison_records = []
for name, model in candidate_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    auc_val = roc_auc_score(y_test, probs) if probs is not None else 0.0
    comparison_records.append({
        "Model Name": name,
        "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
        "ROC-AUC": round(auc_val, 4),
        "Precision (%)": round(precision_score(y_test, preds) * 100, 2),
        "Recall (%)": round(recall_score(y_test, preds) * 100, 2),
        "F1-Score (%)": round(f1_score(y_test, preds) * 100, 2)
    })

comparison_df = pd.DataFrame(comparison_records)
print("\n--- Comparative Model Benchmark ---")
print(comparison_df.to_string(index=False))

fig_comp, ax_comp = plt.subplots(figsize=(10, 4.5))
sns.barplot(data=comparison_df, x='Model Name', y='ROC-AUC', hue='Model Name', legend=False, palette='viridis', ax=ax_comp)
ax_comp.set_title("ROC-AUC Comparison: Base Model vs. Candidate Algorithms", weight='bold', pad=12)
ax_comp.set_ylabel("ROC-AUC Score")
ax_comp.set_ylim(0.5, 0.95)
ax_comp.tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['ROC-AUC']):
    ax_comp.text(i, v + 0.01, f"{v:.3f}", ha='center', weight='bold')
plt.tight_layout()
comp_b64 = fig_to_b64(fig_comp)

# Feature Importance
feat_imp = pd.Series(base_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
top10 = feat_imp.head(10)
top10.plot(kind='barh', color='#008080', edgecolor='black', ax=ax_fi)
ax_fi.set_title("Top 10 Most Influential Features in the Base Model (Improved)", weight='bold', pad=12)
ax_fi.set_xlabel("Relative Gini Importance Score")
ax_fi.set_ylabel("Feature")
ax_fi.invert_yaxis()

for i, v in enumerate(top10):
    ax_fi.text(v + 0.005, i, f"{v*100:.1f}%", va='center', weight='bold')

plt.tight_layout()
fi_b64 = fig_to_b64(fig_fi)

# Model serialization
joblib.dump(base_rf, "RandomForest_Base.pkl")
joblib.dump(base_rf, "chama_loan_rf_model.pkl")

# Save computed data to cache
cache = {
    'features': list(X.columns),
    'cv_scores': {k: [float(cv_scores['test_' + k].mean()), float(cv_scores['test_' + k].std())] for k in scoring},
    'report': classification_report(y_test, y_pred, target_names=['Defaulted / Rejected (0)', 'Approved (1)']),
    'scen_df': scen_df.to_string(index=False),
    'comp_df': comparison_df.to_string(index=False),
    'top5_fi': [(col, float(score)) for col, score in feat_imp.head(5).items()],
    'holdout_b64': holdout_b64,
    'scen_b64': scen_b64,
    'comp_b64': comp_b64,
    'fi_b64': fi_b64
}

with open('scratch/pipeline_cache.json', 'w') as f:
    json.dump(cache, f)

print("Pipeline executed and verified successfully! Cache saved.")
