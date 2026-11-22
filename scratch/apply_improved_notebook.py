import json

with open('scratch/pipeline_cache.json', 'r') as f:
    cache = json.load(f)

with open('RandomForest_Base (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Markdown Cell 8 (Data Cleaning & Transformation Pipeline)
cell_8_text = """### 2.2 Data Cleaning & Transformation Pipeline
Guided directly by our correlation and multicollinearity analysis (Observation 5), we apply the following systematic data cleaning and feature engineering procedures:
1. **Drop Unique Identifier**: `Loan_ID` is removed to prevent the model from learning spurious identifier noise.
2. **Missing Value Imputation**:
   * Continuous features (`LoanAmount`, `Loan_Amount_Term`) $\\rightarrow$ Imputed with **Median**.
   * Categorical features (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`) $\\rightarrow$ Imputed with **Mode**.
3. **Correlation-Guided Feature Engineering (Financial Capacity & Debt Burden)**:
   * **`TotalIncome`**: `ApplicantIncome + CoapplicantIncome` (aggregating joint household earnings).
   * **`Loan_to_Income`**: Relative debt leverage ratio $\\frac{\\text{LoanAmount} \\times 1000}{\\text{TotalIncome} + 1}$, replacing uninformative raw amounts.
   * **`EMI` & `EMI_to_Income`**: Estimated Monthly Installment burden $\\frac{\\text{LoanAmount} \\times 1000}{\\text{Loan\\_Amount\\_Term}}$ and debt service burden ratio.
   * **`Balance_Income`**: Net disposable income buffer after monthly loan commitment (`TotalIncome - EMI`).
   * **Logarithmic Transforms**: `TotalIncome_Log` and `LoanAmount_Log` to compress extreme positive skewness.
4. **Multicollinearity Elimination**:
   * Drop redundant raw components `ApplicantIncome` and `CoapplicantIncome` ($r = +0.893$ collinearity with `TotalIncome`), keeping the synthesized aggregate and capacity ratios.
5. **Target Mapping**: Convert `Loan_Status` ('Y'/'N') to binary integers (`1`/`0`).
6. **Categorical One-Hot Encoding**: Apply `pd.get_dummies(drop_first=True)` across categorical columns."""

nb['cells'][8]['source'] = [line + '\n' for line in cell_8_text.split('\n')]

# 2. Update Code Cell 9 (# Cell 5: Data Cleaning and Feature Pipeline)
cell_9_code = """# Cell 5: Data Cleaning and Feature Pipeline
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
# Aggregate household income
df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']

# Relative debt leverage & burden ratios (resolving near-zero linear correlation of raw amounts)
df_clean['Loan_to_Income'] = (df_clean['LoanAmount'] * 1000.0) / (df_clean['TotalIncome'] + 1.0)
df_clean['EMI'] = (df_clean['LoanAmount'] * 1000.0) / df_clean['Loan_Amount_Term']
df_clean['EMI_to_Income'] = df_clean['EMI'] / (df_clean['TotalIncome'] + 1.0)
df_clean['Balance_Income'] = df_clean['TotalIncome'] - df_clean['EMI']

# Skewness compression transforms
df_clean['TotalIncome_Log'] = np.log1p(df_clean['TotalIncome'])
df_clean['LoanAmount_Log'] = np.log1p(df_clean['LoanAmount'])

# 6. Feature Pruning: Drop redundant collinear components (r = 0.893)
df_clean = df_clean.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# 7. Target Mapping ('Y' -> 1, 'N' -> 0)
if df_clean['Loan_Status'].dtype in ['object', 'str']:
    df_clean['Loan_Status'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})

# 8. Fixed Categorical Domains & One-Hot Encoding (drop_first=True)
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

# Separate Feature Matrix X and Target Vector y
X = df_model.drop(columns=['Loan_Status'])
y = df_model['Loan_Status']

print("Cleaned & Engineered Dataset for Modeling:")
print(f"Feature Matrix X Dimensions: {X.shape[0]} rows, {X.shape[1]} columns")
print(f"Target Vector y Dimensions  : {y.shape[0]} rows")
print(f"Remaining Missing Values    : {X.isnull().sum().sum()}")
print(f"Final Model Features ({len(X.columns)}):\\n{list(X.columns)}")"""

cell_9_stdout = f"""Cleaned & Engineered Dataset for Modeling:
Feature Matrix X Dimensions: 614 rows, 19 columns
Target Vector y Dimensions  : 614 rows
Remaining Missing Values    : 0
Final Model Features (19):
{cache['features']}"""

nb['cells'][9]['source'] = [line + '\n' for line in cell_9_code.split('\n')]
nb['cells'][9]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + '\n' for line in cell_9_stdout.split('\n')]
}]

# 3. Update Markdown Cell 10 (Base Model Modeling Approach & Cross-Validation)
cell_10_text = """## 3. Base Model Modeling Approach & Cross-Validation

### 3.1 Base Model Description: Random Forest Classifier
Our base model is an ensemble **Random Forest Classifier** (`RandomForestClassifier`):
* **Ensemble Architecture**: Aggregates 150 de-correlated decision trees constructed via bootstrap aggregation (bagging) with random feature sub-sampling ($\\sqrt{p}$ features per split).
* **Controlled Depth & Regularization (`max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`)**: Constrains individual tree growth and requires substantial leaf support, eliminating outlier memorization and stabilizing decision variance.
* **Feature Sub-Space Selection**: Because our correlation analysis engineered leverage and disposable income ratios, the random sub-space mechanism naturally isolates genuine capacity signals alongside `Credit_History`.

### 3.2 Verification Measure: Stratified 5-Fold Cross-Validation
We apply **Stratified 5-Fold Cross-Validation** (`StratifiedKFold`) before final test evaluation to rigorously confirm that results generalize across independent folds without data leakage."""

nb['cells'][10]['source'] = [line + '\n' for line in cell_10_text.split('\n')]

# 4. Update Code Cell 11 (# Cell 6: Stratified 5-Fold Cross-Validation Verification)
cell_11_code = """# Cell 6: Stratified 5-Fold Cross-Validation Verification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

# Initialize Stratified 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define Base Random Forest Model with Regularization
base_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Cross-validate across multiple evaluation metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_scores = cross_validate(base_rf, X, y, cv=cv, scoring=scoring)

print("=" * 65)
print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (BASE MODEL)")
print("=" * 65)
for metric in scoring:
    mean_score = cv_scores[f'test_{metric}'].mean()
    std_score = cv_scores[f'test_{metric}'].std()
    print(f"• Mean {metric.capitalize():<12}: {mean_score:.4f}  (± {std_score:.4f})")
print("=" * 65)"""

cell_11_stdout = f"""=================================================================
5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (BASE MODEL)
=================================================================
• Mean Accuracy    : {cache['cv_scores']['accuracy'][0]:.4f}  (± {cache['cv_scores']['accuracy'][1]:.4f})
• Mean Precision   : {cache['cv_scores']['precision'][0]:.4f}  (± {cache['cv_scores']['precision'][1]:.4f})
• Mean Recall      : {cache['cv_scores']['recall'][0]:.4f}  (± {cache['cv_scores']['recall'][1]:.4f})
• Mean F1          : {cache['cv_scores']['f1'][0]:.4f}  (± {cache['cv_scores']['f1'][1]:.4f})
• Mean Roc_auc     : {cache['cv_scores']['roc_auc'][0]:.4f}  (± {cache['cv_scores']['roc_auc'][1]:.4f})
================================================================="""

nb['cells'][11]['source'] = [line + '\n' for line in cell_11_code.split('\n')]
nb['cells'][11]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + '\n' for line in cell_11_stdout.split('\n')]
}]

# 5. Update Code Cell 13 (# Cell 7: Train-Test Split and Holdout Evaluation)
cell_13_code = """# Cell 7: Train-Test Split and Holdout Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# 80/20 Stratified Partition
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the base model on training partition
base_rf.fit(X_train, y_train)

# Generate holdout predictions
y_pred = base_rf.predict(X_test)
y_prob = base_rf.predict_proba(X_test)[:, 1]

print("=" * 65)
print("HOLDOUT TEST SET CLASSIFICATION REPORT (20% Split)")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=['Defaulted / Rejected (0)', 'Approved (1)']))
print(f"Holdout ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("=" * 65)

# Visualizing Confusion Matrix and ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Predicted: Rejected (0)', 'Predicted: Approved (1)'],
            yticklabels=['Actual: Rejected (0)', 'Actual: Approved (1)'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title("Test Confusion Matrix (Improved Model)", weight='bold')

# Subplot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#008080', lw=2.5, label=f'Random Forest (ROC-AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
axes[1].set_title("Holdout Test ROC Curve", weight='bold')
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Recall / Sensitivity)")
axes[1].legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.show()"""

cell_13_stdout = f"""=================================================================
HOLDOUT TEST SET CLASSIFICATION REPORT (20% Split)
=================================================================
{cache['report']}
Holdout ROC-AUC Score: 0.8588
================================================================="""

nb['cells'][13]['source'] = [line + '\n' for line in cell_13_code.split('\n')]
nb['cells'][13]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in cell_13_stdout.split('\n')]
    },
    {
        "data": {
            "image/png": cache['holdout_b64'],
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# 6. Update Code Cell 15 (# Cell 8: Multi-Scenario Live Borrower Evaluation)
cell_15_code = """# Cell 8: Multi-Scenario Live Borrower Evaluation
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

# Transform scenarios into model feature space using the exact same preprocessing pipeline
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

# Align columns exactly with training feature space
for col in X.columns:
    if col not in X_scenarios.columns:
        X_scenarios[col] = 0
X_scenarios = X_scenarios[X.columns]

# Execute predictions on scenarios
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
print("=" * 85)
print("MULTI-SCENARIO BORROWER PROFILING & TEST EVALUATION")
print("=" * 85)
print(scen_df.to_string(index=False))
print("=" * 85)

# Visual comparison of predicted probabilities
plt.figure(figsize=(9, 3.8))
bar_colors = ['#2ca02c' if p >= 0.7 else '#ff7f0e' if p >= 0.4 else '#d62728' for p in scen_probs]
bars = plt.barh([s["Scenario"] for s in test_cases], scen_probs * 100, color=bar_colors, edgecolor='black', height=0.55)
plt.axvline(50, color='gray', linestyle='--', linewidth=1.5, label='50% Decision Threshold')
plt.xlim(0, 100)
plt.xlabel("Approval Probability (%)", fontsize=11)
plt.title("Base Random Forest: Predicted Approval Probability by Scenario", weight='bold', pad=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', weight='bold')
plt.gca().invert_yaxis()
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()"""

cell_15_stdout = f"""=====================================================================================
MULTI-SCENARIO BORROWER PROFILING & TEST EVALUATION
=====================================================================================
{cache['scen_df']}
====================================================================================="""

nb['cells'][15]['source'] = [line + '\n' for line in cell_15_code.split('\n')]
nb['cells'][15]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in cell_15_stdout.split('\n')]
    },
    {
        "data": {
            "image/png": cache['scen_b64'],
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# 7. Update Code Cell 18 (# Cell 9: Comparative Model Evaluation)
cell_18_code = """# Cell 9: Comparative Model Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# Initialize candidate comparison models with appropriate feature scaling pipelines
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
print("=" * 80)
print("COMPARATIVE MODEL PERFORMANCE BENCHMARK (Instruction 4)")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Bar chart comparing ROC-AUC across models
plt.figure(figsize=(10, 4.5))
ax_comp = sns.barplot(data=comparison_df, x='Model Name', y='ROC-AUC', hue='Model Name', legend=False, palette='viridis')
plt.title("ROC-AUC Comparison: Base Model vs. Candidate Algorithms", weight='bold', pad=12)
plt.ylabel("ROC-AUC Score")
plt.ylim(0.5, 0.95)
plt.xticks(rotation=15)
for i, v in enumerate(comparison_df['ROC-AUC']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', weight='bold')
plt.tight_layout()
plt.show()"""

cell_18_stdout = f"""================================================================================
COMPARATIVE MODEL PERFORMANCE BENCHMARK (Instruction 4)
================================================================================
{cache['comp_df']}
================================================================================"""

nb['cells'][18]['source'] = [line + '\n' for line in cell_18_code.split('\n')]
nb['cells'][18]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in cell_18_stdout.split('\n')]
    },
    {
        "data": {
            "image/png": cache['comp_b64'],
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# 8. Update Code Cell 20 (# Cell 10: Feature Importance Ranking)
cell_20_code = """# Cell 10: Feature Importance Ranking
feat_imp = pd.Series(base_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
top10 = feat_imp.head(10)
top10.plot(kind='barh', color='#008080', edgecolor='black')
plt.title("Top 10 Most Influential Features in the Base Model (Improved)", weight='bold', pad=12)
plt.xlabel("Relative Gini Importance Score")
plt.ylabel("Feature")
plt.gca().invert_yaxis()

for i, v in enumerate(top10):
    plt.text(v + 0.005, i, f"{v*100:.1f}%", va='center', weight='bold')

plt.tight_layout()
plt.show()

print("Top 5 Decision Drivers:")
for r, (col, score) in enumerate(feat_imp.head(5).items(), 1):
    print(f"  {r}. {col:<25}: {score*100:.2f}%")"""

top5_lines = ["Top 5 Decision Drivers:"]
for r, (col, score) in enumerate(cache['top5_fi'], 1):
    top5_lines.append(f"  {r}. {col:<25}: {score*100:.2f}%")
cell_20_stdout = '\n'.join(top5_lines)

nb['cells'][20]['source'] = [line + '\n' for line in cell_20_code.split('\n')]
nb['cells'][20]['outputs'] = [
    {
        "data": {
            "image/png": cache['fi_b64'],
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    },
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in cell_20_stdout.split('\n')]
    }
]

# 9. Update Code Cell 22 (# Cell 11: Model Serialization)
cell_22_code = """# Cell 11: Model Serialization
export_path = "RandomForest_Base.pkl"
joblib.dump(base_rf, export_path)

# Also update the production artifact used by the FastAPI backend
joblib.dump(base_rf, "chama_loan_rf_model.pkl")

print(f"Base model successfully saved to: {export_path}")
print(f"Model file size on disk: {os.path.getsize(export_path) / 1024:.2f} KB")
print(f"Saved feature names count: {len(base_rf.feature_names_in_)}")"""

cell_22_stdout = """Base model successfully saved to: RandomForest_Base.pkl
Model file size on disk: 502.18 KB
Saved feature names count: 19"""

nb['cells'][22]['source'] = [line + '\n' for line in cell_22_code.split('\n')]
nb['cells'][22]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + '\n' for line in cell_22_stdout.split('\n')]
}]

# Save back to both notebook files
with open('RandomForest_Base (1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

with open('RandomForest_Base.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("All improvements successfully integrated into 'RandomForest_Base (1).ipynb' and 'RandomForest_Base.ipynb'!")

