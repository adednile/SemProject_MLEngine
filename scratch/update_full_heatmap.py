import json, io, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and prepare full encoded version
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

df_all_encoded = pd.get_dummies(df_clean, drop_first=True)
corr_full = df_all_encoded.corr()

# Generate 16x16 comprehensive heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_full, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0, 
    fmt='.2f', 
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
)
plt.title('Observation 5: Complete Pearson Correlation Heatmap (All 11 Original Features + Engineered TotalIncome)', weight='bold', pad=14, fontsize=12)
plt.xticks(rotation=40, ha='right', fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

target_corr = corr_full['Loan_Status'].drop('Loan_Status').sort_values(ascending=False)
stdout_text = "Linear Correlation with Target (Loan_Status) Across ALL Encoded Features:\n"
for col, val in target_corr.items():
    stdout_text += f"  • {col:<24}: r = {val:+.3f}\n"

# Load notebook
with open('RandomForest_Base (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the code in Cell 6 (which is code cell 4)
obs5_code = """# Cell 4: Observation 5 - Complete Pearson Correlation Analysis (All 11 Features + TotalIncome)
# To evaluate all 11 features (numerical + categorical), we temporarily encode categorical variables:
df_corr_all = df_raw.drop(columns=['Loan_ID']).dropna(subset=['Loan_Status']).copy()

# Median imputation for numeric, mode for categorical to compute correlations
for col in df_corr_all.select_dtypes(include=[np.number]).columns:
    df_corr_all[col] = df_corr_all[col].fillna(df_corr_all[col].median())
for col in df_corr_all.select_dtypes(include=['object', 'str']).columns:
    if col != 'Loan_Status':
        df_corr_all[col] = df_corr_all[col].fillna(df_corr_all[col].mode()[0])

df_corr_all['TotalIncome'] = df_corr_all['ApplicantIncome'] + df_corr_all['CoapplicantIncome']
df_corr_all['Loan_Status'] = df_corr_all['Loan_Status'].map({'Y': 1, 'N': 0})

# One-hot encode categorical features so Pearson's r can be computed across all features
df_corr_encoded = pd.get_dummies(df_corr_all, drop_first=True)
corr_matrix_full = df_corr_encoded.corr(method='pearson')

# Plot complete 16x16 correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_matrix_full, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0, 
    fmt='.2f', 
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
)
plt.title("Observation 5: Complete Pearson Correlation Heatmap (All 11 Features + TotalIncome)", weight='bold', pad=14, fontsize=12)
plt.xticks(rotation=40, ha='right', fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
plt.tight_layout()
plt.show()

# Print correlations of all features with the target
print("Linear Correlation with Target (Loan_Status) Across ALL Encoded Features:")
target_corr_all = corr_matrix_full['Loan_Status'].drop('Loan_Status').sort_values(ascending=False)
for col, val in target_corr_all.items():
    print(f"  • {col:<24}: r = {val:+.3f}")"""

nb['cells'][6]['source'] = [line + '\n' for line in obs5_code.split('\n')]
nb['cells'][6]['outputs'] = [
    {
        "data": {
            "image/png": img_b64,
            "text/plain": ["<Figure size ...>"]
        },
        "metadata": {},
        "output_type": "display_data"
    },
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [line + '\n' for line in stdout_text.strip().split('\n')]
    }
]

# Update explanation in Cell 7 to reflect all features
cell_7_src = ''.join(nb['cells'][7]['source'])
# Update observation 5 part
updated_obs5_explanation = """5. **Observation 5 (Complete Correlation & Multicollinearity Analysis Across All 11 Features)**:
   * **Why Categorical Features Require Encoding**: Pearson's correlation requires numeric values. In the raw dataset, 6 features (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`) are text strings ('Male', 'Yes', 'Urban', etc.). By dummy-encoding them, we can analyze the complete 16x16 correlation structure.
   * **Target Correlation Hierarchy**: 
     - `Credit_History` possesses the single highest correlation with `Loan_Status` ($r = +0.541$).
     - `Property_Area_Semiurban` ($r = +0.137$) and `Married_Yes` ($r = +0.091$) exhibit positive correlations with loan approval.
     - `Education_Not Graduate` ($r = -0.086$) and `CoapplicantIncome` ($r = -0.059$) show mild negative correlations.
     - Pure income metrics (`ApplicantIncome` $r = -0.005$, `TotalIncome` $r = -0.031$) have near-zero linear correlation, showing that higher income alone is not a linear predictor of loan approval without credit discipline.
   * **Multicollinearity Insights**:
     - Extreme linear collinearity exists between `ApplicantIncome` and `TotalIncome` ($r = +0.893$).
     - Moderate collinearity exists between `LoanAmount` and `TotalIncome` ($r = +0.620$).
   * **Base Model Justification**: While linear models (like Logistic Regression) suffer from variance inflation under collinear predictors ($r > 0.8$), our **Random Forest Classifier** is structurally immune because it subsamples features ($\sqrt{p}$) at each split and splits non-linearly without matrix inversion."""

# Replace observation 5 text in cell 7
parts = cell_7_src.split('5. **Observation 5')
new_cell_7 = parts[0].strip() + '\n\n' + updated_obs5_explanation
nb['cells'][7]['source'] = [line + '\n' for line in new_cell_7.split('\n')]

# Save back to both files
with open('RandomForest_Base (1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

with open('RandomForest_Base.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Observation 5 to include all 11 original features + TotalIncome!")

