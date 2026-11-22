import json, io, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Re-generate the previous numeric correlation heatmap
df_raw = pd.read_csv('SVM-based Loan Predictor.csv')
df_corr_temp = df_raw.copy()
df_corr_temp['TotalIncome'] = df_corr_temp['ApplicantIncome'] + df_corr_temp['CoapplicantIncome'].fillna(0)
df_corr_temp['Loan_Status_Numeric'] = df_corr_temp['Loan_Status'].map({'Y': 1, 'N': 0})

numerical_cols = [
    'ApplicantIncome', 'CoapplicantIncome', 'TotalIncome', 
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status_Numeric'
]

corr_matrix = df_corr_temp[numerical_cols].corr(method='pearson')

plt.figure(figsize=(9, 6.5))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0, 
    fmt='.2f', 
    linewidths=0.8,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
)
plt.title("Observation 5: Pearson Correlation Heatmap of Financial Predictors & Loan Status", weight='bold', pad=14, fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

target_corr = corr_matrix['Loan_Status_Numeric'].drop('Loan_Status_Numeric').sort_values(ascending=False)
stdout_text = "Linear Correlation with Target (Loan_Status):\n"
for col, val in target_corr.items():
    stdout_text += f"  • {col:<22}: r = {val:+.3f}\n"

# 2. Previous Code string
prev_obs5_code = """# Cell 4: Observation 5 - Pearson Correlation Analysis & Multicollinearity Heatmap
df_corr_temp = df_raw.copy()
# Create temporary numeric features for correlation analysis
df_corr_temp['TotalIncome'] = df_corr_temp['ApplicantIncome'] + df_corr_temp['CoapplicantIncome'].fillna(0)
df_corr_temp['Loan_Status_Numeric'] = df_corr_temp['Loan_Status'].map({'Y': 1, 'N': 0})

numerical_cols = [
    'ApplicantIncome', 'CoapplicantIncome', 'TotalIncome', 
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status_Numeric'
]

# Calculate Pearson correlation matrix
corr_matrix = df_corr_temp[numerical_cols].corr(method='pearson')

plt.figure(figsize=(9, 6.5))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0, 
    fmt='.2f', 
    linewidths=0.8,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
)
plt.title("Observation 5: Pearson Correlation Heatmap of Financial Predictors & Loan Status", weight='bold', pad=14, fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print key pairwise correlations with target
print("Linear Correlation with Target (Loan_Status):")
target_corr = corr_matrix['Loan_Status_Numeric'].drop('Loan_Status_Numeric').sort_values(ascending=False)
for col, val in target_corr.items():
    print(f"  • {col:<22}: r = {val:+.3f}")"""

# 3. Previous Markdown explanation string
prev_obs5_explanation = """5. **Observation 5 (Linear Correlation & Multicollinearity Analysis)**:
   * **Target Correlation**: Past `Credit_History` exhibits by far the strongest linear correlation with `Loan_Status` ($r = +0.562$). In stark contrast, raw household income features (`ApplicantIncome` $r = -0.005$, `TotalIncome` $r = -0.031$) show near-zero linear correlation with approval. This empirically confirms that income level alone does not determine loan repayment; repayment discipline and credit history dominate.
   * **Multicollinearity Discovery**: There is an extremely strong linear correlation between `ApplicantIncome` and `TotalIncome` ($r = +0.893$), and a moderate correlation between `LoanAmount` and `TotalIncome` ($r = +0.571$), reflecting that larger households borrow more against higher income.
   * **Model Architecture Takeaway**: Standard linear algorithms (like unregularized Logistic Regression) suffer from variance inflation when given highly collinear predictors ($r > 0.8$). This provides strong theoretical justification for our **Random Forest Classifier**: tree ensembles with random feature subsampling ($\\sqrt{p}$ features per split) are structurally immune to multicollinearity, preventing redundant features from destabilizing decision boundaries."""

# Load notebook
with open('RandomForest_Base (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 6 (Code cell 4)
nb['cells'][6]['source'] = [line + '\n' for line in prev_obs5_code.split('\n')]
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

# Update Cell 7 (Explanation)
cell_7_src = ''.join(nb['cells'][7]['source'])
parts = cell_7_src.split('5. **Observation 5')
new_cell_7 = parts[0].strip() + '\n\n' + prev_obs5_explanation
nb['cells'][7]['source'] = [line + '\n' for line in new_cell_7.split('\n')]

# Save back to both files
with open('RandomForest_Base (1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

with open('RandomForest_Base.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully reverted to the previous iteration of the correlation heatmap matrix!")

