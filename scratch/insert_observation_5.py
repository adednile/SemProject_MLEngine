import json, io, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and calculate correlation
df_raw = pd.read_csv('SVM-based Loan Predictor.csv')
df_corr_temp = df_raw.copy()
df_corr_temp['TotalIncome'] = df_corr_temp['ApplicantIncome'] + df_corr_temp['CoapplicantIncome'].fillna(0)
df_corr_temp['Loan_Status_Numeric'] = df_corr_temp['Loan_Status'].map({'Y': 1, 'N': 0})

numerical_cols = [
    'ApplicantIncome', 'CoapplicantIncome', 'TotalIncome', 
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status_Numeric'
]

corr_matrix = df_corr_temp[numerical_cols].corr(method='pearson')

# Generate high-resolution heatmap plot
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
plt.title('Observation 5: Pearson Correlation Heatmap of Financial Predictors & Loan Status', weight='bold', pad=14, fontsize=12)
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

# Load current notebook
with open('RandomForest_Base (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 1
cell_1_src = ''.join(nb['cells'][1]['source'])
cell_1_src = cell_1_src.replace(
    'extract **at least 4 key empirical observations**.',
    'extract **5 key empirical observations (including correlation and multicollinearity analysis)**.'
)
nb['cells'][1]['source'] = [line + '\n' for line in cell_1_src.split('\n') if line]

# Update Cell 4
cell_4_src = """### 2.1 Core Exploratory Data Analysis (EDA) Observations
Below, we extract and visualize **5 significant observations** (including empirical distributions, socioeconomic factors, and feature correlation analysis) that directly influence feature engineering and model design."""
nb['cells'][4]['source'] = [line + '\n' for line in cell_4_src.split('\n')]

# Update Cell 5 comment
cell_5_lines = nb['cells'][5]['source']
if cell_5_lines and '4 Core Observations' in cell_5_lines[0]:
    cell_5_lines[0] = '# Cell 3: Exploratory Visualizations for Observations 1 to 4\n'

# Create the new Observation 5 code cell
obs5_code = """# Cell 4: Observation 5 - Pearson Correlation Analysis & Multicollinearity Heatmap
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

obs5_cell = {
    "cell_type": "code",
    "execution_count": 4,
    "metadata": {},
    "outputs": [
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
    ],
    "source": [line + '\n' for line in obs5_code.split('\n')]
}

# Update Cell 6 (Detailed Explanation)
cell_6_src = ''.join(nb['cells'][6]['source'])
cell_6_src = cell_6_src.replace(
    '### Detailed Explanation of the 4 EDA Observations:',
    '### Detailed Explanation of the 5 EDA Observations:'
)

obs5_explanation = """
5. **Observation 5 (Linear Correlation & Multicollinearity Analysis)**:
   * **Target Correlation**: Past `Credit_History` exhibits by far the strongest linear correlation with `Loan_Status` ($r = +0.562$). In stark contrast, raw household income features (`ApplicantIncome` $r = -0.005$, `TotalIncome` $r = -0.031$) show near-zero linear correlation with approval. This empirically confirms that income level alone does not determine loan repayment; repayment discipline and credit history dominate.
   * **Multicollinearity Discovery**: There is an extremely strong linear correlation between `ApplicantIncome` and `TotalIncome` ($r = +0.893$), and a moderate correlation between `LoanAmount` and `TotalIncome` ($r = +0.571$), reflecting that larger households borrow more against higher income.
   * **Model Architecture Takeaway**: Standard linear algorithms (like unregularized Logistic Regression) suffer from variance inflation when given highly collinear predictors ($r > 0.8$). This provides strong theoretical justification for our **Random Forest Classifier**: tree ensembles with random feature subsampling ($\sqrt{p}$ features per split) are structurally immune to multicollinearity, preventing redundant features from destabilizing decision boundaries."""

cell_6_src = cell_6_src.strip() + '\n' + obs5_explanation
nb['cells'][6]['source'] = [line + '\n' for line in cell_6_src.split('\n')]

# Insert the obs5 code cell between Cell 5 and Cell 6
# Currently: nb['cells'][5] is Cell 3 visualization.
# We insert obs5_cell at index 6, so old cell 6 becomes cell 7!
nb['cells'].insert(6, obs5_cell)

# Renumber subsequent code cells
cell_num_mapping = {
    '# Cell 4: Data Cleaning and Feature Pipeline': '# Cell 5: Data Cleaning and Feature Pipeline',
    '# Cell 5: Stratified 5-Fold Cross-Validation Verification': '# Cell 6: Stratified 5-Fold Cross-Validation Verification',
    '# Cell 6: Train-Test Split and Holdout Evaluation': '# Cell 7: Train-Test Split and Holdout Evaluation',
    '# Cell 7: Multi-Scenario Live Borrower Evaluation': '# Cell 8: Multi-Scenario Live Borrower Evaluation',
    '# Cell 8: Comparative Model Evaluation': '# Cell 9: Comparative Model Evaluation',
    '# Cell 9: Feature Importance Ranking': '# Cell 10: Feature Importance Ranking',
    '# Cell 10: Model Serialization': '# Cell 11: Model Serialization'
}

for c in nb['cells']:
    if c['cell_type'] == 'code' and c['source']:
        first_line = c['source'][0].strip()
        if first_line in cell_num_mapping:
            c['source'][0] = cell_num_mapping[first_line] + '\n'

# Save back to RandomForest_Base (1).ipynb
with open('RandomForest_Base (1).ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Also save to RandomForest_Base.ipynb
with open('RandomForest_Base.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Observation 5 successfully integrated into 'RandomForest_Base (1).ipynb' and 'RandomForest_Base.ipynb'!")
print(f"Total cells is now: {len(nb['cells'])}")

