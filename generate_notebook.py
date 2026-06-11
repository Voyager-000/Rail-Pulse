import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# 🚂 RailPulse: AI-Powered Railway Risk & Resource Allocation
**Hackathon Presentation Notebook**

This notebook walks through the data generation, feature engineering, and model training process for RailPulse. 
Our model predicts coach-level overcrowding, ticketless travel, and station congestion using existing booking data.
"""

code_1 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the synthetic training data
df = pd.read_csv('../data/train_data.csv')
print(f"Dataset loaded: {len(df):,} records")
df.head()
"""

text_2 = """\
## 📊 1. Exploratory Data Analysis
Let's look at the distribution of occupancy and the impact of festivals.
"""

code_2 = """\
plt.figure(figsize=(10, 5))
sns.histplot(df['occupancy_ratio'], bins=50, kde=True, color='purple')
plt.title('Distribution of Coach Occupancy Ratio')
plt.axvline(1.0, color='red', linestyle='--', label='100% Capacity')
plt.legend()
plt.show()
"""

code_3 = """\
# Impact of festivals on overcrowding
festival_impact = df.groupby('day_type')['overcrowding_label'].mean() * 100
print("Overcrowding Probability by Day Type:")
print(festival_impact)
"""

text_3 = """\
## ⚙️ 2. Feature Engineering & Model Training
We use XGBoost classifiers to predict two separate risks:
1. **Overcrowding Risk** (Probability of >100% capacity + standing passengers)
2. **Ticketless Travel Risk** (Probability of unauthorized passengers)
"""

code_4 = """\
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Features for Overcrowding Model
features = ['is_holiday', 'is_weekend', 'occupancy_ratio', 'historical_avg_occupancy', 'route_popularity']
X = df[features]
y = df['overcrowding_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(f"AUC Score: {roc_auc_score(y_test, proba):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, preds))
"""

text_4 = """\
## 📈 3. Feature Importance
What drives the AI's decisions? Let's check what features the model prioritizes.
"""

code_5 = """\
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Top Risk Factors for Overcrowding')
plt.xlabel('XGBoost Feature Importance')
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_5)
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/RailPulse_Model.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook generated successfully at notebooks/RailPulse_Model.ipynb")
