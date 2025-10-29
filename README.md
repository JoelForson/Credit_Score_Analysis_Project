# Credit Score Classification Model
### Executive Summary
<p>This project develops a machine learning classification system to predict credit scores (Good, Standard, Poor) using demographic, financial, and behavioral data from 100,000 customer records spanning 8 months. The model simulates a data-driven credit assessment tool that financial institutions could deploy to enhance risk prediction and lending decisions.<p>
Through comprehensive data preprocessing, feature engineering, and evaluation of multiple algorithms, the project achieved 80% overall accuracy with balanced performance across credit categories. The Random Forest Classifier emerged as the best-performing model with an F1-macro score of 0.77, demonstrating robust predictive capability while maintaining fairness across all three credit classes.
Key findings reveal that engineered financial ratios—particularly debt-to-income ratio and credit utilization—are strong indicators of creditworthiness. This work emphasizes both the predictive power and ethical importance of transparent, interpretable AI in financial modeling.

# Project Overview
### Objective

Clean and preprocess raw credit data containing missing values, formatting errors, and outliers
Engineer meaningful features that capture financial risk indicators
Train and compare multiple machine learning models for classification
Evaluate model performance using cross-validation and balanced metrics
Deliver insights into key predictors of credit score categories

### Dataset

#### Source: Synthetic credit bureau data
- Size: 100,000 records → 98,800 after cleaning (1,200 dropped due to missing Monthly_Balance)
- Time Period: 8 months of monthly observations (12,500 unique customers × 8 months)
- Features: 28 columns including demographic, financial, and behavioral indicators
- Target: Credit Score (Good, Standard, Poor)

### Class Distribution
- Credit Score Count Percentage: 
  - Good: 17.8%,
  - Standard: 53.2%
  - Poor:29.0% 
<p>Significant class imbalance detected, addressed through stratified sampling and class weighting.</p>

🔍 Data Understanding & Quality Issues
Original Dataset Issues

### Missing Values:

- Monthly_Inhand_Salary: 15,002 missing (15%)
- Credit_History_Age: 9,030 missing (9%)
- Type_of_Loan: 11,408 missing (11.4%)
- Num_of_Delayed_Payment: 7,002 missing (7%)


### Data Type Errors:

- Age stored as object (should be integer)
- Annual_Income stored as object (contained trailing underscores)
- Num_of_Loan stored as object (mixed with special characters)


### Outliers & Invalid Values:

- Age: 7,580 (invalid), -500 (negative)
- Num_Bank_Accounts: up to 1,798 (unrealistic)
- Num_Credit_Card: 1,499 (unrealistic)
- Interest_Rate: 5,797% (impossible)
- Annual_Income: 10,909,427 (extreme outlier)


### Special Characters:

- SSN: "#F%$D@*&8" placeholder for 5,572 records
- Payment_Behaviour: "!@9#%8" (7,600 records)
- Multiple columns contained "_" as null indicators

### Feature Statistics (After Cleaning)

| Feature               | Threshold           | Rows Affected |
|-----------------------|-------------------|---------------|
| Age                   | <= 110 and > 0    | ~1,500        |
| Num_Bank_Accounts     | < 100 and >= 1    | ~800          |
| Num_Credit_Card       | < 50              | ~1,200        |
| Interest_Rate         | < 50 %            | ~1,000        |
| Num_Credit_Inquiries  | < 500             | ~500          |


# 🛠️ Data Preprocessing Pipeline
## 1. Data Cleaning

#### Actions Taken:

- ✅ Removed 1,200 rows with missing Monthly_Balance (critical feature)
- ✅ Forward-filled missing Monthly_Inhand_Salary (uses previous known value for same customer)
- ✅ Replaced missing categorical values with mode:
  - Num_of_Delayed_Payment → mode = 19
  - Amount_invested_monthly → mode = "10000" → replaced with numeric values
  - Num_Credit_Inquiries → mode = 4
- ✅ Filled missing Type_of_Loan with "No loan" category

## 2. Data Type Standardization
```python
# Removed trailing underscores and special characters
df['Age'] = df['Age'].replace('_', '', regex=True).astype('Int64')
df['Annual_Income'] = df['Annual_Income'].replace('_', '', regex=True).astype('float64')
df['Outstanding_Debt'] = df['Outstanding_Debt'].replace('_', '', regex=True).astype('float64')
```
## 3. Outlier Treatment
<p>Applied capping to maintain data integrity:</p>

| Feature               | Threshold           | Rows Affected |
|-----------------------|-------------------|---------------|
| Age                   | <= 110 and > 0    | ~1,500        |
| Num_Bank_Accounts     | < 100 and >= 1    | ~800          |
| Num_Credit_Card       | < 50              | ~1,200        |
| Interest_Rate         | < 50 %            | ~1,000        |
| Num_Credit_Inquiries  | < 500             | ~500          |

</p>Invalid values replaced via forward fill (inherits last valid value per customer).</p>

## 4. Feature Engineering
New Feature: debt_to_income_ratio
```python
pythondf['debt_to_income_ratio'] = np.where(
    df['Annual_Income'] > 0,
    df['Outstanding_Debt'] / df['Annual_Income'],
    0
)
```
Transformation: Credit_History_Age → Credit_History_Months
```python Converted "22 Years and 3 Months" → 267 months
def parse_age(age):
    match = re.match(r"(\d+)\s+Years\s+and\s+(\d+)\s+Months", age)
    if match:
        years, months = map(int, match.groups())
        return years * 12 + months
    return np.nan
```
## 6. Encoding

- Ordinal Encoder: Applied to Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour
- Label Encoder: Target variable Credit_Score (Good=0, Poor=1, Standard=2)

## 7. Scaling
StandardScaler applied to all numeric features (excluding target):
```python Custom transformer to preserve DataFrame structure
class DataFrameStandardScaler:
    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.columns] = self.scaler.transform(X_scaled[self.columns])
        return X_scaled
```

# 🤖 Model Development
### Train-Test Split

- Training Set: 79,040 samples (80%)
- Test Set: 19,760 samples (20%)
Method: Stratified split to preserve class proportions

### Cross-Validation Strategy

Method: 5-Fold Stratified K-Fold
Purpose: Ensure each fold maintains class distribution
Metrics: F1-macro (primary), F1-weighted, Accuracy

Models Evaluated
1. Random Forest Classifier ⭐ Best Performer
```python
pythonRandomForestClassifier(class_weight='balanced', random_state=42)
```
# Cross-Validation Results:

- Fold 1: 0.769
- Fold 2: 0.766
- Fold 3: 0.773
- Fold 4: 0.775
- Fold 5: 0.772
Mean F1-Macro: 0.771

### Test Set Performance:
MetricScore Overall Accuracy: 80% | F1-Macro: 0.78 | F1-Weighted: 0.80
#### Class-Level Performance:
ClassPrecisionRecall F1-Score Support 
- Good0.770.710.743,516
- Poor0.790.800.805,730
- Standard0.810.820.8110,514
- 
### Confusion Matrix (Normalized):
### Confusion Matrix (Normalized)

| True \ Pred | Good | Poor | Standard |
|------------|------|------|---------|
| Good       | 0.71 | 0.18 | 0.11    |
| Poor       | 0.12 | 0.80 | 0.08    |
| Standard   | 0.07 | 0.11 | 0.82    |

## 2. Decision Tree Classifier
``` python
DecisionTreeClassifier(class_weight='balanced', random_state=42)
```
F1-Macro Score: 0.668
Issue: Overfitting; lower generalization than Random Forest

### 3. Logistic Regression
```python
LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
```
- **F1-Macro Score**: 0.605
- **Insight**: Struggles with non-linear relationships in credit data

---

## 📈 Model Interpretation

### Why Random Forest Won

1. **Handles Non-Linearity**: Credit risk relationships are complex and non-linear
2. **Resistant to Outliers**: Ensemble averaging reduces impact of extreme values
3. **Feature Interactions**: Automatically captures relationships between features (e.g., debt-to-income × payment history)
4. **Class Balancing**: `class_weight='balanced'` addressed imbalance effectively

### Key Predictive Features

Based on Random Forest feature importance (top 10):

1. Outstanding_Debt - Total amount owed (strongest predictor)
2. Interest_Rate - Cost of borrowing indicates risk profile
3. Credit_History_Months - Length of credit history
4. Delay_from_due_date - Average payment delay
5. Changed_Credit_Limit - Credit limit adjustments over time
6. Credit_Mix - Diversity of credit types (Good/Standard/Bad)
7. debt_to_income_ratio - Engineered feature showing repayment capacity
8. Monthly_Balance - Available cash flow
9. Credit_Utilization_Ratio - Percentage of credit used
10. Amount_invested_monthly - Savings/investment behavior


---

## 🏗️ Project Structure
```
CreditScore_ML/
│
├── data/
│   ├── Credit_train.csv                    # Raw training data (100K rows)
│   ├── Credit_test.csv                     # Raw test data
│   └── CleanedCreditScoreData.csv          # Processed dataset (98.8K rows)
│
├── notebooks/
│   ├── Credit_Score_data_understanding.ipynb   # EDA & visualization
│   ├── CreditScore_PortProject_cleaning.ipynb  # Data preprocessing
│   └── CreditScore_PortProject_ML.ipynb        # Model training & evaluation
│
├── README.md                               # This file
└── requirements.txt                        # Python dependencies

💻 Technologies Used
CategoryToolsLanguagePython 3.12Data Manipulationpandas, NumPyMachine Learningscikit-learnVisualizationMatplotlib, SeabornStatistical AnalysisSciPy
Key Libraries
pythonimport pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score

🚀 Quick Start
Installation
bash# Clone repository
git clone https://github.com/yourusername/credit-score-ml.git
cd credit-score-ml

# Install dependencies
pip install -r requirements.txt
Running the Pipeline
Step 1: Data Cleaning
bashjupyter notebook notebooks/CreditScore_PortProject_cleaning.ipynb
Step 2: Model Training
bashjupyter notebook notebooks/CreditScore_PortProject_ML.ipynb
Making Predictions
python# Load preprocessed data
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load cleaned data
df = pd.read_csv('data/CleanedCreditScoreData.csv')

# Prepare features and target
X = df.drop(['Credit_Score', 'ID', 'Customer_ID', 'SSN', 
             'Month', 'Name', 'Occupation', 'Type_of_Loan'], axis=1)
y = df['Credit_Score']

# Train model
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)

# Predict
predictions = rf_clf.predict(X_test)

📐 Data Preprocessing Details
Pipeline Architecture
pythonPipeline([
    ('preprocess', Preprocessor(
        numeric_cols=17 features,
        categorical_cols=['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'],
        target_col=['Credit_Score'],
        col_to_drop=['ID', 'Customer_ID', 'SSN', 'Month', 'Name', 'Occupation', 'Type_of_Loan']
    )),
    ('scaler', DataFrameStandardScaler(exclude_cols=['Credit_Score']))
])
Feature List
Numeric Features (17):

Age
Annual_Income
Monthly_Inhand_Salary
Num_Bank_Accounts
Num_Credit_Card
Interest_Rate
Num_of_Loan
Delay_from_due_date
Num_of_Delayed_Payment
Changed_Credit_Limit
Num_Credit_Inquiries
Outstanding_Debt
Credit_Utilization_Ratio
Total_EMI_per_month
Amount_invested_monthly
Monthly_Balance
Credit_History_Months (engineered)

Categorical Features (3):

Credit_Mix (Good, Standard, Bad, _)
Payment_of_Min_Amount (Yes, No, NM)
Payment_Behaviour (6 categories + 1 invalid)

Engineered Features (2):

debt_to_income_ratio = Outstanding_Debt / Annual_Income
Credit_History_Months = Converted from "X Years and Y Months" string


📊 Model Comparison
ModelAccuracyF1-MacroF1-WeightedTraining TimeBest ForRandom Forest0.800.770.80~45sProduction deploymentDecision Tree0.760.670.75~5sQuick baselineLogistic Regression0.790.610.77~8sInterpretability
Why Random Forest Excels
✅ Handles Imbalance: class_weight='balanced' ensures minority class (Good) receives proper attention
✅ Robust to Outliers: Despite extreme income values, ensemble averaging maintains stability
✅ Feature Interactions: Automatically captures complex relationships (e.g., high debt + low income = Poor)
✅ Prevents Overfitting: Out-of-bag validation and max_depth constraints

🎯 Key Insights
1. Financial Behavior Matters Most
Top 3 Predictors:

Credit Utilization Ratio (28-36% is optimal)
Outstanding Debt (inversely correlated with Good scores)
Debt-to-Income Ratio (engineered feature showing repayment capacity)

2. Payment History is Critical

Customers with 0 delayed payments → 85% likelihood of Good/Standard score
>20 delayed payments → 78% likelihood of Poor score

3. Credit Mix Diversity

"Good" credit mix → 65% probability of Good/Standard score
No credit history ("_" category) → 55% probability of Poor score

4. Model Fairness
Balanced Performance Across Classes:

Good: 74% F1-score (hardest to predict due to smallest sample size)
Poor: 80% F1-score
Standard: 81% F1-score (easiest due to largest sample size)


📉 Limitations & Challenges
Data Quality

Synthetic nature: Data may not reflect real-world credit distributions
Missing data patterns: Monthly_Inhand_Salary often missing after first month (suggests data collection issue)
Extreme outliers: Some values (e.g., Annual_Income > $24M) required aggressive capping

Model Limitations

No temporal features: Monthly observations not leveraged for time-series analysis
Linear interpolation: Credit_History_Age gaps filled with linear interpolation (may introduce noise)
Class imbalance: Good credit scores underrepresented (17.8% vs 53.2% Standard)


🔮 Future Enhancements
Short-Term (Next Sprint)

Hyperparameter Tuning

python   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30],
       'min_samples_split': [2, 5, 10]
   }

Handle Imbalance with SMOTE

python   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

Feature Selection: Remove low-importance features to reduce dimensionality

Long-Term (Future Releases)

Explainable AI

Integrate SHAP values for model transparency
Generate customer-specific credit score explanations


Advanced Models

XGBoost (resolve dependency issues)
LightGBM for faster training
Neural networks for deep learning approach


Deployment

Build Flask/FastAPI REST API
Create Streamlit dashboard for interactive predictions
Dockerize application for production


Temporal Analysis

Leverage 8-month time series data
Build LSTM/GRU for credit score trajectory prediction
Detect deteriorating credit health early




📝 Installation Requirements
txtpandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
jupyter>=1.0.0
Save as requirements.txt and install:
bashpip install -r requirements.txt

#Areas for improvement:

- Feature engineering ideas
- Alternative modeling approaches
- Deployment strategies
- Code optimization

👨‍💼 Author
Joel Forson
B.S. in Business Administration
Concentration: Management Information Systems & Business Analytics
Institution: Northeastern University
