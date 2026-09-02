# 🏦 Bank Marketing — Machine Learning Preprocessing & Feature Selection

## 📌 Project Overview

This project presents a complete practical implementation of **Data Preprocessing and Feature Selection** using the **Bank Marketing Dataset**.

The main objective is to understand how raw banking campaign data can be cleaned, transformed, analyzed, and prepared for Machine Learning.

Instead of directly using ready-made Machine Learning functions, the major preprocessing and feature-selection techniques are first implemented **from scratch** using Python, Pandas, NumPy, loops, conditions, lists, and dictionaries. The manually calculated results are then verified using appropriate libraries.

The project follows the workflow:

> **Understand → Calculate → Code → Verify → Interpret**

The dataset represents a **bank marketing campaign**, where the target variable `y` indicates whether a customer subscribed to a term deposit.

Therefore, this project is a **Binary Classification** problem.

---

## 🎯 Problem Statement

Banks conduct marketing campaigns to contact customers and encourage them to subscribe to financial products such as term deposits.

The objective of this project is to analyze customer and campaign-related information and identify the important factors associated with whether a customer subscribes to a term deposit.

### Target Variable

- `yes` → Customer subscribed to a term deposit
- `no` → Customer did not subscribe to a term deposit

### Problem Type

**Binary Classification**

---

## 🎯 Project Objectives

- Understand the Bank Marketing dataset.
- Perform Exploratory Data Analysis.
- Calculate basic statistical measures.
- Identify missing values.
- Detect duplicate records.
- Check invalid and inconsistent data.
- Implement Label Encoding from scratch.
- Implement One-Hot Encoding from scratch.
- Detect outliers using IQR.
- Detect outliers using Z-score.
- Analyze and justify outlier treatment.
- Apply suitable data transformations.
- Implement Min-Max Normalization from scratch.
- Implement Standardization from scratch.
- Implement Train-Test Split from scratch.
- Understand and prevent Data Leakage.
- Implement Variance Threshold from scratch.
- Implement Pearson Correlation from scratch.
- Implement Chi-Square from scratch.
- Implement ANOVA F-Test from scratch.
- Implement Mutual Information from scratch.
- Select important features.
- Compare the dataset before and after feature selection.
- Verify manual calculations using libraries.
- Create meaningful visualizations.
- Validate feature selection using a Machine Learning model.

---

# 📊 Dataset Description

The project uses the **Bank Marketing Dataset** containing information about customers and their interactions with a bank marketing campaign.

| Parameter | Value |
|---|---:|
| Total Records | 45,211 |
| Total Columns | 17 |
| Input Features | 16 |
| Target Variable | `y` |
| Numerical Features | 7 |
| Categorical Features | 9 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Problem Type | Binary Classification |

## Numerical Features

```text
age
balance
day
duration
campaign
pdays
previous
```

## Categorical Features

```text
job
marital
education
default
housing
loan
contact
month
poutcome
```

## Target

```text
y
```

---

# 🔍 Exploratory Data Analysis

The following aspects are analyzed:

- First few records
- Last few records
- Dataset shape
- Column names
- Data types
- Number of unique values
- Missing values
- Descriptive statistics
- Target distribution

For numerical features, the project calculates:

- Mean
- Median
- Mode
- Minimum
- Maximum
- Variance
- Standard Deviation
- Range

Selected statistical calculations are implemented manually and then verified using Pandas/NumPy.

---

# 🧹 Data Preprocessing

## 1. Missing Value Analysis

The dataset contains:

```text
Missing Values = 0
```

There are no literal `NaN` values in the dataset.

Some categorical variables contain the value `unknown`. These values are retained as explicit categories instead of automatically converting them into missing values.

### Decision

No missing-value imputation is required because there are zero literal missing values.

---

## 2. Duplicate Detection

Duplicate observations are checked before further processing.

### Result

```text
Original Records      : 45,211
Duplicate Records     : 0
Records After Treatment: 45,211
```

Since there are no duplicate records, no duplicate rows are removed.

---

## 3. Invalid and Inconsistent Data

The dataset is checked for:

- Impossible age values
- Invalid numerical values
- Negative values where they are not meaningful
- Invalid campaign values
- Invalid day values
- Negative duration
- Negative previous-contact counts
- Special values such as `pdays = -1`
- Inconsistent categorical values

Domain-specific values are not blindly removed.

For example:

```text
pdays = -1
```

is treated as a special meaningful value rather than automatically considering it an error.

Similarly, a negative bank balance can be a valid real-world observation.

---

# 🔤 Categorical Data Encoding

## Label Encoding

Label Encoding converts categorical values into numerical values.

Example:

```text
Category    Encoded
-------------------
A              0
B              1
C              2
```

The mapping is created manually using a Python dictionary.

### Advantages

- Simple representation
- Useful for ordinal variables
- Convenient for binary categories

### Limitation

For nominal categories, assigning numbers may create an artificial ordering.

---

## One-Hot Encoding

One-Hot Encoding converts each category into a separate binary column.

Example:

```text
marital    marital_married    marital_single

married           1                  0
single            0                  1
```

One-Hot Encoding is useful for nominal categorical variables because it does not create an artificial numerical ordering.

Both Label Encoding and One-Hot Encoding are implemented from scratch.

---

# 📈 Outlier Detection

Two different methods are used for detecting potential outliers.

## IQR Method

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

The project calculates:

- Q1
- Q3
- IQR
- Lower Bound
- Upper Bound
- Number of Outliers

---

## Z-Score Method

```text
Z = (X - μ) / σ
```

where:

```text
μ = Mean
σ = Standard Deviation
```

Potential extreme observations are identified using:

```text
|Z| > 3
```

---

## Outlier Treatment

An observation is not removed simply because it is mathematically identified as an outlier.

For example:

- Extremely high bank balance can be a valid observation.
- Long call duration can be a valid observation.
- Campaign-related extreme values can contain useful information.
- Special codes such as `pdays = -1` may have domain meaning.

Therefore, outliers are analyzed using both statistical methods and domain understanding.

The decision in this project is to **retain valid extreme observations** rather than blindly deleting them.

---

# 🔄 Data Transformation

The project demonstrates logarithmic transformation using:

```text
X' = log(1 + X)
```

The implementation uses:

```python
np.log1p(X)
```

The `duration` feature is used for demonstration because it is non-negative and strongly right-skewed.

The distribution is visualized before and after transformation.

---

# ⚖️ Feature Scaling

## Min-Max Normalization

Formula:

```text
X' = (X - Xmin) / (Xmax - Xmin)
```

The transformed values generally lie between:

```text
0 and 1
```

The calculation is implemented from scratch.

---

## Standardization

Formula:

```text
Z = (X - μ) / σ
```

where:

```text
μ = Mean
σ = Standard Deviation
```

The mean and standard deviation are calculated manually.

---

# 🔀 Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The train-test splitting logic is implemented manually using:

- Index creation
- Randomization
- Shuffling
- Training/test index separation

The ready-made `train_test_split()` function is not used for the primary implementation.

---

# 🚨 Data Leakage Prevention

Data Leakage occurs when information from the test dataset influences training or preprocessing.

## ❌ Incorrect Approach

```text
Complete Dataset
       ↓
Calculate Mean / Standard Deviation
       ↓
Standardize Dataset
       ↓
Train-Test Split
```

This allows information from the test set to influence preprocessing.

## ✅ Correct Approach

```text
Raw Dataset
     ↓
Train-Test Split
     ↓
Learn Parameters from Training Data
     ↓
Transform Training Data
     ↓
Transform Test Data using Training Parameters
```

This principle is applied to:

- Missing-value parameters
- Encoding mappings
- Normalization
- Standardization
- Supervised Feature Selection

---

# 🔬 Feature Selection

Feature Selection means selecting useful features from the existing features.

It is different from Feature Extraction.

The following feature-selection techniques are implemented:

1. Variance Threshold
2. Pearson Correlation
3. Chi-Square
4. ANOVA F-Test
5. Mutual Information

---

## 1️⃣ Variance Threshold

Formula:

```text
Variance(X) = Σ(xi - x̄)² / n
```

The method identifies:

- Zero-variance features
- Near-zero-variance features
- Low-variance features

A feature with zero variance has the same value for every observation and therefore provides very little discriminatory information.

---

## 2️⃣ Pearson Correlation

Pearson Correlation measures the linear relationship between two numerical variables.

Formula:

```text
              Σ(xi - x̄)(yi - ȳ)
r = -------------------------------------------
    √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

It is used to analyze:

- Numerical Feature → Target relationship
- Numerical Feature → Numerical Feature relationship
- Potential redundant features

A low Pearson correlation does not necessarily mean that no relationship exists because the relationship may be nonlinear.

---

## 3️⃣ Chi-Square Feature Selection

Chi-Square is used for categorical/discrete feature and categorical target relationships.

Formula:

```text
χ² = Σ (O - E)² / E
```

where:

```text
O = Observed Frequency
E = Expected Frequency
```

Expected frequency:

```text
E = (Row Total × Column Total) / Grand Total
```

Degrees of freedom:

```text
df = (r - 1)(c - 1)
```

The project manually calculates:

1. Contingency table
2. Observed frequencies
3. Expected frequencies
4. Chi-Square contribution
5. Total Chi-Square
6. Degrees of freedom
7. Interpretation

The manual result is verified using SciPy.

---

## 4️⃣ ANOVA F-Test

ANOVA is used to study the relationship between a numerical feature and a categorical target.

The project calculates:

- Overall mean
- Group means
- Between-group variation
- Within-group variation
- Degrees of freedom
- F-statistic

A relatively high F-statistic indicates that the group means differ substantially relative to the within-group variation.

### ANOVA Assumptions

The classical ANOVA test assumes:

- Independent observations
- Approximately normally distributed observations within groups
- Reasonably homogeneous group variances

---

## 5️⃣ Mutual Information

Mutual Information measures the amount of information shared between two variables.

### Entropy

```text
H(Y) = -Σ P(y) log₂ P(y)
```

### Mutual Information

```text
MI(X;Y) = H(Y) - H(Y|X)
```

MI can capture more general statistical dependence, including nonlinear relationships.

For the from-scratch implementation, suitable discrete/categorical variables are used.

MI is measured in **bits** because the logarithm uses base 2.

---

# 📊 Feature Selection Method Comparison

| Method | Feature Type | Target Type | Main Purpose |
|---|---|---|---|
| Variance Threshold | Numerical | Not Required | Detect low-variance features |
| Pearson Correlation | Numerical | Numerical/Binary | Measure linear relationship |
| Chi-Square | Categorical/Discrete | Categorical | Measure statistical dependence |
| ANOVA F-Test | Numerical | Categorical | Compare target groups |
| Mutual Information | Discrete/Categorical | Discrete/Categorical | Measure shared information |

Different techniques are selected according to the data type and relationship being investigated.

---

# ⭐ Final Feature Selection

Based on the implemented feature-selection strategy, the final selected features are:

```text
duration
pdays
previous
campaign
balance
poutcome
month
contact
housing
job
```

## Selected Features

**10 Features**

## Removed Features

```text
age
marital
education
default
loan
day
```

### Feature Count

```text
Original Features : 16
Selected Features : 10
Removed Features  : 6
```

The final decision considers:

- Variance
- Feature-target relationship
- Statistical dependence
- Feature redundancy
- Data type
- Interpretability

---

# 📊 Before vs After Feature Selection

| Stage | Number of Features |
|---|---:|
| Original Dataset | 16 |
| After Preprocessing | 16 |
| After Feature Selection | 10 |

Therefore:

```text
16 → 10
```

Feature selection removes **6 features** from the original input feature set.

---

# 📉 Data Visualization

The project contains meaningful visualizations including:

- Age Distribution Histogram
- Balance Box Plot
- Target Distribution Bar Chart
- Duration vs Target Scatter Plot
- Numerical Feature Correlation Heatmap
- Before/After Transformation Histogram

Each graph contains:

- Appropriate title
- X-axis label
- Y-axis label
- Interpretation

---

# 🤖 Model-Based Validation

A Logistic Regression model is used for practical validation.

## Model A — All Features

The model is trained using all suitable preprocessed features.

## Model B — Selected Features

The same model is trained using only the selected features.

The models are compared using:

- Number of Features
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Model Complexity

The objective is not simply to maximize accuracy.

Feature selection is also evaluated based on:

- Reduced dimensionality
- Reduced redundancy
- Improved interpretability
- Reduced complexity
- Maintained or improved performance

---

# 🧠 From-Scratch Implementations

The following techniques are implemented manually:

```text
Mean
Median
Mode
Variance
Standard Deviation
Label Encoding
One-Hot Encoding
IQR Outlier Detection
Z-Score Outlier Detection
Log Transformation
Min-Max Normalization
Standardization
Train-Test Split
Variance Threshold
Pearson Correlation
Chi-Square
ANOVA F-Test
Mutual Information
```

After the manual implementation, appropriate library functions are used for verification.

---

# 🔎 Library Verification

The project follows:

```text
Step 1 → Understand Formula
Step 2 → Implement From Scratch
Step 3 → Calculate Result
Step 4 → Apply Library Function
Step 5 → Compare Results
Step 6 → Interpret
```

Libraries used for verification include:

- Pandas
- NumPy
- SciPy
- Scikit-learn

---

# 📌 Key Findings

1. The dataset contains **45,211 observations**.
2. The dataset contains **17 columns**.
3. There are **16 input features**.
4. There are **7 numerical features**.
5. There are **9 categorical features**.
6. The target variable is `y`.
7. The problem is a Binary Classification problem.
8. There are **0 literal missing values**.
9. There are **0 duplicate records**.
10. Several numerical variables contain extreme observations.
11. Extreme observations are not automatically removed because they may be valid real-world values.
12. `duration` shows a strong numerical relationship with the target.
13. Categorical variables such as `poutcome`, `month`, `contact`, `housing`, and `job` provide useful target-related information.
14. No numerical feature has zero/near-zero variance requiring removal.
15. IQR and Z-score can identify different outliers because they use different statistical principles.
16. Train-Test Split is performed before learned preprocessing.
17. Data Leakage is avoided by learning preprocessing parameters only from training data.
18. Feature selection reduces the feature space from **16 to 10 features**.
19. Manual calculations are verified using appropriate libraries.
20. The final selected feature set is more focused and interpretable for subsequent Machine Learning.

---

# 📁 Project Structure

```text
ML-Preprocessing-Feature-Selection/
│
├── README.md
│
├── dataset/
│   └── bank-full.csv
│
├── notebooks/
│   └── main_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── feature_selection.py
│
├── results/
│   ├── graphs/
│   └── outputs/
│
└── report/
    └── final_summary.pdf
```

---

# 👥 Team Members

| Student | Roll Number | Contribution |
|---|---|---|
| Student 1 | TODO | TODO |
| Student 2 | TODO | TODO |
| Student 3 | TODO | TODO |
| Student 4 | TODO | TODO |

---

# 📚 Dataset Source

**Dataset:** Bank Marketing Dataset

**Dataset Source:** Add the original dataset source/link here.

---

# 🔗 Google Colab

**Google Colab Link:** Add your Colab link here.

---

# 🔗 GitHub Repository

**GitHub Repository:** Add your GitHub repository link here.

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Jupyter Notebook
- Google Colab
- Git
- GitHub

---

# 📦 Installation

```bash
pip install pandas numpy matplotlib scipy scikit-learn jupyter
```

---

# ▶️ How to Run

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Enter the Project

```bash
cd ML-Preprocessing-Feature-Selection
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn jupyter
```

### Open Notebook

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

Make sure the dataset is available at:

```text
dataset/bank-full.csv
```

Then run all notebook cells from beginning to end.

---

# 🌿 GitHub Workflow

```bash
git clone <repository-url>

git status

git add .

git commit -m "Add bank marketing preprocessing analysis"

git push origin main
```

---

# 🎓 Viva Preparation

Every group member should be prepared to explain:

- Why preprocessing is required
- Why this dataset was selected
- Missing-value treatment
- Mean vs Median
- Label Encoding
- One-Hot Encoding
- IQR
- Z-score
- Outlier treatment
- Normalization
- Standardization
- Train-Test Split
- Data Leakage
- Feature Selection
- Feature Selection vs Feature Extraction
- Variance Threshold
- Pearson Correlation
- Chi-Square
- ANOVA F-Test
- Mutual Information
- Nonlinear relationships
- Why MI uses bits
- Why preprocessing parameters should be learned from training data
- Final selected features
- From-scratch preprocessing implementation
- From-scratch feature-selection implementation

---

# 🏁 Conclusion

This project demonstrates a complete Machine Learning preprocessing and feature-selection workflow using the Bank Marketing Dataset.

The project focuses on understanding:

- Why preprocessing is required
- How preprocessing techniques work
- How the mathematical formulas are applied
- How techniques can be implemented from scratch
- How manual implementations can be verified
- How appropriate feature-selection methods are selected
- How data leakage can be avoided
- How feature selection affects the final ML-ready dataset

The core philosophy of this project is:

> **Understand → Calculate → Code → Verify → Interpret**

The final goal is not simply to produce a clean dataset, but to demonstrate a clear understanding of the complete Machine Learning preprocessing and feature-selection pipeline.
