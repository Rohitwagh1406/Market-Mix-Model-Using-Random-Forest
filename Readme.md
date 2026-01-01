# Machine Learning–Based Marketing Mix Modeling (MMM)

## Project Overview
This project implements an **end-to-end Machine Learning–based Marketing Mix Model (MMM)** to quantify **incremental sales contribution of marketing channels** and support **data-driven budget optimization**.

Unlike traditional linear MMM approaches, this solution leverages **Random Forest modeling with SHAP-based explainability** to capture **non-linear relationships, carryover effects, lagged impact, and diminishing returns** of marketing investments.

The pipeline closely mirrors **real-world MMM frameworks used in Pharma, FMCG, and Digital Marketing organizations**, making it highly relevant for industry roles in **Data Analytics, Data Science, and Applied Machine Learning**.

---

## Business Objective
Marketing teams invest across multiple channels (Digital, Email, Display, HCP promotions) but face challenges in:
- Identifying **true incremental impact** of each channel
- Accounting for **lagged and carryover effects**
- Understanding **diminishing returns**
- Making **budget reallocation decisions**

**Key Question Answered:**
> Which marketing channels drive incremental sales, by how much, and at what saturation point?

---

## Solution Highlights
- End-to-end **Machine Learning MMM pipeline**
- **Adstock, Power (saturation), and Lag transformations**
- **Random Forest regression** for non-linear modeling
- **SHAP value decomposition** for explainability
- **Permutation-based importance (R² drop)**
- **Concave response curves** for ROI optimization
- Model validation using **R² and MAPE**

---

## End-to-End Pipeline
```
Raw Marketing Data
        ↓
Adstock + Power + Lag Transformations
        ↓
Feature Engineering & Aggregation
        ↓
Random Forest Model Training
        ↓
SHAP Contribution Decomposition
        ↓
Model Performance Evaluation
        ↓
Permutation Importance Validation
        ↓
Response Curve Generation
```

---

## Key Modeling Components

### 1. Advanced Media Transformations
- **Adstock** → captures carryover effect of media spend
- **Power Transformation** → models diminishing marginal returns
- **Lag Effect** → captures delayed impact on sales

These transformations ensure **behavioral realism**, not just statistical fit.

---

### 2. Machine Learning Model
- **Algorithm:** Random Forest Regressor
- Handles **non-linearity, interaction effects, and multicollinearity**
- More robust than traditional linear regression–based MMMs

---

### 3. Model Explainability with SHAP
- Full **SHAP value computation** at observation level
- Channel-wise **incremental sales attribution**
- Base sales isolated as **intercept**
- SHAP additivity verified against model predictions

This converts a black-box ML model into a **fully explainable business tool**.

---

### 4. Model Performance
- **R²** → aggregate prediction accuracy
- **MAPE** → business-friendly error metric
- Evaluated at **time-series aggregation level**

---

### 5. Variable Importance (Robustness Check)
- **Permutation-based R² drop method**
- Variables categorized as Very High / High / Medium / Low importance
- Confirms SHAP results using an independent approach

---

### 6. Response Curves (Investment Optimization)
- Concave (Hill-type) response curves generated per channel
- Identifies marginal ROI, saturation point, and over-investment zones

Supports **media planning and budget optimization decisions**.

---

##  Key Outputs
- Channel-wise **12-month incremental sales contribution**
- SHAP values in **long-format for granular explainability**
- Model accuracy metrics (**R², MAPE**)
- Permutation importance table
- Response curve datasets for optimization

---

## Tech Stack
- **R**
- randomForest
- fastshap
- data.table
- dplyr, tidyr
- ggplot2
- caret, iml

---

## Repository Structure
```
├── Data.csv
├── MMM_ML_Pipeline.R
├── MMX_code_output.xlsx
├── README.md
```

---

## Why This Project Stands Out
- Industry-aligned MMM implementation
- Combines **Machine Learning + Explainability**
- Strong focus on **business impact and ROI**
- Covers full lifecycle: data → model → insights → optimization
- Production-ready analytics mindset

---

## Author
**Rohit Wagh**  
Data Scientist | Machine Learning | Marketing Analytics

GitHub: https://github.com/Rohitwagh1406/Market-Mix-Model-Using-Random-Forest.git

---

## Future Enhancements
- Bayesian MMM comparison
- Time-based cross-validation
- Automated hyperparameter tuning
- Budget optimization engine
- BI dashboard integration (Power BI / Tableau)

---

NOTE: If you find this project valuable, consider starring the repository!

