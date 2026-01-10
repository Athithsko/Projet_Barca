# AI Tool Usage Disclosure

This document describes how AI tools were used during the development of this FC Barcelona 2024-25 La Liga analysis project, in compliance with the HEC Lausanne Advanced Programming course AI Tools Policy.

---

## Tools Used

### 1. DeepSeek
**Purpose:** Debugging and code optimization

**How it was used:**
- Debugging Python errors in the data loading pipeline
- Optimization suggestions for scikit-learn model configurations
- Understanding error messages and finding solutions
- Assistance with pandas DataFrame operations

**Example use case:**
- Asked for help debugging a `KeyError` when loading CSV files with European number formats (commas instead of dots)
- Received suggestions on using `decimal=','` parameter in `pd.read_csv()`

---

### 2. Claude (Anthropic)
**Purpose:** Code review, visualization design, and documentation

**How it was used:**
- Code review suggestions for improving code structure
- Algorithm verification (checking K-Means implementation correctness)
- Assistance with matplotlib/seaborn visualization design
- Help with Barcelona-themed color palettes for plots
- LaTeX report formatting assistance

**Example use case:**
- Asked for review of my `elbow_method_analysis()` function
- Received feedback on using second derivative for optimal k detection

---

### 3. Gemini Pro (Google)
**Purpose:** Technical writing and documentation

**How it was used:**
- Report structuring and organization suggestions
- Methodology validation (checking scientific soundness)
- Documentation review for clarity and completeness
- Help with academic writing style

**Example use case:**
- Asked for feedback on Discussion section structure
- Received suggestions on how to present the "Pedri Anomaly" finding

---

## Learning Moments

Through AI assistance, I deepened my understanding of:

1. **StandardScaler usage:** Why Logistic Regression needs feature scaling but tree-based models don't
2. **Clustering validation:** Difference between classification metrics (accuracy, precision) and clustering metrics (Silhouette, Davies-Bouldin)
3. **Temporal splitting:** Why time-based train/test splits are essential for match prediction models
4. **ROC-AUC interpretation:** How to evaluate model discrimination beyond raw accuracy

---

## Code Understanding

I confirm that I understand all code in this project:

- **Data preprocessing:** I understand how `load_team_data()` converts European formats and creates binary variables
- **ML pipeline:** I understand how `train_ml_models()` applies StandardScaler only to Logistic Regression and why
- **Cross-validation:** I understand why Stratified K-Fold is necessary for imbalanced classes
- **K-Means clustering:** I understand the elbow method and why I used second derivative for k selection
- **Impact Score:** I designed the weighting system myself based on football domain knowledge

---

## Declaration

I, the author of this project, confirm that:
- I understand every line of code in this repository
- AI tools were used as learning aids, not as ghostwriters
- All analytical decisions and interpretations are my own
- I can explain any part of this project during the presentation

*Last updated: January 2026*
