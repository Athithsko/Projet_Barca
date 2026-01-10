# Project Proposal: FC Barcelona 2024-25 Season Analysis

## Project Title
**Predicting FC Barcelona Match Outcomes and Identifying Key Player Impact Using Machine Learning**

---

## Research Questions

1. **Can we predict FC Barcelona's match outcomes (Win/Not Win) based on in-game performance metrics?**
2. **Which key players have the most measurable impact on team performance, and how can we profile them using clustering?**

---

## Project Description

This project applies machine learning techniques to analyze FC Barcelona's 2024-25 La Liga season. The analysis combines two complementary approaches:

**Part 1: Supervised Learning (Match Prediction)**

I will build classification models to predict match outcomes using performance metrics such as expected goals (xG), possession percentage, shots on target, and opponent strength. The goal is to identify which factors most strongly influence whether Barcelona wins a match.

**Part 2: Unsupervised Learning (Player Profiling)**

Using K-Means clustering, I will analyze four key players (Pedri, Raphinha, Lamine Yamal, Iñigo Martínez) to identify distinct performance profiles. This addresses the question: do traditional statistics accurately capture each player's contribution to the team?

---

## Data Sources

All data will be collected from **FBref.com**, a comprehensive football statistics database:

- **Team-level data:** 38 La Liga matches with metrics including xG, xGA, possession, shots, passes, and match results
- **Player-level data:** Individual statistics for selected players including goals, assists, progressive passes, tackles, and advanced metrics

Data format: CSV files with manual collection and preprocessing to handle European number formats.

---

## Technical Approach

| Component | Method |
|-----------|--------|
| Classification | Logistic Regression, Random Forest, Gradient Boosting |
| Clustering | K-Means with Elbow Method optimization |
| Evaluation | ROC-AUC, Cross-Validation, Silhouette Score |
| Visualization | Matplotlib, Seaborn |

---

## Why This Project?

Football analytics is a growing field that combines domain expertise with data science. This project demonstrates practical ML applications while exploring whether advanced metrics like xG provide predictive value, and whether clustering can reveal insights that traditional statistics miss.
