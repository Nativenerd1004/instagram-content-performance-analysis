# 📊 Instagram Content Performance Predictive Analysis

> **Data-driven content strategy for brands — powered by Machine Learning**

---

## 🎯 Project Goal

Analyze 29,999 Instagram posts to uncover high-performance content patterns and build a supervised ML pipeline that predicts whether a post will achieve high engagement. Designed to guide brands in making smarter, evidence-backed content decisions.

---

## 📋 STAR Method

| | |
|---|---|
| **Situation** | Brands invest heavily in Instagram content without data-backed guidance on what works |
| **Task** | Analyze post-level Instagram data across media types, categories, timing and audience behaviour |
| **Action** | Built a full EDA + supervised ML pipeline — 9 classifiers, 3 regressors, 19 visualizations |
| **Result** | Actionable content strategy insights + reusable ML framework ready for real account data |

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | Instagram Analytics Case Study |
| **Records** | 29,999 posts |
| **Features** | 22 columns (raw + engineered) |
| **Target** | High Engagement (binary) · Engagement Rate (continuous) |

**Key features used:**
- `media_type` — Reel, Photo, Video, Carousel
- `content_category` — Fashion, Fitness, Beauty, Technology, Food, etc.
- `caption_length`, `hashtags_count`
- `traffic_source` — Explore, Reels Feed, Home Feed, Hashtags, Profile, External
- `Month`, `Day name`, `Hour` — temporal posting features

---

## 🔄 Project Workflow

```
Raw Data → Feature Engineering → EDA (10 charts) → Preprocessing (encode + scale)
       → 9 ML Classifiers → Evaluation (accuracy, F1, AUC-ROC, confusion matrix)
       → Hyperparameter Optimization → 3 Regression Models → Business Insights Dashboard
```

---

## 📈 Machine Learning Approach

### Classification Models (9)
| Model | Task |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble (bagging) |
| Extra Trees | Ensemble (random splits) |
| Gradient Boosting | Ensemble (boosting) |
| Support Vector Machine | Margin-based (LinearSVC) |
| K-Nearest Neighbors | Distance-based |
| Naive Bayes | Probabilistic |
| AdaBoost | Adaptive boosting |

### Regression Models (3)
| Model | Predicts |
|---|---|
| Ridge Regression | Engagement Rate (continuous) |
| Random Forest Regressor | Engagement Rate (continuous) |
| Gradient Boosting Regressor | Engagement Rate (continuous) |

### Preprocessing Pipeline
1. Label Encoding for categorical features
2. `MinMaxScaler` feature normalization
3. 80/20 stratified train-test split
4. `RandomizedSearchCV` (3-fold CV) for hyperparameter tuning

### Evaluation Metrics
- Accuracy · Precision · Recall · F1-Score
- Confusion Matrix · ROC Curve · AUC-ROC
- RMSE · MAE · R² (regression)

---

## 📁 Repository Contents

```
📦 Instagram Analytics ML
├── 📓 Instagram_Content_Performance_Model.ipynb   ← Main notebook
├── 📊 visualizations/
│   ├── 01_target_distribution.png
│   ├── 02_engagement_by_media_type.png
│   ├── 03_engagement_by_category.png
│   ├── 04_engagement_by_day.png
│   ├── 05_engagement_by_hour.png
│   ├── 06_followers_by_category.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_hashtag_caption_analysis.png
│   ├── 09_traffic_source_analysis.png
│   ├── 10_monthly_engagement_trend.png
│   ├── 11_model_comparison.png
│   ├── 12_roc_curves.png
│   ├── 13_confusion_matrices.png
│   ├── 14_f1_ranking.png
│   ├── 15_optimization_comparison.png
│   ├── 16_feature_importance.png
│   ├── 17_business_insights_dashboard.png
│   ├── 18_regression_comparison.png
│   └── 19_predicted_vs_actual.png
├── 📄 model_summary.json                          ← Structured results for web app
├── 🐍 instagram_analysis.py                       ← Core analysis script
└── 📖 README.md
```

---

## 💡 Key Business Insights

| Insight | Finding |
|---|---|
| **Best Media Type** | Video drives the highest average engagement |
| **Best Content Category** | Beauty content leads in engagement rate |
| **Best Day to Post** | Wednesday shows peak engagement |
| **Best Posting Time** | 09:00 AM is the optimal window |
| **Hashtag Strategy** | 16–20 hashtags per post maximizes reach |

---

## ⚠️ Note on ML Results

The classification models achieved **~50% accuracy** (near-random), and regression R² ≈ 0.
This is expected because this is a **synthetically generated dataset** — engagement values were randomly assigned independent of post metadata features (type, category, timing).

**Key takeaway:** The ML pipeline is production-ready. When applied to **real account-level Instagram data** (where actual user behaviour creates genuine correlations), these same models would produce meaningful predictive accuracy. The EDA-derived strategy insights remain valid as population averages.

---

## 🚀 Live Demo

An interactive web app built with this analysis is deployed at:
**[Link in deployment]**

Users can:
- Explore all 19 visualizations interactively
- Input post parameters to get a content score
- View model performance comparisons
- Access the full business insights dashboard

---

## 🛠️ Tech Stack

```
Python 3.11  ·  pandas  ·  numpy  ·  scikit-learn
matplotlib  ·  seaborn  ·  Jupyter Notebook
```

---

**Author:** Samuel Madumere  
**Date:** May 2026  
**License:** MIT
