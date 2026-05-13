# Instagram Content Performance — Predictive Analytics

End-to-end supervised machine learning pipeline on 29,999 Instagram posts. Uncovers which content formats, categories, and posting strategies drive the highest engagement — giving brands a data-backed content playbook.

**Live App →** [Streamlit Dashboard](https://share.streamlit.io/nativenerd1004/instagram-content-performance-analysis)  
**Notebook →** [`Instagram_Content_Performance_Model.ipynb`](Instagram_Content_Performance_Model.ipynb)

---

## The Problem

Brands spend time and budget creating Instagram content without knowing what will actually perform. This project answers: *What content parameters predict high engagement?*

---

## Dataset

| Property | Value |
|---|---|
| Records | 29,999 Instagram posts |
| Features | 22 columns (raw + engineered) |
| Source | Instagram Analytics Case Study |

**Key columns used as features:**

| Feature | Type | Description |
|---|---|---|
| `media_type` | Categorical | Reel, Photo, Video, Carousel |
| `content_category` | Categorical | Beauty, Fitness, Fashion, Tech, Food, etc. |
| `caption_length` | Numeric | Character count of post caption |
| `hashtags_count` | Numeric | Number of hashtags used |
| `traffic_source` | Categorical | Explore, Reels Feed, Hashtags, Home Feed, etc. |
| `Day name` | Categorical | Day of the week post was published |
| `Hour` | Numeric | Hour of day post was published |
| `Month` | Numeric | Month of year |

**Target variable:** `high_engagement` — binary label (1 if engagement rate ≥ median, 0 otherwise)

---

## Project Workflow

```
Raw CSV (29,999 rows)
    ↓
Feature Engineering  →  Label encoding · Temporal extraction · MinMaxScaler
    ↓
EDA (10 visualizations)  →  Engagement by media type, category, day, hour, hashtags
    ↓
Train / Test Split (80/20 stratified)
    ↓
9 Classification Models  →  Evaluated on Accuracy · Precision · Recall · F1 · AUC-ROC
    ↓
Hyperparameter Tuning  →  RandomizedSearchCV (3-fold CV, 10 iterations)
    ↓
3 Regression Models  →  Predict continuous engagement rate (RMSE · MAE · R²)
    ↓
Business Insights Dashboard  →  Actionable content strategy recommendations
```

---

## Models Tested

### Classification (9 models)

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.5018 | **0.5036** | 0.5007 |
| K-Nearest Neighbors | 0.5032 | 0.4965 | 0.5031 |
| Extra Trees | 0.5003 | 0.4949 | 0.5019 |
| Logistic Regression | 0.4988 | 0.4986 | 0.4926 |
| Support Vector Machine | 0.4957 | 0.4945 | 0.4941 |
| Naive Bayes | 0.4957 | 0.4925 | 0.4931 |
| AdaBoost | 0.4922 | 0.4923 | 0.4945 |
| Decision Tree | 0.4980 | 0.4921 | 0.4980 |
| Random Forest | 0.4982 | 0.4863 | 0.5015 |

### Regression (3 models)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 49.73 | 11.54 | -0.001 |
| Gradient Boosting Reg. | 49.77 | 11.60 | -0.003 |
| Random Forest Reg. | 50.10 | 12.86 | -0.016 |

### Why ~50% accuracy?

All models achieved near-random performance (~50% accuracy, R² ≈ 0). This is the most important finding:

> **The dataset is synthetically generated.** Engagement values (likes, reach, impressions) were randomly assigned and do not correlate with post metadata features. In real-world Instagram data, post type, content category, and posting time create genuine engagement signals — and this same pipeline would produce meaningful predictive accuracy.

This is a common pattern in public synthetic datasets used for learning. The ML pipeline is production-ready; the limitation is the data, not the methodology.

---

## Key EDA Findings (Content Strategy Insights)

Even with synthetic engagement values, population-level averages surface useful patterns:

| Insight | Finding |
|---|---|
| 🎬 Best media type | **Video** (14.62% avg engagement rate) |
| 📌 Best content category | **Beauty** (15.66% avg ER) |
| 📅 Best day to post | **Wednesday** (15.11% avg ER) |
| ⏰ Best posting hour | **09:00** |
| #️⃣ Hashtag strategy | **16–20 hashtags** per post |
| 📊 Caption length | **150–300 characters** optimal |

---

## Visualizations (19 charts)

| # | Chart |
|---|---|
| 01 | Target distribution — High vs Low Engagement |
| 02 | Engagement rate by media type |
| 03 | Engagement rate by content category |
| 04 | Engagement rate by day of week |
| 05 | Engagement rate by hour of day |
| 06 | Followers gained by content category |
| 07 | Correlation heatmap — all metrics |
| 08 | Hashtag count & caption length vs engagement |
| 09 | Traffic source analysis |
| 10 | Monthly engagement trend |
| 11 | Model performance comparison (9 classifiers) |
| 12 | ROC curves — all 9 models |
| 13 | Confusion matrices — top 4 models |
| 14 | Model ranking by F1-score |
| 15 | Hyperparameter optimization effect |
| 16 | Feature importance |
| 17 | Business insights dashboard |
| 18 | Regression models — RMSE & R² comparison |
| 19 | Predicted vs actual engagement rate |

---

## Repository Structure

```
├── Instagram_Content_Performance_Model.ipynb   ← Full analysis notebook
├── streamlit_app.py                            ← Interactive web app
├── instagram_analysis.py                       ← Core analysis script
├── model_summary.json                          ← Structured results
├── requirements.txt                            ← Python dependencies
├── visualizations/                             ← All 19 chart images
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/Nativenerd1004/instagram-content-performance-analysis
cd instagram-content-performance-analysis
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py

# Or run the full analysis
python instagram_analysis.py
```

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Streamlit`

---

**Author:** Samuel Madumere ([@Nativenerd1004](https://github.com/Nativenerd1004))  
**Date:** May 2026  
**License:** MIT
