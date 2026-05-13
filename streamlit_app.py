"""
Instagram Content Performance — Interactive Analytics Dashboard
Author: Samuel Madumere
"""
import streamlit as st
import json
import os
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Analytics | Samuel Madumere",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Force dark background everywhere ── */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
  [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background-color: #0D0D0D !important;
    color: #E0E0E0 !important;
  }
  /* Sidebar */
  [data-testid="stSidebarContent"] { background-color: #1A1A2E !important; }
  /* Cards / metric blocks */
  [data-testid="metric-container"] {
    background: #16213E; border: 1px solid #2A2A4A; border-radius: 10px; padding: 12px;
  }
  [data-testid="stMetricLabel"]  > div { color: #888 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
  [data-testid="stMetricValue"]  > div { color: #00D4FF !important; font-size: 1.5rem !important; font-weight: 700 !important; }
  /* Headers */
  h1 { color: #E0E0E0 !important; font-weight: 700 !important; }
  h2 { color: #00D4FF !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-top: 1.2rem !important; }
  h3 { color: #FF3CAC !important; font-weight: 600 !important; font-size: 0.95rem !important; }
  /* Dividers */
  hr { border-color: #2A2A4A !important; }
  /* Selectbox / slider */
  .stSelectbox > div > div, .stSlider { background: #1A1A2E !important; border-color: #2A2A4A !important; }
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #1A1A2E; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: #888 !important; }
  .stTabs [aria-selected="true"] { color: #00D4FF !important; background: rgba(0,212,255,0.1) !important; border-radius: 6px !important; }
  /* Info / warning boxes */
  .stAlert { background: #16213E !important; border: 1px solid #2A2A4A !important; }
  /* Images */
  img { border-radius: 8px; }
  /* Tables */
  .stDataFrame { background: #16213E !important; }
  /* Captions */
  .stCaption { color: #888 !important; }
  /* Progress bar */
  .stProgress > div > div { background: #00D4FF !important; }
  /* Buttons */
  .stButton button { background: rgba(0,212,255,0.1) !important; color: #00D4FF !important; border: 1px solid #00D4FF !important; border-radius: 8px !important; }
  .stButton button:hover { background: rgba(0,212,255,0.2) !important; }
  /* Footer hide */
  footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_summary():
    p = Path(__file__).parent / "model_summary.json"
    with open(p) as f:
        return json.load(f)

DATA = load_summary()
INS  = DATA["insights"]
CAT_ENG   = DATA["category_avg_engagement"]
MEDIA_ENG = DATA["media_avg_engagement"]
DAY_ENG   = DATA["day_avg_engagement"]
CLS_SCORES = DATA["classifier_scores"]
REG_SCORES = DATA["regression_scores"]

VIZ_DIR = Path(__file__).parent / "visualizations"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 InstaML Analytics")
    st.markdown("**Samuel Madumere**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🎨 Visualizations", "🤖 ML Models", "🔮 Content Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** 29,999 posts")
    st.markdown(f"**Models:** 9 classifiers + 3 regressors")
    st.markdown(f"**Best ER:** {INS['best_engagement_rate']:.2f}%")
    st.markdown("---")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Nativenerd1004/instagram-content-performance-analysis)")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("📊 Instagram Performance Dashboard")
    st.caption("29,999 posts · 9 ML classifiers · 3 regression models · May 2026")

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Posts Analysed",  "29,999")
    c2.metric("Best Media Type", INS["best_media_type"])
    c3.metric("Best Category",   INS["best_category"])
    c4.metric("Best Day",        INS["best_day"])
    c5.metric("Median Eng. Rate", f"{INS['median_engagement_rate']:.2f}%")
    c6.metric("Best Hashtags",   f"{INS['best_hashtag_range']} tags")

    st.divider()

    # Two columns: charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📌 Avg Engagement by Content Category")
        import pandas as pd
        cat_df = pd.DataFrame.from_dict(CAT_ENG, orient="index", columns=["Avg ER (%)"]) \
                   .sort_values("Avg ER (%)", ascending=True)
        st.bar_chart(cat_df, color="#7B2FBE", height=320)

    with col_b:
        st.markdown("### 🎬 Avg Engagement by Media Type")
        media_df = pd.DataFrame.from_dict(MEDIA_ENG, orient="index", columns=["Avg ER (%)"]) \
                     .sort_values("Avg ER (%)", ascending=True)
        st.bar_chart(media_df, color="#00D4FF", height=320)

    # Day of week
    st.markdown("### 📅 Avg Engagement by Day of Week")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_df = pd.DataFrame({"Day": day_order, "Avg ER (%)": [DAY_ENG.get(d, 0) for d in day_order]}).set_index("Day")
    st.bar_chart(day_df, color="#FF3CAC", height=220)

    st.divider()

    # Strategy tips
    st.markdown("### 💡 Content Strategy Recommendations")
    t1,t2,t3 = st.columns(3)
    with t1:
        st.info(f"**🎬 Media Format**\n\nPost **{INS['best_media_type']}s** for the highest average engagement rate ({INS['best_engagement_rate']:.2f}%).")
    with t2:
        st.info(f"**📌 Content Niche**\n\n**{INS['best_category']}** content leads in engagement across all categories in this dataset.")
    with t3:
        st.info(f"**📅 Timing**\n\nPublish on **{INS['best_day']}** at **{INS['best_hour']:02d}:00** for peak audience activity.")

    t4,t5,_ = st.columns(3)
    with t4:
        st.info(f"**#️⃣ Hashtag Strategy**\n\nUse **{INS['best_hashtag_range']} hashtags** per post for optimal discoverability.")
    with t5:
        st.info("**✍️ Caption Length**\n\nAim for **150–300 characters** — the sweet spot for engagement in this dataset.")

    st.divider()
    st.warning(
        "**⚠️ Note on ML Performance**  \n"
        "All 9 classifiers achieved ~50% accuracy and regression R² ≈ 0. "
        "This is expected — the dataset is **synthetically generated**, meaning engagement values were "
        "randomly assigned and do not correlate with post metadata. "
        "The ML pipeline is production-ready and would perform significantly better on "
        "**real account-level Instagram data** where genuine engagement signals exist."
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Visualizations":
    st.title("🎨 Analysis Visualizations")
    st.caption("19 charts across EDA, ML evaluation, and regression analysis. Click to expand any image.")

    CHARTS = [
        ("01_target_distribution.png",      "Target Distribution — High vs Low Engagement"),
        ("02_engagement_by_media_type.png",  "Engagement Rate by Media Type"),
        ("03_engagement_by_category.png",    "Engagement Rate by Content Category"),
        ("04_engagement_by_day.png",         "Engagement Rate by Day of Week"),
        ("05_engagement_by_hour.png",        "Engagement Rate by Hour of Day"),
        ("06_followers_by_category.png",     "Followers Gained by Content Category"),
        ("07_correlation_heatmap.png",       "Correlation Heatmap — All Metrics"),
        ("08_hashtag_caption_analysis.png",  "Hashtag Count & Caption Length vs Engagement"),
        ("09_traffic_source_analysis.png",   "Traffic Source Analysis"),
        ("10_monthly_engagement_trend.png",  "Monthly Engagement Trend"),
        ("11_model_comparison.png",          "ML Model Performance Comparison"),
        ("12_roc_curves.png",                "ROC Curves — All 9 Models"),
        ("13_confusion_matrices.png",        "Confusion Matrices — Top 4 Models"),
        ("14_f1_ranking.png",                "Model Ranking by F1-Score"),
        ("15_optimization_comparison.png",   "Hyperparameter Optimization Effect"),
        ("16_feature_importance.png",        "Feature Importance"),
        ("17_business_insights_dashboard.png","Business Insights Dashboard"),
        ("18_regression_comparison.png",     "Regression Models — RMSE & R²"),
        ("19_predicted_vs_actual.png",       "Predicted vs Actual Engagement Rate"),
    ]

    filter_opt = st.radio("Filter", ["All (19)", "📊 EDA (10)", "🤖 ML Results (7)", "📈 Regression (2)"], horizontal=True)
    if filter_opt == "📊 EDA (10)":        charts = CHARTS[:10]
    elif filter_opt == "🤖 ML Results (7)": charts = CHARTS[10:17]
    elif filter_opt == "📈 Regression (2)": charts = CHARTS[17:]
    else:                                   charts = CHARTS

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(charts):
                fname, title = charts[i+j]
                img_path = VIZ_DIR / fname
                if img_path.exists():
                    with col:
                        st.image(str(img_path), caption=title, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    st.caption("9 supervised classifiers + 3 regression models evaluated on 6,000 test samples (20% holdout)")

    import pandas as pd

    # Metrics summary
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Classifier", DATA["best_classifier"])
    c2.metric("Best F1-Score", f"{CLS_SCORES[DATA['best_classifier']]['f1']:.4f}")
    c3.metric("Best AUC-ROC",  f"{CLS_SCORES[DATA['best_classifier']]['roc_auc']:.4f}")
    c4.metric("Train/Test Split", "80 / 20")

    st.divider()

    # Classifier table
    st.markdown("## 📊 Classification Results — 9 Models")
    cls_df = pd.DataFrame(CLS_SCORES).T.rename(columns={
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-Score", "roc_auc": "AUC-ROC"
    }).sort_values("F1-Score", ascending=False)
    cls_df = cls_df.applymap(lambda v: f"{v:.4f}")
    st.dataframe(cls_df, use_container_width=True, height=360)

    st.divider()

    # F1 bar chart
    st.markdown("## 📊 F1-Score Comparison")
    f1_df = pd.DataFrame({n: s["f1"] for n,s in CLS_SCORES.items()}, index=["F1"]).T \
              .sort_values("F1", ascending=True)
    st.bar_chart(f1_df, color="#FF3CAC", height=300)

    st.divider()

    # Regression table
    st.markdown("## 📈 Regression Results — Predicting Engagement Rate")
    reg_df = pd.DataFrame(REG_SCORES).T.rename(columns={
        "rmse": "RMSE ↓", "mae": "MAE ↓", "r2": "R² ↑"
    }).applymap(lambda v: f"{v:.4f}")
    st.dataframe(reg_df, use_container_width=True)

    st.divider()

    # Methodology
    st.markdown("## 🔬 Methodology")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Target Variable**")
        st.code("high_engagement = 1 if engagement_rate >= median else 0", language="python")
        st.markdown("**Feature Scaler**")
        st.code("MinMaxScaler (applied after 80/20 split)", language="python")
    with m2:
        st.markdown("**Features Used**")
        st.code("\n".join(DATA["feature_cols"]), language="text")
        st.markdown("**Optimization**")
        st.code("RandomizedSearchCV(n_iter=10, cv=3, scoring='f1')", language="python")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CONTENT PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Content Predictor":
    st.title("🔮 Content Score Predictor")
    st.caption("Configure your post parameters and get an EDA-driven content performance score with actionable recommendations.")

    col_form, col_score = st.columns([1, 1])

    with col_form:
        st.markdown("## ⚙️ Post Configuration")
        media_type       = st.selectbox("Media Type",       ["Reel","Video","Photo","Carousel"])
        content_category = st.selectbox("Content Category", ["Beauty","Photography","Lifestyle","Fashion","Comedy","Music","Fitness","Food","Technology","Travel"])
        traffic_source   = st.selectbox("Traffic Source",   ["Explore","Reels Feed","Hashtags","Home Feed","Profile","External"])
        post_day         = st.selectbox("Day to Post",      ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        post_hour        = st.slider("Posting Hour", 0, 23, 9, format="%d:00")
        hashtag_count    = st.slider("Number of Hashtags", 0, 30, 18)
        caption_length   = st.slider("Caption Length (characters)", 10, 2200, 150, step=10)

    # Score calculation (EDA-derived segment averages)
    m_eng  = MEDIA_ENG.get(media_type, 14.0)
    c_eng  = CAT_ENG.get(content_category, 14.0)
    d_eng  = DAY_ENG.get(post_day, 14.0)
    m_max  = max(MEDIA_ENG.values())
    c_max  = max(CAT_ENG.values())
    d_max  = max(DAY_ENG.values())

    media_score    = (m_eng / m_max) * 25
    cat_score      = (c_eng / c_max) * 25
    day_score      = (d_eng / d_max) * 20
    hash_score     = (1.0 if 16<=hashtag_count<=20 else 0.7 if 11<=hashtag_count<=25 else 0.4) * 15
    caption_score  = (1.0 if 150<=caption_length<=300 else 0.7 if 50<=caption_length<=500 else 0.4) * 10
    hour_score     = (1.0 if 8<=post_hour<=11 else 0.6 if 7<=post_hour<=18 else 0.3) * 5
    total_score    = round(media_score + cat_score + day_score + hash_score + caption_score + hour_score)

    tier  = ("🟢 High Performer" if total_score>=80
             else "🔵 Good Post"  if total_score>=60
             else "🟡 Average"    if total_score>=40
             else "🔴 Low Performer")

    with col_score:
        st.markdown("## 📈 Performance Score")

        st.markdown(f"""
        <div style="background:#16213E;border:2px solid #2A2A4A;border-radius:16px;padding:2rem;text-align:center;">
            <div style="font-size:5rem;font-weight:800;color:{'#00FF88' if total_score>=80 else '#00D4FF' if total_score>=60 else '#FF6B35' if total_score>=40 else '#FF3CAC'};line-height:1;">
                {total_score}
            </div>
            <div style="font-size:1rem;color:#888;margin-bottom:0.5rem;">out of 100</div>
            <div style="font-size:1.2rem;font-weight:700;color:#E0E0E0;">{tier}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Score Breakdown**")
        breakdown = [
            ("Media Type",     media_score,   25, f"{media_type} → {m_eng:.2f}% avg ER"),
            ("Category",       cat_score,     25, f"{content_category} → {c_eng:.2f}% avg ER"),
            ("Day of Week",    day_score,     20, f"{post_day} → {d_eng:.2f}% avg ER"),
            ("Hashtag Count",  hash_score,    15, f"{hashtag_count} tags"),
            ("Caption Length", caption_score, 10, f"{caption_length} chars"),
            ("Posting Hour",   hour_score,     5, f"{post_hour:02d}:00"),
        ]
        for label, pts, max_pts, detail in breakdown:
            st.markdown(f"**{label}** `{pts:.1f}/{max_pts}`  ·  {detail}")
            st.progress(int(pts / max_pts * 100))

    st.divider()

    # Recommendations
    tips = []
    if media_type != INS["best_media_type"]:
        tips.append(f"Switch to **{INS['best_media_type']}** — the highest-performing media type ({m_max:.2f}% avg ER)")
    if content_category != INS["best_category"]:
        tips.append(f"**{INS['best_category']}** content drives the highest engagement in this dataset")
    if post_day != INS["best_day"]:
        tips.append(f"Publish on **{INS['best_day']}** for peak engagement ({DAY_ENG.get(INS['best_day'],0):.2f}% avg ER)")
    if not (16 <= hashtag_count <= 20):
        tips.append(f"Use **{INS['best_hashtag_range']} hashtags** for optimal reach")
    if not (150 <= caption_length <= 300):
        tips.append("Write captions of **150–300 characters** — the sweet spot for engagement")
    if tips:
        st.markdown("## 💡 Recommendations to Improve Your Score")
        for tip in tips:
            st.markdown(f"→ {tip}")
    else:
        st.success("✅ Your configuration is already optimised based on this dataset's patterns!")

    st.divider()
    st.markdown("## 🏆 Optimal Post Configuration (EDA Reference)")
    o1,o2,o3,o4,o5,o6 = st.columns(6)
    o1.metric("Media Type",   INS["best_media_type"])
    o2.metric("Category",     INS["best_category"])
    o3.metric("Day",          INS["best_day"])
    o4.metric("Hour",         f"{INS['best_hour']:02d}:00")
    o5.metric("Hashtags",     INS["best_hashtag_range"])
    o6.metric("Caption",      "150-300 chars")
