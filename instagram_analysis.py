"""
Instagram Content Performance Predictive Model
===============================================
Goal: Predict whether an Instagram post will be high-performing (high engagement),
enabling brands to make data-driven content decisions.
"""

# ─── 1. IMPORTS ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ─── DARK THEME ───────────────────────────────────────────────────────────────
BG      = "#0D0D0D"
SURFACE = "#1A1A2E"
CARD    = "#16213E"
ACCENT1 = "#00D4FF"   # cyan
ACCENT2 = "#7B2FBE"   # purple
ACCENT3 = "#FF6B35"   # orange
ACCENT4 = "#00FF88"   # green
ACCENT5 = "#FF3CAC"   # pink
TEXT    = "#E0E0E0"
GRID    = "#2A2A4A"

PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, "#FFD700", "#FF4560", "#00E396"]

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.4,
    'axes.grid': True, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
})

def save_fig(name, tight=True):
    path = f"{VIZ_DIR}/{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ Saved {path}")

# ─── 2. LOAD DATA ─────────────────────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv("/Users/apple/Desktop/Instagram Analytics Datasets/Full Dataset/Intagram_CaseStudy_Cleaned_sql.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Quick sanity check
print("\n📊 Dataset Overview:")
print(df.describe())
print("\n🔍 Null values:")
print(df.isnull().sum())
print("\n📋 Value counts:")
for col in ['media_type', 'traffic_source', 'content_category', 'Day name']:
    print(f"\n{col}:\n{df[col].value_counts()}")

# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n⚙️  Feature engineering...")

# Target variable: High Engagement (1 = above median engagement rate, 0 = below)
median_er = df['engagement_rate_calc'].median()
df['high_engagement'] = (df['engagement_rate_calc'] >= median_er).astype(int)
print(f"  Median engagement rate: {median_er:.4f}")
print(f"  High engagement posts: {df['high_engagement'].sum()} / {len(df)}")

# Post timing features already exist: Hour, Month, Day name
# Engagement tier for EDA
df['engagement_tier'] = pd.cut(df['engagement_rate_calc'],
    bins=[0, df['engagement_rate_calc'].quantile(0.33),
             df['engagement_rate_calc'].quantile(0.67), float('inf')],
    labels=['Low', 'Medium', 'High'])

# ─── 4. EDA VISUALIZATIONS ────────────────────────────────────────────────────
print("\n🎨 Generating EDA visualizations...")

# --- 4.1 Target Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

labels = ['Low Engagement', 'High Engagement']
counts = df['high_engagement'].value_counts().sort_index()
bars = axes[0].bar(labels, counts.values, color=[ACCENT2, ACCENT1], edgecolor=BG, linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
axes[0].set_title('Post Engagement Distribution', color=TEXT, fontsize=14, fontweight='bold', pad=12)
axes[0].set_ylabel('Number of Posts', color=TEXT)

wedges, texts, autotexts = axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
    colors=[ACCENT2, ACCENT1], startangle=90, textprops={'color': TEXT},
    wedgeprops={'edgecolor': BG, 'linewidth': 2})
for at in autotexts:
    at.set_color(BG); at.set_fontweight('bold')
axes[1].set_title('Engagement Split', color=TEXT, fontsize=14, fontweight='bold', pad=12)
save_fig("01_target_distribution")

# --- 4.2 Engagement Rate by Media Type ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.patch.set_facecolor(BG)

media_eng = df.groupby('media_type')['engagement_rate_calc'].mean().sort_values(ascending=False)
colors = PALETTE[:len(media_eng)]
bars = axes[0].bar(media_eng.index, media_eng.values, color=colors, edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars, media_eng.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
axes[0].set_title('Avg Engagement Rate by Media Type', color=TEXT, fontsize=13, fontweight='bold')
axes[0].set_ylabel('Avg Engagement Rate (%)', color=TEXT)
axes[0].set_xlabel('Media Type', color=TEXT)

df.boxplot(column='engagement_rate_calc', by='media_type', ax=axes[1],
           patch_artist=True, notch=False)
axes[1].set_title('Engagement Rate Distribution by Media Type', color=TEXT, fontsize=13, fontweight='bold')
axes[1].set_xlabel('Media Type', color=TEXT)
axes[1].set_ylabel('Engagement Rate (%)', color=TEXT)
plt.suptitle('')
for patch, color in zip(axes[1].findobj(matplotlib.patches.PathPatch), PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
save_fig("02_engagement_by_media_type")

# --- 4.3 Engagement by Content Category ---
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
cat_eng = df.groupby('content_category')['engagement_rate_calc'].mean().sort_values(ascending=False)
bars = ax.bar(cat_eng.index, cat_eng.values, color=PALETTE[:len(cat_eng)], edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars, cat_eng.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.2f}%', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax.set_title('Average Engagement Rate by Content Category', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Avg Engagement Rate (%)', color=TEXT)
ax.set_xlabel('Content Category', color=TEXT)
plt.xticks(rotation=30, ha='right')
save_fig("03_engagement_by_category")

# --- 4.4 Engagement by Day of Week ---
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_eng   = df.groupby('Day name')['engagement_rate_calc'].mean().reindex(day_order)

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.plot(day_order, day_eng.values, color=ACCENT1, linewidth=2.5, marker='o',
        markersize=9, markerfacecolor=ACCENT3, markeredgecolor=BG, markeredgewidth=1.5)
ax.fill_between(day_order, day_eng.values, alpha=0.15, color=ACCENT1)
for i, (day, val) in enumerate(zip(day_order, day_eng.values)):
    ax.annotate(f'{val:.2f}%', (day, val), textcoords='offset points',
                xytext=(0, 10), ha='center', color=TEXT, fontsize=9, fontweight='bold')
ax.set_title('Average Engagement Rate by Day of Week', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Avg Engagement Rate (%)', color=TEXT)
ax.set_xlabel('Day of Week', color=TEXT)
save_fig("04_engagement_by_day")

# --- 4.5 Engagement by Hour ---
hour_eng = df.groupby('Hour')['engagement_rate_calc'].mean()

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.plot(hour_eng.index, hour_eng.values, color=ACCENT4, linewidth=2.5, marker='o',
        markersize=7, markerfacecolor=ACCENT5, markeredgecolor=BG, markeredgewidth=1.5)
ax.fill_between(hour_eng.index, hour_eng.values, alpha=0.15, color=ACCENT4)
ax.set_title('Average Engagement Rate by Hour of Day', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Avg Engagement Rate (%)', color=TEXT)
ax.set_xlabel('Hour of Day (0 = Midnight)', color=TEXT)
ax.set_xticks(range(0, 24))
save_fig("05_engagement_by_hour")

# --- 4.6 Followers Gained by Content Category ---
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
cat_fol = df.groupby('content_category')['followers_gained'].mean().sort_values(ascending=False)
bars = ax.bar(cat_fol.index, cat_fol.values, color=PALETTE[:len(cat_fol)], edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars, cat_fol.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax.set_title('Average Followers Gained by Content Category', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Avg Followers Gained', color=TEXT)
ax.set_xlabel('Content Category', color=TEXT)
plt.xticks(rotation=30, ha='right')
save_fig("06_followers_by_category")

# --- 4.7 Correlation Heatmap ---
numeric_cols = ['likes', 'comments', 'shares', 'saves', 'reach', 'impressions',
                'caption_length', 'hashtags_count', 'followers_gained',
                'engagement_rate_calc', 'saves_rates', 'share_rate', 'Hour', 'Month']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG)
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            linecolor=BG, ax=ax, annot_kws={'size': 8, 'color': TEXT},
            cbar_kws={'shrink': 0.7})
ax.set_title('Correlation Heatmap — Instagram Metrics', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_facecolor(SURFACE)
save_fig("07_correlation_heatmap")

# --- 4.8 Hashtag Count vs Engagement ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.patch.set_facecolor(BG)

hashtag_bins = pd.cut(df['hashtags_count'], bins=[0, 5, 10, 15, 20, 30], labels=['1-5','6-10','11-15','16-20','21-30'])
hash_eng = df.groupby(hashtag_bins)['engagement_rate_calc'].mean()
bars = axes[0].bar(hash_eng.index.astype(str), hash_eng.values, color=PALETTE[:5], edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars, hash_eng.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}%', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
axes[0].set_title('Engagement Rate by Hashtag Count', color=TEXT, fontsize=13, fontweight='bold')
axes[0].set_xlabel('Hashtag Count Range', color=TEXT)
axes[0].set_ylabel('Avg Engagement Rate (%)', color=TEXT)

caption_bins = pd.cut(df['caption_length'], bins=[0, 50, 150, 300, 500, 2200],
                      labels=['Micro\n<50','Short\n50-150','Medium\n150-300','Long\n300-500','Epic\n500+'])
cap_eng = df.groupby(caption_bins)['engagement_rate_calc'].mean()
bars2 = axes[1].bar(cap_eng.index.astype(str), cap_eng.values, color=PALETTE[:5], edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars2, cap_eng.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}%', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
axes[1].set_title('Engagement Rate by Caption Length', color=TEXT, fontsize=13, fontweight='bold')
axes[1].set_xlabel('Caption Length', color=TEXT)
axes[1].set_ylabel('Avg Engagement Rate (%)', color=TEXT)
save_fig("08_hashtag_caption_analysis")

# --- 4.9 Traffic Source Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.patch.set_facecolor(BG)
ts_eng  = df.groupby('traffic_source')['engagement_rate_calc'].mean().sort_values(ascending=False)
ts_cnt  = df['traffic_source'].value_counts()

bars = axes[0].barh(ts_eng.index, ts_eng.values, color=PALETTE[:len(ts_eng)], edgecolor=BG, linewidth=1.2)
for bar, val in zip(bars, ts_eng.values):
    axes[0].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}%', va='center', color=TEXT, fontsize=10, fontweight='bold')
axes[0].set_title('Avg Engagement by Traffic Source', color=TEXT, fontsize=13, fontweight='bold')
axes[0].set_xlabel('Avg Engagement Rate (%)', color=TEXT)

wedges, texts, autotexts = axes[1].pie(ts_cnt.values, labels=ts_cnt.index,
    autopct='%1.1f%%', colors=PALETTE[:len(ts_cnt)], startangle=90,
    textprops={'color': TEXT}, wedgeprops={'edgecolor': BG, 'linewidth': 2})
for at in autotexts:
    at.set_color(BG); at.set_fontweight('bold')
axes[1].set_title('Traffic Source Distribution', color=TEXT, fontsize=13, fontweight='bold')
save_fig("09_traffic_source_analysis")

# --- 4.10 Monthly Trend ---
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(BG)
month_order = list(range(1, 13))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_eng = df.groupby('Month')['engagement_rate_calc'].mean().reindex(month_order)
ax.bar(month_names[:len(month_eng)], month_eng.values, color=PALETTE[:len(month_eng)],
       edgecolor=BG, linewidth=1.2, width=0.6)
for i, val in enumerate(month_eng.values):
    ax.text(i, val + 0.03, f'{val:.2f}%', ha='center', va='bottom', color=TEXT,
            fontsize=9, fontweight='bold')
ax.plot(month_names[:len(month_eng)], month_eng.values, color=ACCENT1,
        linewidth=2, marker='D', markersize=6, markerfacecolor=ACCENT5, markeredgecolor=BG)
ax.set_title('Average Engagement Rate by Month', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Avg Engagement Rate (%)', color=TEXT)
ax.set_xlabel('Month', color=TEXT)
save_fig("10_monthly_engagement_trend")

print("\n✅ All 10 EDA visualizations saved.")

# ─── 5. PREPROCESSING ─────────────────────────────────────────────────────────
print("\n⚙️  Preprocessing for ML...")

feature_cols = ['media_type', 'caption_length', 'hashtags_count',
                'traffic_source', 'content_category', 'Hour', 'Month', 'Day name']
target_col = 'high_engagement'

df_ml = df[feature_cols + [target_col]].copy()

le = LabelEncoder()
cat_cols = ['media_type', 'traffic_source', 'content_category', 'Day name']
encoders = {}
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"  Encoded columns: {cat_cols}")
print(f"  Label mappings:")
for col, mapping in encoders.items():
    print(f"    {col}: {mapping}")

X = df_ml[feature_cols]
y = df_ml[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")
print(f"  Class balance — Train: {np.bincount(y_train)}  Test: {np.bincount(y_test)}")

# ─── 6. MODEL TRAINING ────────────────────────────────────────────────────────
print("\n🤖 Training 9 supervised ML models...")

models = {
    "Logistic Regression":     LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":           DecisionTreeClassifier(random_state=42),
    "Random Forest":           RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Extra Trees":             ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Support Vector Machine":  CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3),
    "K-Nearest Neighbors":     KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":             GaussianNB(),
    "AdaBoost":                AdaBoostClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_prob  = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc,
        "fpr":       fpr,
        "tpr":       tpr,
    }
    print(f"Acc={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}  AUC={roc_auc:.4f}")

# ─── 7. MODEL EVALUATION VISUALIZATIONS ──────────────────────────────────────
print("\n📊 Generating model evaluation visualizations...")

# --- 7.1 Model Comparison Bar Chart ---
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k in ['accuracy','precision','recall','f1','roc_auc']}
    for name, res in results.items()
}).T.sort_values('f1', ascending=False)

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG)
x = np.arange(len(metrics_df))
width = 0.17
metric_colors = [ACCENT1, ACCENT4, ACCENT3, ACCENT5, ACCENT2]
metric_names  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
for i, (met, color, label) in enumerate(zip(metric_names, metric_colors, metric_labels)):
    bars = ax.bar(x + i*width, metrics_df[met], width, label=label, color=color,
                  edgecolor=BG, linewidth=0.8, alpha=0.85)
ax.set_xticks(x + width*2)
ax.set_xticklabels(metrics_df.index, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Score', color=TEXT)
ax.set_title('Model Performance Comparison', color=TEXT, fontsize=15, fontweight='bold', pad=14)
ax.set_ylim(0, 1.12)
ax.legend(loc='upper right', framealpha=0.2, labelcolor=TEXT, facecolor=SURFACE)
ax.axhline(0.8, color=GRID, linestyle='--', linewidth=1, alpha=0.6, label='0.8 baseline')
save_fig("11_model_comparison")

# --- 7.2 ROC Curves ---
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(BG)
for (name, res), color in zip(results.items(), PALETTE):
    ax.plot(res['fpr'], res['tpr'], color=color, linewidth=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0,1],[0,1], color=GRID, linestyle='--', linewidth=1.5, label='Random Classifier')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', color=TEXT, fontsize=11)
ax.set_ylabel('True Positive Rate', color=TEXT, fontsize=11)
ax.set_title('ROC Curves — All Models', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='lower right', framealpha=0.2, labelcolor=TEXT, facecolor=SURFACE, fontsize=9)
save_fig("12_roc_curves")

# --- 7.3 Confusion Matrices (top 4 models by F1) ---
top4 = metrics_df.head(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(BG)
fig.suptitle('Confusion Matrices — Top 4 Models', color=TEXT, fontsize=15, fontweight='bold', y=1.01)
for ax, name in zip(axes.flatten(), top4):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False, linewidths=1, linecolor=BG,
                annot_kws={'size': 14, 'color': TEXT, 'fontweight': 'bold'})
    ax.set_title(f"{name}\nF1={results[name]['f1']:.4f}", color=TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', color=TEXT); ax.set_ylabel('Actual', color=TEXT)
    ax.set_facecolor(SURFACE)
    ax.set_xticklabels(['Low Eng.', 'High Eng.'], color=TEXT)
    ax.set_yticklabels(['Low Eng.', 'High Eng.'], color=TEXT, rotation=0)
save_fig("13_confusion_matrices")

# --- 7.4 F1 Score Ranking ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
sorted_f1 = metrics_df['f1'].sort_values()
colors_f1 = [ACCENT1 if i == len(sorted_f1)-1 else ACCENT2 for i in range(len(sorted_f1))]
bars = ax.barh(sorted_f1.index, sorted_f1.values, color=colors_f1, edgecolor=BG, linewidth=1.2, height=0.6)
for bar, val in zip(bars, sorted_f1.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax.set_xlabel('F1-Score', color=TEXT, fontsize=11)
ax.set_title('Model Ranking by F1-Score', color=TEXT, fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(0, 1.1)
ax.axvline(0.8, color=GRID, linestyle='--', linewidth=1.5, alpha=0.7)
save_fig("14_f1_ranking")

print("✅ Model evaluation visualizations saved.")

# ─── 8. HYPERPARAMETER OPTIMIZATION ──────────────────────────────────────────
best_model_name = metrics_df['f1'].idxmax()
print(f"\n🔧 Best model: {best_model_name} — running hyperparameter optimization...")

if best_model_name == "Random Forest":
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
elif best_model_name == "Extra Trees":
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    base_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)
elif best_model_name == "Gradient Boosting":
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:
    param_dist = {'C': [0.1, 1, 10], 'max_iter': [500, 1000]}
    base_model = LogisticRegression(random_state=42)

rs = RandomizedSearchCV(base_model, param_dist, n_iter=10, cv=3, scoring='f1',
                        n_jobs=-1, random_state=42, verbose=0)
rs.fit(X_train_sc, y_train)
best_opt = rs.best_estimator_

y_pred_opt  = best_opt.predict(X_test_sc)
y_prob_opt  = best_opt.predict_proba(X_test_sc)[:, 1]
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_opt)
auc_opt = auc(fpr_opt, tpr_opt)

print(f"  Best params: {rs.best_params_}")
print(f"  Optimized — Acc:{accuracy_score(y_test,y_pred_opt):.4f}  "
      f"F1:{f1_score(y_test,y_pred_opt):.4f}  AUC:{auc_opt:.4f}")

# --- 8.1 Before vs After Optimization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
for ax, (y_p, label) in zip(axes, [(results[best_model_name]['y_pred'], 'Base Model'),
                                    (y_pred_opt, 'Optimized Model')]):
    cm = confusion_matrix(y_test, y_p)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                linewidths=1, linecolor=BG, annot_kws={'size': 16, 'color': TEXT, 'fontweight': 'bold'})
    ax.set_title(f"{best_model_name}\n{label}", color=TEXT, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', color=TEXT); ax.set_ylabel('Actual', color=TEXT)
    ax.set_facecolor(SURFACE)
    ax.set_xticklabels(['Low Eng.', 'High Eng.'], color=TEXT)
    ax.set_yticklabels(['Low Eng.', 'High Eng.'], color=TEXT, rotation=0)
fig.suptitle(f'Optimization Effect on {best_model_name}', color=TEXT, fontsize=14, fontweight='bold')
save_fig("15_optimization_comparison")

# ─── 9. FEATURE IMPORTANCE ────────────────────────────────────────────────────
print("\n🔍 Computing feature importance...")

if hasattr(best_opt, 'feature_importances_'):
    fi = pd.Series(best_opt.feature_importances_, index=feature_cols).sort_values(ascending=False)
else:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_opt, X_test_sc, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    fi = pd.Series(r.importances_mean, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
color_map = [ACCENT1 if v == fi.max() else ACCENT2 if v >= fi.median() else ACCENT5
             for v in fi.values]
bars = ax.barh(fi.index[::-1], fi.values[::-1], color=color_map[::-1], edgecolor=BG, linewidth=1.2, height=0.6)
for bar, val in zip(bars, fi.values[::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', color=TEXT, fontsize=10, fontweight='bold')
ax.set_xlabel('Feature Importance Score', color=TEXT, fontsize=11)
ax.set_title(f'Feature Importance — {best_model_name}', color=TEXT, fontsize=14, fontweight='bold', pad=12)
save_fig("16_feature_importance")

# ─── 10. BUSINESS INSIGHTS DASHBOARD ─────────────────────────────────────────
print("\n💡 Generating business insights dashboard...")

best_day  = day_eng.idxmax()
best_hour = hour_eng.idxmax()
best_cat  = cat_eng.idxmax()
best_mt   = media_eng.idxmax()

# Optimal hashtag range
hash_eng_dict = {str(k): v for k, v in hash_eng.items() if not pd.isna(v)}
best_hash = max(hash_eng_dict, key=hash_eng_dict.get)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('Content Strategy Insights — Instagram Performance Guide',
             color=TEXT, fontsize=17, fontweight='bold', y=1.01)

gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)

# Panel 1: Best Media Type
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
ax1.axis('off')
ax1.text(0.5, 0.75, '🎬 Best Media Type', ha='center', va='center',
         color=ACCENT1, fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.5, 0.40, best_mt, ha='center', va='center',
         color=TEXT, fontsize=22, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.5, 0.15, f'Avg ER: {media_eng.max():.2f}%', ha='center', va='center',
         color=ACCENT4, fontsize=11, transform=ax1.transAxes)
ax1.set_title('Top Media Type', color=TEXT, fontsize=11, pad=6)

# Panel 2: Best Content Category
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD)
ax2.axis('off')
ax2.text(0.5, 0.75, '📌 Best Category', ha='center', va='center',
         color=ACCENT3, fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.40, best_cat, ha='center', va='center',
         color=TEXT, fontsize=18, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.15, f'Avg ER: {cat_eng.max():.2f}%', ha='center', va='center',
         color=ACCENT4, fontsize=11, transform=ax2.transAxes)
ax2.set_title('Top Content Category', color=TEXT, fontsize=11, pad=6)

# Panel 3: Best Day
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(CARD)
ax3.axis('off')
ax3.text(0.5, 0.75, '📅 Best Day to Post', ha='center', va='center',
         color=ACCENT5, fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.40, best_day, ha='center', va='center',
         color=TEXT, fontsize=18, fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.15, f'Avg ER: {day_eng.max():.2f}%', ha='center', va='center',
         color=ACCENT4, fontsize=11, transform=ax3.transAxes)
ax3.set_title('Best Posting Day', color=TEXT, fontsize=11, pad=6)

# Panel 4: Best Hour
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(CARD)
ax4.axis('off')
ax4.text(0.5, 0.75, '⏰ Best Hour to Post', ha='center', va='center',
         color=ACCENT1, fontsize=12, fontweight='bold', transform=ax4.transAxes)
hour_label = f"{best_hour:02d}:00"
ax4.text(0.5, 0.40, hour_label, ha='center', va='center',
         color=TEXT, fontsize=22, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.15, f'Avg ER: {hour_eng.max():.2f}%', ha='center', va='center',
         color=ACCENT4, fontsize=11, transform=ax4.transAxes)
ax4.set_title('Optimal Posting Time', color=TEXT, fontsize=11, pad=6)

# Panel 5: Best Hashtag Range
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(CARD)
ax5.axis('off')
ax5.text(0.5, 0.75, '#️⃣ Best Hashtag Range', ha='center', va='center',
         color=ACCENT3, fontsize=12, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.5, 0.40, f'{best_hash} tags', ha='center', va='center',
         color=TEXT, fontsize=18, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.5, 0.15, f'Avg ER: {max(hash_eng_dict.values()):.2f}%', ha='center', va='center',
         color=ACCENT4, fontsize=11, transform=ax5.transAxes)
ax5.set_title('Hashtag Strategy', color=TEXT, fontsize=11, pad=6)

# Panel 6: Model Performance Summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD)
ax6.axis('off')
ax6.text(0.5, 0.88, '🤖 Best ML Model', ha='center', va='center',
         color=ACCENT5, fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.68, best_model_name.replace(' ', '\n'), ha='center', va='center',
         color=TEXT, fontsize=13, fontweight='bold', transform=ax6.transAxes)
best_res = results[best_model_name]
for i, (label, val) in enumerate([('Accuracy', best_res['accuracy']),
                                    ('F1-Score', best_res['f1']),
                                    ('AUC-ROC', best_res['roc_auc'])]):
    ax6.text(0.5, 0.42 - i*0.15, f'{label}: {val:.4f}', ha='center', va='center',
             color=ACCENT4, fontsize=10, transform=ax6.transAxes)
ax6.set_title('Predictive Model Summary', color=TEXT, fontsize=11, pad=6)

save_fig("17_business_insights_dashboard")
print("✅ Business insights dashboard saved.")

# ─── 11. PRINT FINAL SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*65)
print("📊 FINAL MODEL PERFORMANCE SUMMARY")
print("="*65)
print(f"{'Model':<26} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("-"*65)
for name, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
    print(f"{name:<26} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
          f"{res['recall']:>7.4f} {res['f1']:>7.4f} {res['roc_auc']:>7.4f}")
print("="*65)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"   F1-Score:  {results[best_model_name]['f1']:.4f}")
print(f"   AUC-ROC:   {results[best_model_name]['roc_auc']:.4f}")

print(f"\n💡 KEY INSIGHTS FOR CONTENT STRATEGY:")
print(f"   • Best media type:      {best_mt} ({media_eng.max():.2f}% avg ER)")
print(f"   • Best content category:{best_cat} ({cat_eng.max():.2f}% avg ER)")
print(f"   • Best day to post:     {best_day} ({day_eng.max():.2f}% avg ER)")
print(f"   • Best hour to post:    {best_hour:02d}:00 ({hour_eng.max():.2f}% avg ER)")
print(f"   • Optimal hashtag range:{best_hash} tags")

print(f"\n✅ All visualizations saved to: visualizations/")

# ─── 11. REGRESSION: Predict Engagement Rate ─────────────────────────────────
print("\n📈 Running regression to predict engagement rate (continuous target)...")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_reg   = df_ml[feature_cols]
y_reg   = df['engagement_rate_calc'].values

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r   = MinMaxScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s  = scaler_r.transform(Xr_test)

reg_models = {
    "Ridge Regression":         Ridge(alpha=1.0),
    "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting Reg.":   GradientBoostingRegressor(n_estimators=100, random_state=42),
}

reg_results = {}
for rname, rmodel in reg_models.items():
    print(f"  Training {rname}...", end=" ")
    rmodel.fit(Xr_train_s, yr_train)
    yr_pred = rmodel.predict(Xr_test_s)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    mae  = mean_absolute_error(yr_test, yr_pred)
    r2   = r2_score(yr_test, yr_pred)
    reg_results[rname] = {"model": rmodel, "rmse": rmse, "mae": mae, "r2": r2, "y_pred": yr_pred}
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

# Regression comparison chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
names_r = list(reg_results.keys())
rmse_vals = [reg_results[n]['rmse'] for n in names_r]
r2_vals   = [reg_results[n]['r2']   for n in names_r]

bars = axes[0].bar(names_r, rmse_vals, color=[ACCENT1, ACCENT3, ACCENT5], edgecolor=BG, width=0.5)
for bar, val in zip(bars, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.4f}', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
axes[0].set_title('Regression Models — RMSE (lower is better)', color=TEXT, fontsize=13, fontweight='bold')
axes[0].set_ylabel('RMSE', color=TEXT)
axes[0].tick_params(axis='x', rotation=15)

bars2 = axes[1].bar(names_r, r2_vals, color=[ACCENT1, ACCENT3, ACCENT5], edgecolor=BG, width=0.5)
for bar, val in zip(bars2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
axes[1].set_title('Regression Models — R² Score (higher is better)', color=TEXT, fontsize=13, fontweight='bold')
axes[1].set_ylabel('R² Score', color=TEXT)
axes[1].tick_params(axis='x', rotation=15)
save_fig("18_regression_comparison")

# Predicted vs Actual for best regressor
best_reg_name = max(reg_results, key=lambda n: reg_results[n]['r2'])
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
sample_idx = np.random.choice(len(yr_test), 500, replace=False)
ax.scatter(yr_test[sample_idx], reg_results[best_reg_name]['y_pred'][sample_idx],
           color=ACCENT1, alpha=0.4, s=20, edgecolors='none')
lims = [min(yr_test.min(), reg_results[best_reg_name]['y_pred'].min()),
        max(yr_test.max(), reg_results[best_reg_name]['y_pred'].max())]
ax.plot(lims, lims, color=ACCENT3, linewidth=2, linestyle='--', label='Perfect Prediction')
ax.set_xlabel('Actual Engagement Rate (%)', color=TEXT, fontsize=11)
ax.set_ylabel('Predicted Engagement Rate (%)', color=TEXT, fontsize=11)
ax.set_title(f'Predicted vs Actual — {best_reg_name}\nR²={reg_results[best_reg_name]["r2"]:.4f}',
             color=TEXT, fontsize=13, fontweight='bold')
ax.legend(labelcolor=TEXT, facecolor=SURFACE)
save_fig("19_predicted_vs_actual")

print(f"\n✅ Regression analysis complete. Best regressor: {best_reg_name}")

# ─── 12. SAVE SUMMARY JSON ────────────────────────────────────────────────────
def to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

summary = {
    "best_classifier": best_model_name,
    "best_regressor": best_reg_name,
    "classifier_scores": {
        name: {
            "accuracy": float(res['accuracy']), "f1": float(res['f1']),
            "roc_auc": float(res['roc_auc']), "precision": float(res['precision']),
            "recall": float(res['recall'])
        } for name, res in results.items()
    },
    "regression_scores": {
        name: {"rmse": float(r['rmse']), "mae": float(r['mae']), "r2": float(r['r2'])}
        for name, r in reg_results.items()
    },
    "insights": {
        "best_media_type": str(best_mt),
        "best_category": str(best_cat),
        "best_day": str(best_day),
        "best_hour": int(best_hour),
        "best_hashtag_range": str(best_hash),
        "best_engagement_rate": float(media_eng.max()),
        "median_engagement_rate": float(median_er),
    },
    "encoders": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoders.items()},
    "feature_cols": feature_cols,
    "category_avg_engagement": {str(k): float(v) for k, v in cat_eng.items()},
    "media_avg_engagement":    {str(k): float(v) for k, v in media_eng.items()},
    "day_avg_engagement":      {str(k): float(v) for k, v in day_eng.items() if not pd.isna(v)},
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✅ model_summary.json saved.")
