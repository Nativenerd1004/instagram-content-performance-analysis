"""Regression extension — runs after main analysis visualizations are already saved."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BG      = "#0D0D0D"; SURFACE = "#1A1A2E"; CARD = "#16213E"
ACCENT1 = "#00D4FF"; ACCENT2 = "#7B2FBE"; ACCENT3 = "#FF6B35"
ACCENT4 = "#00FF88"; ACCENT5 = "#FF3CAC"; TEXT = "#E0E0E0"; GRID = "#2A2A4A"
PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5, "#FFD700", "#FF4560", "#00E396"]

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.4,
    'axes.grid': True, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
})

def save_fig(name):
    path = f"visualizations/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ Saved {path}")

# Load & prep
df = pd.read_csv("/Users/apple/Desktop/Instagram Analytics Datasets/Full Dataset/Intagram_CaseStudy_Cleaned_sql.csv")
median_er = df['engagement_rate_calc'].median()
df['high_engagement'] = (df['engagement_rate_calc'] >= median_er).astype(int)

feature_cols = ['media_type', 'caption_length', 'hashtags_count',
                'traffic_source', 'content_category', 'Hour', 'Month', 'Day name']
le = LabelEncoder()
cat_cols = ['media_type', 'traffic_source', 'content_category', 'Day name']
encoders = {}
df_ml = df[feature_cols].copy()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col])
    encoders[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

X_reg = df_ml[feature_cols]
y_reg = df['engagement_rate_calc'].values
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler_r   = MinMaxScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s  = scaler_r.transform(Xr_test)

print("📈 Training regression models...")
reg_models = {
    "Ridge Regression":         Ridge(alpha=1.0),
    "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting Reg.":   GradientBoostingRegressor(n_estimators=100, random_state=42),
}
reg_results = {}
for rname, rmodel in reg_models.items():
    print(f"  {rname}...", end=" ")
    rmodel.fit(Xr_train_s, yr_train)
    yr_pred = rmodel.predict(Xr_test_s)
    rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
    mae  = float(mean_absolute_error(yr_test, yr_pred))
    r2   = float(r2_score(yr_test, yr_pred))
    reg_results[rname] = {"model": rmodel, "rmse": rmse, "mae": mae, "r2": r2, "y_pred": yr_pred}
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

# --- Chart 18: Regression comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
names_r   = list(reg_results.keys())
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

# --- Chart 19: Predicted vs Actual ---
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

# Aggregate stats for JSON
cat_eng   = df.groupby('content_category')['engagement_rate_calc'].mean()
media_eng = df.groupby('media_type')['engagement_rate_calc'].mean()
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_eng   = df.groupby('Day name')['engagement_rate_calc'].mean().reindex(day_order)
best_mt   = media_eng.idxmax()
best_cat  = cat_eng.idxmax()
best_day  = day_eng.idxmax()
hour_eng  = df.groupby('Hour')['engagement_rate_calc'].mean()
best_hour = hour_eng.idxmax()
hashtag_bins = pd.cut(df['hashtags_count'], bins=[0,5,10,15,20,30], labels=['1-5','6-10','11-15','16-20','21-30'])
hash_eng = df.groupby(hashtag_bins)['engagement_rate_calc'].mean()
best_hash = str(hash_eng.idxmax())

summary = {
    "best_regressor": best_reg_name,
    "regression_scores": {
        name: {"rmse": r['rmse'], "mae": r['mae'], "r2": r['r2']}
        for name, r in reg_results.items()
    },
    "insights": {
        "best_media_type": str(best_mt),
        "best_category": str(best_cat),
        "best_day": str(best_day),
        "best_hour": int(best_hour),
        "best_hashtag_range": best_hash,
        "best_engagement_rate": float(media_eng.max()),
        "median_engagement_rate": float(median_er),
    },
    "encoders": encoders,
    "feature_cols": feature_cols,
    "category_avg_engagement": {str(k): float(v) for k, v in cat_eng.items()},
    "media_avg_engagement":    {str(k): float(v) for k, v in media_eng.items()},
    "day_avg_engagement":      {str(k): float(v) for k, v in day_eng.items() if not pd.isna(v)},
    "hour_avg_engagement":     {str(int(k)): float(v) for k, v in hour_eng.items()},
}
with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✅ model_summary.json saved.")
print(f"\n🏆 Best Regressor: {best_reg_name}")
for k, v in reg_results[best_reg_name].items():
    if k != 'model' and k != 'y_pred':
        print(f"   {k.upper()}: {v:.4f}")
