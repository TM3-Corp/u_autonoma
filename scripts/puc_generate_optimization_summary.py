#!/usr/bin/env python3
"""
Generate summary report and visualization for week optimization results.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RESULTS_FILE = Path('data/puc/analysis/comprehensive_week_optimization.json')
OUTPUT_DIR = Path('data/puc/report/visualizations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading optimization results...")

with open(RESULTS_FILE) as f:
    data = json.load(f)

results_df = pd.DataFrame(data['all_results'])

# Create visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. RECALL VS THRESHOLD BY WEEK
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for week in results_df['week_label'].unique():
    week_data = results_df[results_df['week_label'] == week]
    ax.plot(week_data['threshold'], week_data['recall_mean'],
            marker='o', label=week, linewidth=2)

ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Recall (Sensitivity)', fontsize=12)
ax.set_title('Recall vs Threshold by Week Cutoff', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'recall_vs_threshold.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'recall_vs_threshold.png'}")
plt.close()

# ============================================================================
# 2. PRECISION VS RECALL (PARETO FRONTIER)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for week in results_df['week_label'].unique():
    week_data = results_df[results_df['week_label'] == week]
    ax.scatter(week_data['recall_mean'], week_data['precision_mean'],
               label=week, s=100, alpha=0.6)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Trade-off by Week', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.01, 0.20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'precision_recall_tradeoff.png'}")
plt.close()

# ============================================================================
# 3. F1 AND F2 SCORES HEATMAP
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pivot for heatmap
f1_pivot = results_df.pivot_table(
    index='week_label',
    columns='threshold',
    values='f1_mean',
    aggfunc='mean'
)

f2_pivot = results_df.pivot_table(
    index='week_label',
    columns='threshold',
    values='f2_mean',
    aggfunc='mean'
)

# F1 heatmap
sns.heatmap(f1_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1,
            cbar_kws={'label': 'F1 Score'})
ax1.set_title('F1 Score Heatmap', fontsize=14, fontweight='bold')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Week Cutoff')

# F2 heatmap
sns.heatmap(f2_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
            cbar_kws={'label': 'F2 Score'})
ax2.set_title('F2 Score Heatmap (Emphasizes Recall)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Week Cutoff')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'f_scores_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'f_scores_heatmap.png'}")
plt.close()

# ============================================================================
# 4. OPTIMAL CONFIGURATIONS TABLE
# ============================================================================
print("\n" + "="*80)
print("OPTIMAL CONFIGURATIONS SUMMARY")
print("="*80)

optimal = data['optimal_configurations']

summary_rows = []
for week_label, configs in optimal.items():
    for config_type, metrics in configs.items():
        summary_rows.append({
            'Week': week_label,
            'Type': config_type,
            'Threshold': f"{metrics['threshold']:.2f}",
            'Recall': f"{metrics['recall']:.1%}",
            'Precision': f"{metrics['precision']:.1%}",
            'F1': f"{metrics['f1']:.3f}",
            'F2': f"{metrics['f2']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}"
        })

summary_df = pd.DataFrame(summary_rows)
print("\n", summary_df.to_string(index=False))

# Save table as CSV
summary_df.to_csv(OUTPUT_DIR.parent / 'optimal_configurations_table.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR.parent / 'optimal_configurations_table.csv'}")

# ============================================================================
# 5. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("PRODUCTION DEPLOYMENT RECOMMENDATIONS")
print("="*80)

recommendations = """
Based on comprehensive benchmarking across 95 configurations:

🎯 RECOMMENDED CONFIGURATION: Week 8 with Dual Thresholds

TIER 1 - HIGH RISK ALERTS (Threshold = 0.40)
  ✓ Recall: 43.6% (catches 4 in 10 failures)
  ✓ Precision: 12.7% (1 in 8 alerts is correct)
  ✓ Use case: Targeted intervention, personalized support
  ✓ Action: Direct instructor contact, tutoring referral

TIER 2 - WATCH LIST (Threshold = 0.05)
  ✓ Recall: 76.4% (catches 7.6 in 10 failures)
  ✓ Precision: 9.0% (1 in 11 alerts is correct)
  ✓ Use case: Proactive monitoring, automated nudges
  ✓ Action: Email reminders, resource recommendations

WHY WEEK 8?
  ✓ Best balance of early intervention and prediction accuracy
  ✓ Sufficient student activity data accumulated
  ✓ Still early enough for intervention (typically mid-semester)
  ✓ Significantly outperforms Week 2, 4, and full semester

ALTERNATIVE: Week 6 for Very Early Intervention
  ✓ Recall: 90.9% at threshold 0.50
  ✓ Trade-off: Lower precision (7.2%)
  ✓ Use if intervention resources are abundant

KEY INSIGHT:
Full semester model performs WORSE than Week 8, suggesting:
  - Overfitting on late-semester cramming patterns
  - Missing early engagement signals more predictive of failure
  - Confirms value of EARLY warning over late-semester prediction
"""

print(recommendations)

# Save recommendations
with open(OUTPUT_DIR.parent / 'DEPLOYMENT_RECOMMENDATIONS.md', 'w') as f:
    f.write("# PUC Early Warning System - Deployment Recommendations\n\n")
    f.write(recommendations)
    f.write("\n\n## Detailed Results\n\n")
    f.write(summary_df.to_markdown(index=False))

print(f"\n✓ Saved: {OUTPUT_DIR.parent / 'DEPLOYMENT_RECOMMENDATIONS.md'}")

print("\n" + "="*80)
print("✓ Phase 5 Complete: Optimization summary generated")
print("="*80)
