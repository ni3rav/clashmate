#!/usr/bin/env python3
"""
ClashMate - Elixir Cost Prediction Model
Visualization script for generating statistical plots and summary tables.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Create graphs directory if it doesn't exist
os.makedirs('graphs', exist_ok=True)

# Set matplotlib style for academic presentation
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3

# ============================================================================
# 1. Dataset Growth Plot
# ============================================================================
print("Generating dataset growth plot...")
fig, ax = plt.subplots(figsize=(8, 6))

stages = ['Extracted', 'Cleaned', 'Augmented']
counts = [155, 120, 600]

bars = ax.bar(stages, counts, color='gray', edgecolor='black', linewidth=1)

# Add value labels above each bar
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom', fontsize=11, color='black')

ax.set_xlabel('Stage', color='black', fontsize=12)
ax.set_ylabel('Number of Cards', color='black', fontsize=12)
ax.set_title('Dataset Size Progression', fontsize=14, color='black', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('graphs/graph_dataset_growth.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Elixir Cost Distribution
# ============================================================================
print("Generating elixir cost distribution plot...")
fig, ax = plt.subplots(figsize=(10, 6))

elixir_costs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
card_counts = [5, 14, 26, 36, 20, 12, 5, 1, 1]
total_cards = sum(card_counts)

bars = ax.bar(elixir_costs, card_counts, color='gray', edgecolor='black', linewidth=1)

# Add data labels with count and percentage
for bar, count in zip(bars, card_counts):
    height = bar.get_height()
    percentage = (count / total_cards) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=9, color='black')

ax.set_xlabel('Elixir Cost', color='black', fontsize=12)
ax.set_ylabel('Card Count', color='black', fontsize=12)
ax.set_title('Distribution of Elixir Costs (120 Cards)', fontsize=14, color='black', fontweight='bold')
ax.set_xticks(elixir_costs)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('graphs/graph_elixir_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Feature Importance Plot
# ============================================================================
print("Generating feature importance plot...")
fig, ax = plt.subplots(figsize=(10, 7))

features = [
    'hitpoints_clean',
    'hp_damage_ratio',
    'damage_clean',
    'count_clean',
    'range_clean',
    'dps_clean',
    'hitSpeed_clean',
    'has_area_damage',
    'type_encoded',
    'is_cycle_card'
]
# Updated importance values with non-leaky proxy (is_cycle_card dropped from 28.76% to 0.59%)
importance = [30.70, 17.62, 12.74, 9.94, 8.25, 7.82, 6.31, 3.93, 2.12, 0.59]

# Sort descending (already sorted, but ensure)
data = list(zip(features, importance))
data.sort(key=lambda x: x[1], reverse=True)
features_sorted, importance_sorted = zip(*data)

bars = ax.barh(features_sorted, importance_sorted, color='gray', edgecolor='black', linewidth=1)

# Add value labels on each bar
for bar, val in zip(bars, importance_sorted):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{val:.2f}',
            ha='left', va='center', fontsize=9, color='black', fontweight='bold')

ax.set_xlabel('Importance', color='black', fontsize=12)
ax.set_ylabel('Feature', color='black', fontsize=12)
ax.set_title('Feature Importance in Elixir Prediction (Non-Leaky Proxy)', fontsize=14, color='black', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('graphs/graph_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Train vs Test Metrics Comparison
# ============================================================================
print("Generating train vs test metrics comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
# Updated metrics with non-leaky proxy (methodologically sound performance)
train_values = [0.9358, 0.3879, 0.2878, 9.23]
test_values = [0.8444, 0.6013, 0.4741, 15.24]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='gray', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='lightgray', edgecolor='black', linewidth=1)

# Add value labels above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}' if height < 1 else f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, color='black')

ax.set_xlabel('Metric', color='black', fontsize=12)
ax.set_ylabel('Value', color='black', fontsize=12)
ax.set_title('Model Performance: Train vs Test (Non-Leaky Proxy)', fontsize=14, color='black', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right', frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('graphs/graph_train_test_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Metrics Summary Table
# ============================================================================
print("\n" + "="*70)
print("METRICS SUMMARY TABLE")
print("="*70 + "\n")

table_data = [
    ['R²', '0.9358', '0.8444', 'Good (methodologically sound)'],
    ['RMSE', '0.3879', '0.6013', 'Acceptable'],
    ['MAE', '0.2878', '0.4741', 'Acceptable'],
    ['MAPE', '9.23%', '15.24%', 'Acceptable']
]

headers = ['Metric', 'Train', 'Test', 'Interpretation']
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Calculate and print R² gap
r2_gap = 0.9358 - 0.8444
r2_gap_percent = (r2_gap / 0.9358) * 100
print(f"\nTrain–Test R² gap: {r2_gap:.4f} ({r2_gap_percent:.2f}%)")
print("\nNote: Metrics updated after fixing data leakage in is_cycle_card feature.")
print("The drop in performance (R²: 0.8971 → 0.8444) validates the leak fix.")

# ============================================================================
# Completion Message
# ============================================================================
print("\n" + "="*70)
print("[✓] All graphs generated and saved in /graphs/")
print("="*70)

