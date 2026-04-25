import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


os.makedirs('results', exist_ok=True)


pruning_df = pd.read_csv('results/pruning_metrics.csv')
pruning_df = pruning_df[['config', 'latency_ms', 'latency_std_ms', 'cv_pct',
                         'model_size_mb', 'peak_memory_mb', 'accuracy_pct']]

quant_df = pd.read_csv('results/quantization_metrics.csv')
quant_df = quant_df[['config', 'latency_ms', 'latency_std_ms', 'cv_pct',
                     'model_size_mb', 'peak_memory_mb', 'accuracy_pct']]

stacked_df = pd.read_csv('results/stacked_metrics.csv')
stacked_df = stacked_df[['config', 'latency_ms', 'latency_std_ms', 'cv_pct',
                         'model_size_mb', 'peak_memory_mb', 'accuracy_pct']]

# The quantization CSV includes its own FP32 baseline, but pruning_df already
# has it as Pruned_0pct — drop it here to avoid duplicates in the final merge
quant_df = quant_df[quant_df['config'] != 'FP32_Baseline']

df = pd.concat([pruning_df, quant_df, stacked_df], ignore_index=True)
df.to_csv('results/metrics_all.csv', index=False)
print("Saved: results/metrics_all.csv")
print(df.to_string(index=False))


COLOR_MAP = {
    'Pruned_0pct': '#4C72B0',
    'Pruned_25pct': '#55A868',
    'Pruned_50pct': '#C44E52',
    'Pruned_75pct': '#8172B2',
    'Pruned_90pct': '#CCB974',
    'INT8_Weight_Only': '#64B5CD',
    'Pruned50pct_INT8': '#E77F24',
}
colors = [COLOR_MAP.get(c, '#888888') for c in df['config']]
configs = df['config'].tolist()
x = range(len(configs))


fig, axes = plt.subplots(1, 4, figsize=(26, 6))
fig.suptitle('ResNet-18 CIFAR-10 Compression Benchmark',
             fontsize=16, fontweight='bold')

axes[0].bar(x, df['latency_ms'], color=colors, edgecolor='black', linewidth=0.5,
            yerr=df['latency_std_ms'], capsize=4,
            error_kw={'elinewidth': 1.2, 'ecolor': '#333333'})
axes[0].set_title(
  'Inference Latency (ms / batch)\n± std dev over 500 runs', fontsize=11)
axes[0].set_ylabel('Latency (ms)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(configs, rotation=40, ha='right', fontsize=8)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(df['latency_ms']):
  # Push the label above the error bar cap so it doesn't overlap
  offset = df['latency_std_ms'].iloc[i] + 2
  axes[0].text(i, v + offset, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

axes[1].bar(x, df['model_size_mb'], color=colors,
            edgecolor='black', linewidth=0.5)
axes[1].set_title('Model Size on Disk (MB)', fontsize=11)
axes[1].set_ylabel('Size (MB)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(configs, rotation=40, ha='right', fontsize=8)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(df['model_size_mb']):
  axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

axes[2].bar(x, df['peak_memory_mb'], color=colors,
            edgecolor='black', linewidth=0.5)
axes[2].set_title('Peak Memory Consumption (MB)', fontsize=11)
axes[2].set_ylabel('Memory (MB)')
axes[2].set_xticks(x)
axes[2].set_xticklabels(configs, rotation=40, ha='right', fontsize=8)
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(df['peak_memory_mb']):
  axes[2].text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

axes[3].bar(x, df['cv_pct'], color=colors, edgecolor='black', linewidth=0.5)
axes[3].axhline(y=5.0, color='red', linestyle='--', linewidth=1.2,
                label='5% stability threshold')
axes[3].set_title('Latency Stability\n(CV% over 500 runs)', fontsize=11)
axes[3].set_ylabel('Coefficient of Variation (%)')
axes[3].set_xticks(x)
axes[3].set_xticklabels(configs, rotation=40, ha='right', fontsize=8)
axes[3].grid(axis='y', alpha=0.3)
axes[3].legend(fontsize=9)
for i, v in enumerate(df['cv_pct']):
  axes[3].text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('results/performance_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/performance_charts.png")


fig2, ax = plt.subplots(figsize=(11, 7))
legend_handles = []

for _, row in df.iterrows():
  c = COLOR_MAP.get(row['config'], '#888888')
  ax.errorbar(row['latency_ms'], row['accuracy_pct'],
              xerr=row['latency_std_ms'],
              fmt='o', color=c, markersize=12, markeredgecolor='black',
              markeredgewidth=0.8, elinewidth=1.2, ecolor=c,
              capsize=4, zorder=5)
  legend_handles.append(mpatches.Patch(color=c, label=row['config']))

# Build the Pareto frontier — walk left to right on latency, and only keep a
# point if it's at least as accurate as everything we've seen so far
pareto_candidates = df.sort_values('latency_ms').reset_index(drop=True)
pareto_points = []
max_acc = -1
for _, row in pareto_candidates.iterrows():
  if row['accuracy_pct'] >= max_acc:
    pareto_points.append(row)
    max_acc = row['accuracy_pct']

pareto_df = pd.DataFrame(pareto_points).sort_values('latency_ms')
ax.plot(pareto_df['latency_ms'], pareto_df['accuracy_pct'],
        'k--', linewidth=1.5, zorder=4)

baseline_acc = df[df['config'] == 'Pruned_0pct']['accuracy_pct'].values[0]
threshold = baseline_acc - 5.0
ax.axhline(y=threshold, color='red', linestyle=':', linewidth=1.5)

# Blank patch acts as a visual spacer between model entries and annotation lines
legend_handles.append(mpatches.Patch(color='none', label=''))
legend_handles.append(plt.Line2D([0], [0], color='black', linestyle='--',
                                 linewidth=1.5, label='Pareto Frontier'))
legend_handles.append(plt.Line2D([0], [0], color='red', linestyle=':',
                                 linewidth=1.5,
                                 label=f'Accuracy threshold (baseline − 5% = {threshold:.1f}%)'))

ax.legend(handles=legend_handles, fontsize=9, loc='center right',
          framealpha=0.9, edgecolor='gray')
ax.set_xlabel('Inference Latency (ms per batch) ± std dev', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('Pareto Frontier: Latency vs. Accuracy\nResNet-18 Compression Configurations',
             fontsize=13)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/pareto_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/pareto_frontier.png")
