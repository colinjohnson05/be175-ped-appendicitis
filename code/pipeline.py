import sys
from pathlib import Path
import numpy as np
import pandas as pd
from data_pp import appendicitis_pp
from app_utils import (mask_feature_sets,
                       compute_costs_over_thresholds,
                       get_optimal_threshold,
                       scale_costs,
                       plot_cost_curves_grid,
                       plot_aggregated_mean_curve,
                       plot_sensitivity_lines,
                       total_cost)
from sklearn.model_selection import StratifiedKFold
from xgboost_model import train_xgboost
import matplotlib.pyplot as plt

data_path = Path.cwd() / '..' / 'data' / 'app_data.xlsx'
data, descriptors = appendicitis_pp(data_path)

# Config
PALETTE = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377',
           '#CC3311', '#009988', '#BBBBBB']

FEATURE_SET_COLORS = {
    'Clinical':   '#EE7733',
    'Laboratory': '#0077BB',
    'Ultrasound': '#33BBEE',
}

SENS_COLORS = {
    'FN cost':         '#CC3311',
    'FP cost':         '#EE7733',
    'Clinical cost':   '#EE7733',
    'Laboratory cost': '#0077BB',
    'Ultrasound cost': '#33BBEE',
}

FEATURE_SETS  = ['Clinical', 'Laboratory', 'Ultrasound']
COST_PARAMS   = ['FN cost', 'FP cost', 'Clinical cost', 'Laboratory cost', 'Ultrasound cost']
FOLD_CHANGES  = np.logspace(-1, 1, num=100)
COST_RATIOS   = np.logspace(-1, 1, num=100)
THRESHOLDS    = np.linspace(0.1, 0.9, num=200)
N_THRESHOLDS  = len(THRESHOLDS)

fn_cost = 30000
fp_cost = 20000
cost_sum = fn_cost + fp_cost

feature_cost = {
    'Clinical':   300,
    'Laboratory': 750,
    'Ultrasound': 1500,
}

BASE_COSTS = {
    'FN cost':         fn_cost,
    'FP cost':         fp_cost,
    'Clinical cost':   feature_cost['Clinical'],
    'Laboratory cost': feature_cost['Laboratory'],
    'Ultrasound cost': feature_cost['Ultrasound'],
}

usd_formatter = plt.FuncFormatter(lambda val, _: f'${val:,.0f}')

output_dir = Path.cwd() / '..' / 'results'

# Data prep
groups = data['Class']
y = data['Diagnosis']
X = data.drop(columns=['Class', 'Diagnosis', 'Diagnosis_Presumptive'])

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

N_OUTER_FOLDS = outer_cv.get_n_splits()
N_INNER_FOLDS = inner_cv.get_n_splits()

# Storage arrays
all_outer_costs = {fs: np.zeros((N_OUTER_FOLDS, N_THRESHOLDS)) for fs in FEATURE_SETS}
outer_costs = {fs: np.zeros(N_OUTER_FOLDS) for fs in FEATURE_SETS + ['Baseline']}
outer_roc_auc = {fs: np.zeros(N_OUTER_FOLDS) for fs in FEATURE_SETS + ['Baseline']}
outer_feature_importance = {fs: [] for fs in FEATURE_SETS}
outer_y_test = {}
outer_y_probs = {fs: {} for fs in FEATURE_SETS}

# Main CV Loop
for outer_fold, (outer_train_idx, test_idx) in enumerate(outer_cv.split(X, groups)):

    # Progress statement
    print(f"Outer fold {outer_fold + 1}")

    X_outer_train, X_test = X.iloc[outer_train_idx], X.iloc[test_idx]
    y_outer_train, y_test = y.iloc[outer_train_idx], y.iloc[test_idx]
    outer_groups = groups.iloc[outer_train_idx]

    rows = []
    inner_costs = {fs: np.zeros((N_INNER_FOLDS, N_THRESHOLDS)) for fs in FEATURE_SETS}

    # Inner CV loop
    for inner_fold, (inner_train_idx, val_idx) in enumerate(inner_cv.split(X_outer_train, outer_groups)):

        print(f"Inner fold {inner_fold + 1}")

        X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[val_idx]
        y_inner_train, y_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[val_idx]

        X_inner_train_clinical, X_inner_train_lab, X_inner_train_us = mask_feature_sets(X_inner_train, descriptors)
        X_inner_val_clinical, X_inner_val_lab, X_inner_val_us = mask_feature_sets(X_inner_val, descriptors)

        X_train_dict = {'Clinical': X_inner_train_clinical,
                        'Laboratory': X_inner_train_lab,
                        'Ultrasound': X_inner_train_us}
        X_val_dict = {'Clinical': X_inner_val_clinical,
                      'Laboratory': X_inner_val_lab,
                      'Ultrasound': X_inner_val_us}

        for col_idx, feature_set in enumerate(FEATURE_SETS):
            model, metrics, y_prob = train_xgboost(
                X_train_dict[feature_set], X_val_dict[feature_set],
                y_inner_train, y_val, random_state=42
            )

            costs = compute_costs_over_thresholds(
                y_val, y_prob,
                fn=fn_cost, fp=fp_cost,
                fc=feature_cost[feature_set],
                thresholds=THRESHOLDS
            )

            inner_costs[feature_set][inner_fold] = costs

            optimal_threshold_idx = np.argmin(costs)
            optimal_threshold = THRESHOLDS[optimal_threshold_idx]
            optimal_cost = costs[optimal_threshold_idx]

            rows.append({'Inner Fold': inner_fold + 1,
                         'Feature Set': feature_set,
                         'Threshold': optimal_threshold,
                         'Cost': optimal_cost})

        print(f"Inner fold {inner_fold + 1} complete")

    feature_set_optimal_threshold = {fs: 0 for fs in FEATURE_SETS}

    for col_idx, feature_set in enumerate(FEATURE_SETS):
        mean_costs = inner_costs[feature_set].mean(axis=0)
        std_costs = inner_costs[feature_set].std(axis=0)

        optimal_idx = np.argmin(mean_costs)
        optimal_threshold = THRESHOLDS[optimal_idx]
        optimal_cost = mean_costs[optimal_idx]

        all_outer_costs[feature_set][outer_fold] = mean_costs
        feature_set_optimal_threshold[feature_set] = optimal_threshold

    # Summary
    optimal_outcomes = pd.DataFrame(rows, columns=['Inner Fold', 'Feature Set',
                                                   'Threshold', 'Cost'])
    print(optimal_outcomes)
    print(f'Outer fold {outer_fold + 1} complete')

    # Outer test
    X_outer_train_clinical, X_outer_train_lab, X_outer_train_us = mask_feature_sets(X_outer_train, descriptors)
    X_test_clinical, X_test_lab, X_test_us = mask_feature_sets(X_test, descriptors)

    X_train_dict = {'Clinical': X_outer_train_clinical,
                    'Laboratory': X_outer_train_lab,
                    'Ultrasound': X_outer_train_us}
    X_test_dict = {'Clinical': X_test_clinical,
                   'Laboratory': X_test_lab,
                   'Ultrasound': X_test_us}

    for feature_set in FEATURE_SETS:
        model, metrics, y_prob = train_xgboost(
            X_train_dict[feature_set], X_test_dict[feature_set],
            y_outer_train, y_test, random_state=42
        )

        cost = total_cost(
            y_test, y_prob,
            threshold=feature_set_optimal_threshold[feature_set],
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            feature_cost=feature_cost[feature_set],
        )

        outer_costs[feature_set][outer_fold] = cost
        outer_roc_auc[feature_set][outer_fold] = metrics['roc_auc']
        outer_feature_importance[feature_set].append(metrics['feature_importance'])
        outer_y_probs[feature_set][outer_fold] = y_prob

    outer_y_test[outer_fold] = y_test

    # Baseline — ultrasound model at threshold 0.5
    model, metrics, y_prob = train_xgboost(
        X_train_dict['Ultrasound'], X_test_dict['Ultrasound'],
        y_outer_train, y_test, random_state=42
    )

    outer_costs['Baseline'][outer_fold] = total_cost(
        y_test, y_prob, threshold=0.5,
        fn_cost=fn_cost, fp_cost=fp_cost,
        feature_cost=feature_cost['Ultrasound'],
    )
    outer_roc_auc['Baseline'][outer_fold] = metrics['roc_auc']

# Aggregate cost curve across outer folds
fig_final, axes_final = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharey=True)
fig_final.suptitle('Cross Validation — Mean Cost vs Threshold',
                   fontweight='bold', fontsize=20)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    mean_costs = all_outer_costs[feature_set].mean(axis=0)
    std_costs = all_outer_costs[feature_set].std(axis=0)
    optimal_idx = np.argmin(mean_costs)
    optimal_threshold = THRESHOLDS[optimal_idx]
    optimal_cost = mean_costs[optimal_idx]

    ax = axes_final[col_idx]
    ax.tick_params(axis='both', labelsize=16)
    plot_aggregated_mean_curve(
        ax, THRESHOLDS, mean_costs, std_costs,
        optimal_threshold, optimal_cost, feature_set, FEATURE_SET_COLORS,
        ylabel='Cost (USD)', formatter=usd_formatter
    )

plt.tight_layout()
plt.savefig(output_dir / 'cross_val.pdf', bbox_inches='tight')

# Mean cost barplot
fig, ax = plt.subplots(figsize=(7, 5))
bar_colors = [FEATURE_SET_COLORS.get(fs, PALETTE[4]) for fs in FEATURE_SETS + ['Baseline']]
x = np.arange(len(FEATURE_SETS) + 1)
np.random.seed(42)

for i, feature_set in enumerate(FEATURE_SETS + ['Baseline']):
    costs = outer_costs[feature_set]
    mean_cost = costs.mean()
    std_cost = costs.std()

    ax.bar(x[i], mean_cost, width=0.5, color=bar_colors[i], label=feature_set,
           yerr=std_cost, capsize=5,
           error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})
    ax.scatter(np.random.normal(x[i], 0.05, size=len(costs)),
               costs, color='black', zorder=5, s=25, alpha=0.7)

ax.set_title('Mean Total Cost by Feature Set', fontweight='bold', fontsize=18)
ax.set_ylabel('Total Cost (USD)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_SETS + ['Baseline'], fontsize=14)
ax.yaxis.set_major_formatter(usd_formatter)
ax.tick_params(axis='y', labelsize=14)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'final_model_results.pdf', bbox_inches='tight')

# ROC-AUC
print('\n' + '=' * 60)
print('ROC-AUC SUMMARY (mean ± std across outer folds)')
print('=' * 60)
for feature_set in FEATURE_SETS + ['Baseline']:
    aucs = outer_roc_auc[feature_set]
    fold_str = '  '.join([f'Fold {i + 1}: {auc:.3f}' for i, auc in enumerate(aucs)])
    print(f'\n{feature_set}')
    print(f'  Mean ± SD:  {aucs.mean():.3f} ± {aucs.std():.3f}')
    print(f'  Per fold:   {fold_str}')

# Feature importance plot
print('\n' + '=' * 60)
print('MEAN FEATURE IMPORTANCE (averaged across outer folds)')
print('=' * 60)

fig_imp, axes_imp = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig_imp.suptitle('Mean Feature Importance Across Outer Folds',
                 fontweight='bold', fontsize=20)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    importance_df = pd.concat(outer_feature_importance[feature_set], axis=1).fillna(0)
    mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    std_importance = importance_df.std(axis=1).reindex(mean_importance.index)

    print(f'\n{feature_set} — Top 15 features:')
    print(mean_importance.head(15).to_string())

    top_feats = mean_importance.head(15)
    top_std = std_importance.head(15)
    ax = axes_imp[col_idx]

    ax.barh(top_feats.index[::-1], top_feats.values[::-1],
            xerr=top_std.values[::-1],
            color=FEATURE_SET_COLORS[feature_set], alpha=0.85,
            capsize=3, error_kw={'elinewidth': 1, 'ecolor': 'black'})
    ax.set_title(feature_set, fontsize=18)
    ax.set_xlabel('Mean Importance', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.pdf', bbox_inches='tight')



# Sensitivity Analysis
print("Starting sensitivity analysis")
# Shape: param → feature_set → (n_outer_folds, n_fold_changes)
threshold_sens = {
    param: {fs: np.zeros((N_OUTER_FOLDS, len(FOLD_CHANGES))) for fs in FEATURE_SETS}
    for param in COST_PARAMS
}
cost_sens = {
    param: {fs: np.zeros((N_OUTER_FOLDS, len(FOLD_CHANGES))) for fs in FEATURE_SETS}
    for param in COST_PARAMS
}

# FN/FP ratio sweep — shape: feature_set → (n_outer_folds, n_ratios)
ratio_threshold_sens = {fs: np.zeros((N_OUTER_FOLDS, len(COST_RATIOS))) for fs in FEATURE_SETS}
ratio_cost_sens      = {fs: np.zeros((N_OUTER_FOLDS, len(COST_RATIOS))) for fs in FEATURE_SETS}

print("Starting sweep")
for outer_fold in range(N_OUTER_FOLDS):
    print(f"Outer fold {outer_fold + 1}")
    y_true = outer_y_test[outer_fold]

    for feature_set in FEATURE_SETS:
        y_prob      = outer_y_probs[feature_set][outer_fold]
        base_thresh = feature_set_optimal_threshold[feature_set]

        # ── Individual parameter sweep ─────────────────────────────────────────
        for param in COST_PARAMS:
            for fc_idx, multiplier in enumerate(FOLD_CHANGES):
                fn_s, fp_s, fs_costs = scale_costs(param, multiplier, BASE_COSTS)

                threshold_sens[param][feature_set][outer_fold, fc_idx] = \
                    get_optimal_threshold(
                        y_true, y_prob,
                        fn=fn_s, fp=fp_s,
                        fc=fs_costs[feature_set],
                        thresholds=THRESHOLDS
                    )

                cost_sens[param][feature_set][outer_fold, fc_idx] = \
                    total_cost(
                        y_true, y_prob,
                        threshold    = base_thresh,
                        fn_cost      = fn_s,
                        fp_cost      = fp_s,
                        feature_cost = fs_costs[feature_set]
                    )

        # ── FN/FP ratio sweep ──────────────────────────────────────────────────
        for r_idx, ratio in enumerate(COST_RATIOS):
            fp_ratio = cost_sum / (1 + ratio)
            fn_ratio = ratio * cost_sum / (1 + ratio)

            ratio_threshold_sens[feature_set][outer_fold, r_idx] = \
                get_optimal_threshold(
                    y_true, y_prob,
                    fn=fn_ratio, fp=fp_ratio,
                    fc=feature_cost[feature_set],
                    thresholds=THRESHOLDS
                )

            ratio_cost_sens[feature_set][outer_fold, r_idx] = \
                total_cost(
                    y_true, y_prob,
                    threshold    = ratio_threshold_sens[feature_set][outer_fold, r_idx],
                    fn_cost      = fn_ratio,
                    fp_cost      = fp_ratio,
                    feature_cost = feature_cost[feature_set]
                )

# Sensitivity Plots
for outer_fold in range(N_OUTER_FOLDS):

    y_true = outer_y_test[outer_fold]

    for col_idx, feature_set in enumerate(FEATURE_SETS):
        y_prob = outer_y_probs[feature_set][outer_fold]

        for param in COST_PARAMS:
            fold_thresholds = [
                get_optimal_threshold(
                    y_true, y_prob,
                    fn=fn_s, fp=fp_s,
                    fc=fs_costs[feature_set],
                    thresholds=THRESHOLDS
                )
                for fn_s, fp_s, fs_costs in [
                    scale_costs(param, m, BASE_COSTS) for m in FOLD_CHANGES
                ]
            ]

# Sensitivity plots aggregated across outer folds

# Threshold sensitivity
fig_agg_thresh, axes_agg_thresh = plt.subplots(nrows=1, ncols=3,
                                                figsize=(16, 5), sharey=True)
fig_agg_thresh.suptitle(
    'Aggregated Threshold Sensitivity (Mean ± SD across Outer Folds)',
    fontweight='bold', fontsize=14
)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    ax          = axes_agg_thresh[col_idx]
    sens_matrix = {param: threshold_sens[param][feature_set] for param in COST_PARAMS}
    plot_sensitivity_lines(
        ax, FOLD_CHANGES, sens_matrix, COST_PARAMS, SENS_COLORS,
        xlabel='Cost Multiplier', ylabel='Optimal Threshold', xscale='log'
    )
    ax.set_title(feature_set, fontsize=13)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / 'threshold_sensitivity.pdf', bbox_inches='tight')

# Total cost sensitivity
fig_agg_cost, axes_agg_cost = plt.subplots(figsize=(6, 5))

fig_agg_cost.suptitle(
    'Total Cost Sensitivity (Mean ± SD across Outer Folds)',
    fontweight='bold', fontsize=20
)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    ax          = axes_agg_cost
    sens_matrix = {param: cost_sens[param][feature_set] for param in COST_PARAMS}
    plot_sensitivity_lines(
        ax, FOLD_CHANGES, sens_matrix, [COST_PARAMS[col_idx + 2]], SENS_COLORS,
        xlabel='Cost Multiplier', ylabel='Total Cost (USD)',
        formatter=usd_formatter, xscale='log'
    )

handles, labels = axes_agg_cost.get_legend_handles_labels()
axes_agg_cost.get_legend().remove()
new_order = [0, 2, 4, 1, 3, 5]
reordered_handles = [handles[i] for i in new_order]
reordered_labels = [labels[i] for i in new_order]
by_label = dict(zip(reordered_labels, reordered_handles))
axes_agg_cost.legend(by_label.values(), by_label.keys(), fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'cost_sensitivity.pdf', bbox_inches='tight')


# FN/FP ratio sweep
# Threshold vs FN/FP ratio
fig_ratio_thresh, axes_ratio_thresh = plt.subplots(nrows=1, ncols=3,
                                                    figsize=(16, 5), sharey=True)
fig_ratio_thresh.suptitle(
    'Optimal Threshold vs FN/FP Cost Ratio (Mean ± SD across Outer Folds)',
    fontweight='bold', fontsize=20
)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    ax          = axes_ratio_thresh[col_idx]
    mean_thresh = ratio_threshold_sens[feature_set].mean(axis=0)
    std_thresh  = ratio_threshold_sens[feature_set].std(axis=0)

    ax.plot(COST_RATIOS, mean_thresh,
            color=FEATURE_SET_COLORS[feature_set], linewidth=2, label='Mean threshold')
    ax.fill_between(COST_RATIOS,
                    mean_thresh - std_thresh,
                    mean_thresh + std_thresh,
                    alpha=0.25, color=FEATURE_SET_COLORS[feature_set], label='±1 SD')
    ax.axvline(fn_cost / fp_cost, color='red', linestyle='--',
               linewidth=1, label=f'Base ratio: {fn_cost/fp_cost:.1f}')
    ax.set_xscale('log')
    ax.set_title(feature_set, fontsize=16)
    ax.set_xlabel('FN/FP Cost Ratio', fontsize=16)
    ax.set_ylabel('Optimal Threshold', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'ratio_threshold_sensitivity.pdf', bbox_inches='tight')

# Total cost vs FN/FP ratio
fig_ratio_cost, axes_ratio_cost = plt.subplots(nrows=1, ncols=3,
                                                figsize=(16, 5), sharey=False)
fig_ratio_cost.suptitle(
    'Total Cost vs FN/FP Cost Ratio (Mean ± SD across Outer Folds)',
    fontweight='bold', fontsize=20
)

for col_idx, feature_set in enumerate(FEATURE_SETS):
    ax        = axes_ratio_cost[col_idx]
    mean_cost = ratio_cost_sens[feature_set].mean(axis=0)
    std_cost  = ratio_cost_sens[feature_set].std(axis=0)

    ax.plot(COST_RATIOS, mean_cost,
            color=FEATURE_SET_COLORS[feature_set], linewidth=2, label='Mean cost')
    ax.fill_between(COST_RATIOS,
                    mean_cost - std_cost,
                    mean_cost + std_cost,
                    alpha=0.25, color=FEATURE_SET_COLORS[feature_set], label='±1 SD')
    ax.axvline(fn_cost / fp_cost, color='red', linestyle='--',
               linewidth=1, label=f'Base ratio: {fn_cost/fp_cost:.1f}')
    ax.set_xscale('log')
    ax.set_title(feature_set, fontsize=16)
    ax.set_xlabel('FN/FP Cost Ratio', fontsize=16)
    ax.set_ylabel('Total Cost (USD)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_formatter(usd_formatter)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'ratio_cost_sensitivity.pdf', bbox_inches='tight')