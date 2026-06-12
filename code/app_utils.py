import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def total_cost(y_true, y_prob, threshold, fn_cost, fp_cost,
               feature_cost):
    """
    Computes total cost in USD of a classification decision including
    feature set costs, error costs, and surgery costs for all positive cases.

    :param y_true: Ground truth binary labels (0 = no appendicitis, 1 = appendicitis).
    :type y_true: array-like of shape (n_samples,)
    :param y_prob: Predicted probabilities of appendicitis.
    :type y_prob: array-like of shape (n_samples,)
    :param threshold: Decision threshold for converting probabilities to binary predictions.
    :type threshold: float
    :param fn_cost: Cost of a false negative in USD (missed appendicitis leading to perforation).
    :type fn_cost: float
    :param fp_cost: Cost of a false positive in USD (unnecessary appendectomy).
    :type fp_cost: float
    :param feature_cost: Cost per patient in USD of the diagnostic workup required
                         to obtain the feature set. Use ``300`` for clinical only,
                         ``750`` for clinical and laboratory, and ``1500`` for
                         clinical, laboratory, and ultrasound.
    :type feature_cost: float
    :return: Total cost in USD.
    :rtype: float
    """
    y_pred = (y_prob >= threshold).astype(int)
    n      = len(y_true)

    fn = ((y_pred == 0) & (y_true == 1)).sum()   # missed appendicitis
    fp = ((y_pred == 1) & (y_true == 0)).sum()   # unnecessary surgery

    return (
        (feature_cost * n)      +   # everyone gets the workup
        (fn_cost      * fn)     +   # cost of missed cases
        (fp_cost      * fp)        # cost of unnecessary surgery
    )


def mask_feature_sets(X_data: pd.DataFrame, descriptors: pd.DataFrame):
    """
    Filters X_data columns by variable group to produce three feature sets
    for clinical, clinical+lab, and clinical+lab+ultrasound models.

    :param X_data: Preprocessed feature dataframe with all variables.
    :param descriptors: Data Summary sheet dataframe containing at minimum
                        ``Variable Group`` and ``Variable Name in Data Files``
                        columns. For example: ``pd.read_excel('app_data.xlsx', sheet_name='Data Summary')``.
    :return: Tuple of three dataframes (X_clinical, X_lab, X_us).
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    # Forward fill merged cells in Variable Group column
    if descriptors['Variable Group'].isna().any():
        descriptors['Variable Group'] = descriptors['Variable Group'].ffill()

    # Build group → column name mapping
    variable_groups = ['Demographic / Other', 'Clinical', 'Scoring', 'Laboratory', 'Ultrasound']
    variable_dict = {group: [] for group in variable_groups}

    for group in variable_groups:
        variables = descriptors[descriptors['Variable Group'] == group]
        variables = variables['Variable Name in Data Files'].tolist()
        variable_dict[group] = variables

    # Define cumulative column sets per model
    clinical_cols = variable_dict['Demographic / Other'] + variable_dict['Clinical']
    lab_cols      = clinical_cols + variable_dict['Scoring'] + variable_dict['Laboratory']
    us_cols       = lab_cols + variable_dict['Ultrasound']

    # Filter to only columns present in X_data (guards against dropped columns)
    def filter_cols(cols):
        return X_data[[col for col in cols if col in X_data.columns]]

    X_data_clinical = filter_cols(clinical_cols)
    X_data_lab      = filter_cols(lab_cols)
    X_data_us       = filter_cols(us_cols)

    return X_data_clinical, X_data_lab, X_data_us


def compute_costs_over_thresholds(y_true, y_prob, fn, fp, fc, thresholds):
    """Compute total cost at each threshold value."""
    return np.array([
        total_cost(y_true, y_prob, threshold=t, fn_cost=fn, fp_cost=fp, feature_cost=fc)
        for t in thresholds
    ])


def get_optimal_threshold(y_true, y_prob, fn, fp, fc, thresholds):
    """Return threshold that minimizes total cost."""
    costs = compute_costs_over_thresholds(y_true, y_prob, fn, fp, fc, thresholds)
    return thresholds[np.argmin(costs)]


def scale_costs(param, multiplier, base_costs):
    """
    Return (fn, fp, fs_costs) with one parameter scaled by multiplier.
    multiplier > 1 increases the cost, < 1 decreases it.

    :param param: Name of the cost parameter to scale.
    :param multiplier: Multiplicative scale factor (0.1 to 10).
    :param base_costs: Dict of base cost values.
    :return: Tuple of (fn_cost, fp_cost, feature_set_costs dict).
    """
    fn = base_costs['FN cost']         * (multiplier if param == 'FN cost'         else 1)
    fp = base_costs['FP cost']         * (multiplier if param == 'FP cost'         else 1)
    cc = base_costs['Clinical cost']   * (multiplier if param == 'Clinical cost'   else 1)
    lc = base_costs['Laboratory cost'] * (multiplier if param == 'Laboratory cost' else 1)
    uc = base_costs['Ultrasound cost'] * (multiplier if param == 'Ultrasound cost' else 1)
    return fn, fp, {'Clinical': cc, 'Laboratory': lc, 'Ultrasound': uc}


def plot_cost_curves_grid(axes, inner_fold, col_idx, thresholds, costs,
                          optimal_threshold, optimal_cost, feature_set, feature_set_colors,
                          n_inner_folds):
    """Plot a single cost curve in the per-fold grid."""
    ax = axes[inner_fold, col_idx]
    ax.plot(thresholds, costs, color=feature_set_colors[feature_set], linewidth=1.5)
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=1,
               label=f'Optimal: {optimal_threshold:.2f}')
    ax.scatter(optimal_threshold, optimal_cost, color='red', zorder=5)
    if inner_fold == 0:
        ax.set_title(feature_set)
    if col_idx == 0:
        ax.set_ylabel(f'Inner Fold {inner_fold + 1}\nCost')
    if inner_fold == n_inner_folds - 1:
        ax.set_xlabel('Threshold')
    ax.legend(fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def plot_aggregated_mean_curve(ax, thresholds, mean_costs, std_costs,
                               optimal_threshold, optimal_cost,
                               feature_set, feature_set_colors, ylabel='Cost (USD)',
                               formatter=None):
    """Plot mean cost curve with shaded SD band and optimal threshold marker."""
    color = feature_set_colors[feature_set]
    ax.plot(thresholds, mean_costs, color=color, linewidth=2, label='Mean cost')
    ax.fill_between(thresholds,
                    mean_costs - std_costs,
                    mean_costs + std_costs,
                    alpha=0.25, color=color, label='±1 SD')
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=1,
               label=f'Optimal: {optimal_threshold:.2f}')
    ax.scatter(optimal_threshold, optimal_cost, color='red', zorder=5)
    ax.set_title(feature_set, fontsize=13)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if formatter:
        ax.yaxis.set_major_formatter(formatter)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


def plot_sensitivity_lines(ax, x_vals, sens_matrix, cost_params,
                           sens_colors, xlabel, ylabel,
                           formatter=None, xscale='log'):
    """
    Plot sensitivity lines with shaded SD bands for each cost parameter.

    :param ax: Matplotlib axis to plot on.
    :param x_vals: X-axis values (fold changes or cost ratios).
    :param sens_matrix: Dict of param → (n_folds, n_x_vals) arrays.
    :param cost_params: List of parameter names to plot.
    :param sens_colors: Dict mapping parameter names to colors.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param formatter: Optional matplotlib formatter for y-axis.
    :param xscale: X-axis scale ('log' or 'linear').
    """
    for param in cost_params:
        mean_vals = sens_matrix[param].mean(axis=0)
        std_vals  = sens_matrix[param].std(axis=0)
        ax.plot(x_vals, mean_vals,
                color=sens_colors[param], linewidth=2, label=param)
        ax.fill_between(x_vals,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        alpha=0.15, color=sens_colors[param])
    ax.axvline(1, color='grey', linestyle=':', linewidth=1,
               alpha=0.7, label='Base value')
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    if formatter:
        ax.yaxis.set_major_formatter(formatter)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

