import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from data_scaling import LogEstimator
from data_scaling.utils import generate_metrics


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-num_cardinalities', type=int, default=10)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
num_cardinalities = args.num_cardinalities
model = args.model


# Load scaling law estimates for validation points.
filename = f'model_results/{dataset}/scaling_validation-model={model}.pkl'
with open(filename, 'rb') as f:
    df_dict = pickle.load(f)
num_samples = np.sort(list(df_dict.keys()))

# Load validation results.
filename = f'data_files/{dataset}/mc_validation-inds=1000-samples=1000' \
           f'-min={min_cardinality}-max={max_cardinality}-num=10-model={model}-seed=0.npy'
# filename = f'data_files/{dataset}/mc_validation_combined-inds=1000-min={min_cardinality}-max={max_cardinality}' \
#            f'-num={num_cardinalities}-model={model}.npy'
results = np.load(filename, allow_pickle=True).item()
deltas = results['samples']
cardinalities = results['cards']
relevant_inds = results['relevant_inds']


# Calculate validation ground truth.
reference_results = []
unique_cardinalities = np.unique(cardinalities).astype(int)
estimator = LogEstimator()
for i in range(len(relevant_inds)):
    c, alpha, r2, mean_values, var_values = estimator(deltas[i], cardinalities[i])
    outputs = {'c': c, 'alpha': alpha, 'r2': r2}
    for cardinality, mean in zip(unique_cardinalities, mean_values):
        outputs[f'mean_{cardinality}'] = mean
    reference_results.append(outputs)
reference_df = pd.DataFrame(reference_results)

# Add log |c| entry.
reference_df['log_c'] = np.log(np.abs(reference_df['c']) + 1e-10)
for num in num_samples:
    df_dict[num]['log_c'] = np.log(np.abs(df_dict[num]['c']) + 1e-10)


# Plot accuracy of scaling law predictions (four cardinalities).
fig, axarr = plt.subplots(1, 4, figsize=(20, 4))

for i, cardinality in enumerate([100, 215, 599, 1000]):
    ax = axarr[i]
    mean_preds = df_dict[max(num_samples)]['c'] * (cardinality ** (- df_dict[max(num_samples)]['alpha']))
    r2 = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
    metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
    corr, spearman = metrics['corr'], metrics['spearman']
    ax.scatter(mean_preds, reference_df[f'mean_{cardinality}'],
               label=rf'$R^2 = {r2:.2f}$, $\rho = {corr:.2f}$, $\tau = {spearman:.2f}$', alpha=0.2)
    ax.set_xlabel(r'$c / k^{\alpha}$ (Scaling Law)')
    ax.set_ylabel(r'$\psi_k$ (Empirical)')
    ax.set_title(f'Scaling Law Accuracy @ {cardinality}')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/scaling_metrics_horizontal-{dataset}-{model}.pdf')


# Plot accuracy of scaling law predictions (all cardinalities).
fig, axarr = plt.subplots(len(unique_cardinalities), 1, figsize=(5, 4 * len(unique_cardinalities)))

for i, cardinality in enumerate(unique_cardinalities):
    ax = axarr[i]
    mean_preds = df_dict[max(num_samples)]['c'] * (cardinality ** (- df_dict[max(num_samples)]['alpha']))
    r2 = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
    metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
    corr, spearman = metrics['corr'], metrics['spearman']
    ax.scatter(mean_preds, reference_df[f'mean_{cardinality}'],
               label=rf'$R^2 = {r2:.2f}$, $\rho = {corr:.2f}$, $\tau = {spearman:.2f}$', alpha=0.2)
    ax.set_xlabel(r'$c / k^{\alpha}$ (Scaling Law)')
    ax.set_ylabel(r'$\psi_k$ (Empirical)')
    ax.set_title(f'Scaling Law Accuracy @ {cardinality}')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/scaling_metrics-{dataset}-{model}.pdf')


# Plot convergence of scaling law estimates.
fig, axarr = plt.subplots(2, 3, figsize=(15, 8))

# Generate interpolation metrics.
display_samples = list(filter(lambda num: num <= 100, df_dict.keys()))
display_cardinalities = [100, 359, 599, 1000]
metrics_dict = {cardinality: {num: None for num in display_samples} for cardinality in display_cardinalities}
for cardinality in display_cardinalities:
    for num in display_samples:
        mean_preds = df_dict[num]['c'] * (cardinality ** (- df_dict[num]['alpha']))
        metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
        metrics['r2'] = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
        metrics_dict[cardinality][num] = metrics
metrics_dict = {cardinality: pd.DataFrame(metrics_dict[cardinality]).T for cardinality in metrics_dict}

# R^2.
ax = axarr[0, 0]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['r2'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Blues(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel(r'$R^2$')
ax.set_title(r'Scaling Law $R^2$ (Interpolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

# Pearson correlation.
ax = axarr[0, 1]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['corr'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Blues(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Scaling Law Correlation (Interpolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

# Spearman correlation.
ax = axarr[0, 2]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['spearman'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Blues(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Scaling Law Rank Correlation (Interpolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

# Generate extrapolation metrics.
display_samples = list(filter(lambda num: num <= 100, df_dict.keys()))
display_cardinalities = list(filter(lambda cardinality: cardinality >= 1000, unique_cardinalities))
metrics_dict = {cardinality: {num: None for num in display_samples} for cardinality in display_cardinalities}
for cardinality in display_cardinalities:
    for num in display_samples:
        mean_preds = df_dict[num]['c'] * (cardinality ** (- df_dict[num]['alpha']))
        metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
        metrics['r2'] = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
        metrics_dict[cardinality][num] = metrics
metrics_dict = {k: pd.DataFrame(metrics_dict[k]).T for k in metrics_dict}

# R^2.
ax = axarr[1, 0]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['r2'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Greens(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel(r'$R^2$')
ax.set_title(r'Scaling Law $R^2$ (Extrapolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

# Pearson correlation.
ax = axarr[1, 1]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['corr'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Greens(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Scaling Law Correlation (Extrapolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

# Spearman correlation.
ax = axarr[1, 2]
for i, cardinality in enumerate(display_cardinalities):
    ax.plot(display_samples, metrics_dict[cardinality]['spearman'], marker='o', label=rf'$k = {cardinality}$',
            color=plt.cm.Greens(0.2 + 0.8 * i / len(display_cardinalities)))
ax.set_xlabel('# Samples')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Scaling Law Rank Correlation (Extrapolation)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(f'figures/scaling_convergence-{dataset}-{model}.pdf')


# Plot main scaling law accuracy metrics.
fig, axarr = plt.subplots(1, 3, figsize=(15, 4))

# Generate metrics for maximum likelihood and amortized scaling law estimates.
likelihood_samples = [50, 100, 500]
amortized_samples = [10, 20, 30]
display_cardinalities = unique_cardinalities
# display_cardinalities = list(filter(lambda num: num <= 1000, unique_cardinalities))
metrics_dict = {cardinality: {} for cardinality in display_cardinalities}
for cardinality in display_cardinalities:
    for num in likelihood_samples:
        mean_preds = df_dict[num]['c'] * (cardinality ** (- df_dict[num]['alpha']))
        metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
        metrics['r2'] = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
        metrics_dict[cardinality][f'Likelihood ({num})'] = metrics
    for num in amortized_samples:
        amortized_df = pd.read_csv(f'model_results/{dataset}/amortized-samples={num}-model={model}.csv', index_col=0)
        mean_preds = amortized_df['c'] * (cardinality ** (- amortized_df['alpha']))
        mean_preds = mean_preds[relevant_inds]
        metrics = generate_metrics(reference_df[f'mean_{cardinality}'].values, mean_preds)
        metrics['r2'] = r2_score(reference_df[f'mean_{cardinality}'], mean_preds)
        metrics_dict[cardinality][f'Amortized ({num})'] = metrics
metrics_dict = {cardinality: pd.DataFrame(metrics_dict[cardinality]) for cardinality in metrics_dict}

# R^2.
ax = axarr[0]
for i, num in enumerate(likelihood_samples):
    name = f'Likelihood ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['r2'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Blues(0.5 + 0.5 * i / len(likelihood_samples)))
for i, num in enumerate(amortized_samples):
    name = f'Amortized ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['r2'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Purples(0.3 + 0.7 * i / len(amortized_samples)))
ax.axvline(1000, color='black', linestyle=':', label='Extrapolation Boundary')
ax.set_xlabel(r'Cardinality $k$')
ax.set_ylabel(r'$R^2$')
ax.set_title(r'Scaling Law $R^2$ Accuracy')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xscale('log')
ax.set_ylim(0, 1.1)

# Pearson correlation.
ax = axarr[1]
for i, num in enumerate(likelihood_samples):
    name = f'Likelihood ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['corr'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Blues(0.5 + 0.5 * i / len(likelihood_samples)))
for i, num in enumerate(amortized_samples):
    name = f'Amortized ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['corr'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Purples(0.3 + 0.7 * i / len(amortized_samples)))
ax.axvline(1000, color='black', linestyle=':', label='Extrapolation Boundary')
ax.set_xlabel(r'Cardinality $k$')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Scaling Law Correlation Accuracy')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xscale('log')
ax.set_ylim(0, 1.1)

# Spearman correlation.
ax = axarr[2]
for i, num in enumerate(likelihood_samples):
    name = f'Likelihood ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['spearman'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Blues(0.5 + 0.5 * i / len(likelihood_samples)))
for i, num in enumerate(amortized_samples):
    name = f'Amortized ({num})'
    ax.plot(display_cardinalities,
            [metrics_dict[cardinality][name]['spearman'] for cardinality in display_cardinalities],
            label=name, marker='o', color=plt.cm.Purples(0.3 + 0.7 * i / len(amortized_samples)))
ax.axvline(1000, color='black', linestyle=':', label='Extrapolation Boundary')
ax.set_xlabel(r'Cardinality $k$')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Scaling Law Rank Correlation Accuracy')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xscale('log')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(f'figures/scaling_final-{dataset}-{model}.pdf')
