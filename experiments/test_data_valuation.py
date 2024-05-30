import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_scaling.valuation import calculate_shapley_value
from data_scaling.utils import generate_metrics
from glob import glob


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
model = args.model


# Load samples.
filename = f'data_files/{dataset}/mc_combined-inds=-1' \
           f'-min={min_cardinality}-max={max_cardinality}-model={model}.npy'
results = np.load(filename, allow_pickle=True).item()
deltas = results['samples']
cardinalities = results['cards']

# Load validation samples.
filename = f'data_files/{dataset}/mc_combined-inds=1000' \
           f'-min={min_cardinality}-max={max_cardinality}-model={model}.npy'
results = np.load(filename, allow_pickle=True).item()
validation_deltas = results['samples']
validation_cardinalities = results['cards']
relevant_inds = results['relevant_inds']

# Load maximum likelihood scaling law parameters for all points.
filename = f'model_results/{dataset}/scaling_all-model={model}.csv'
scaling_df = pd.read_csv(filename, index_col=0)

# Load maximum likelihood scaling law parameters for validation points.
filename = f'model_results/{dataset}/scaling_validation-model={model}.pkl'
with open(filename, 'rb') as f:
    scaling_df_validation = pickle.load(f)

# Load amortized scaling law parameters for all points.
amortized_filenames = glob(f'model_results/{dataset}/amortized-samples=*-model={model}.csv')
amortized_df_dict = {}
for filename in amortized_filenames:
    num_samples = int(filename.split('=')[1].split('-')[0])
    amortized_df_dict[num_samples] = pd.read_csv(filename, index_col=0)
amortized_df = amortized_df_dict[500]


# NOTE we don't calculate any metrics using these results
# # Generate data valuation results for entire dataset.
# num_samples = list(scaling_df_validation.keys())
# results = {}

# # Monte Carlo.
# for num in num_samples:
#     results[f'Monte Carlo {num}'] = np.mean(deltas[:, :num], axis=1)

# # Scaling law estimation.
# results['Likelihood'] = np.array([
#     calculate_shapley_value(scaling_df['c'][i], scaling_df['alpha'][i], min_cardinality, max_cardinality)
#     for i in range(len(scaling_df))])

# # Amortized scaling law estimation.
# results['Amortized'] = np.array([
#     calculate_shapley_value(amortized_df['c'][i], amortized_df['alpha'][i], min_cardinality, max_cardinality)
#     for i in range(len(amortized_df))]
# )

# # Save results.
# df = pd.DataFrame(results)
# filename = f'model_results/{dataset}/valuation-model={model}.csv'
# df.to_csv(filename)


# Generate data valuation results for validation inds only.
ground_truth = np.mean(validation_deltas, axis=1)
results = {'Ground Truth': ground_truth}

# Monte Carlo.
for num in num_samples:
    results[f'Monte Carlo {num}'] = np.mean(deltas[relevant_inds, :num], axis=1)

# Scaling law estimation.
for num in list(scaling_df_validation.keys()):
    results[f'Likelihood {num}'] = np.array([
        calculate_shapley_value(scaling_df_validation[num]['c'][i],
                                scaling_df_validation[num]['alpha'][i],
                                min_cardinality,
                                max_cardinality)
        for i in range(len(scaling_df_validation[num]))])

# Amortized scaling law estimation.
for num in list(amortized_df_dict.keys()):
    results[f'Amortized {num}'] = np.array([
        calculate_shapley_value(amortized_df_dict[num]['c'][i],
                                amortized_df_dict[num]['alpha'][i],
                                min_cardinality,
                                max_cardinality)
        for i in relevant_inds]
    )

# Save results for validation inds only.
validation_df = pd.DataFrame(results)
filename = f'model_results/{dataset}/valuation_val-model={model}.csv'
validation_df.to_csv(filename)


# Generate accuracy metrics.
metrics = {}

# Monte Carlo.
for num in num_samples:
    metrics[f'Monte Carlo {num}'] = generate_metrics(validation_df[f'Monte Carlo {num}'].values, ground_truth)

# Scaling law estimation.
for num in num_samples:
    metrics[f'Likelihood {num}'] = generate_metrics(validation_df[f'Likelihood {num}'].values, ground_truth)

# Amortized scaling law estimation.
for num in np.sort(list(amortized_df_dict.keys())):
    metrics[f'Amortized {num}'] = generate_metrics(validation_df[f'Amortized {num}'].values, ground_truth)

# Save metrics dataframe.
metrics_df = pd.DataFrame(metrics)
filename = f'model_results/{dataset}/valuation_metrics-model={model}.csv'
metrics_df.to_csv(filename)

# Plot metrics.
fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
display_samples = [num for num in num_samples if 10 <= num <= 100]
# name = dataset
# name = dataset.upper()
name = dataset.title()

# Explained variance.
ax = axarr[0]
ax.plot(display_samples, [metrics_df[f'Likelihood {num}']['expl_var'] for num in display_samples],
        label='Likelihood', marker='o', color='C0')
ax.plot(display_samples, [metrics_df[f'Amortized {num}']['expl_var'] for num in display_samples],
        label='Amortized', marker='o', color='tab:purple')
ax.plot(display_samples, [metrics_df[f'Monte Carlo {num}']['expl_var'] for num in display_samples],
        label='Monte Carlo', marker='o', color='C1')
ax.set_ylabel(r'$R^2$')
ax.set_xlabel('# Samples')
ax.set_title(rf'{name} $R^2$ Convergence')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.5, 1.05])

# Correlation.
ax = axarr[1]
ax.plot(display_samples, [metrics_df[f'Likelihood {num}']['corr'] for num in display_samples],
        label='Likelihood', marker='o', color='C0')
ax.plot(display_samples, [metrics_df[f'Amortized {num}']['corr'] for num in display_samples],
        label='Amortized', marker='o', color='tab:purple')
ax.plot(display_samples, [metrics_df[f'Monte Carlo {num}']['corr'] for num in display_samples],
        label='Monte Carlo', marker='o', color='C1')
ax.set_ylabel('Pearson Correlation')
ax.set_xlabel('# Samples')
ax.set_title(f'{name} Correlation Convergence')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.5, 1.05])

# Rank correlation.
ax = axarr[2]
ax.plot(display_samples, [metrics_df[f'Likelihood {num}']['spearman'] for num in display_samples],
        label='Likelihood', marker='o', color='C0')
ax.plot(display_samples, [metrics_df[f'Amortized {num}']['spearman'] for num in display_samples],
        label='Amortized', marker='o', color='tab:purple')
ax.plot(display_samples, [metrics_df[f'Monte Carlo {num}']['spearman'] for num in display_samples],
        label='Monte Carlo', marker='o', color='C1')
ax.set_ylabel('Spearman Correlation')
ax.set_xlabel('# Samples')
ax.set_title(f'{name} Rank Correlation Convergence')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0.5, 1.05])

plt.tight_layout()
filename = f'figures/valuation-{dataset}-{model}.pdf'
plt.savefig(filename)
