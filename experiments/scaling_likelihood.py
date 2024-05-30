'''
Fit individualized scaling laws using the maximum likelihood estimator.
'''

import tqdm
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from data_scaling import LikelihoodEstimator


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
deltas = torch.tensor(results['samples'])
cardinalities = torch.tensor(results['cards'])

# Load validation results.
filename = f'data_files/{dataset}/mc_validation-inds=1000-samples=1000' \
           f'-min={min_cardinality}-max={max_cardinality}-num=10-model={model}-seed=0.npy'
results = np.load(filename, allow_pickle=True).item()
relevant_inds = results['relevant_inds']

# Setup for running estimator.
estimator = LikelihoodEstimator()


# Fit estimator for all points and all samples.
results = []
for i in tqdm.tqdm(range(len(deltas))):
    # Setup.
    delta = deltas[i]
    cards = cardinalities[i]

    # Run estimator and store results.
    c, alpha, sigma, beta = estimator(delta, cards)
    results.append({'alpha': alpha, 'c': c, 'sigma': sigma, 'beta': beta})

# Save dataframe as csv.
df = pd.DataFrame(results)
filename = f'model_results/{dataset}/scaling_all-model={model}.csv'
df.to_csv(filename)


# Fit estimator for validation inds with different numbers of samples.
num_samples = [5] + list(range(10, 100, 10)) + list(range(100, 501, 50))
results = {num: [] for num in num_samples}
for i in tqdm.tqdm(relevant_inds):
    # Setup.
    delta = deltas[i]
    cards = cardinalities[i]

    # Fit estimator with different numbers of samples.
    c_list = []
    alpha_list = []
    for num in num_samples:
        # Run estimator.
        c, alpha, sigma, beta = estimator(delta[:num], cards[:num])

        # Store results.
        c_list.append(c)
        alpha_list.append(alpha)
        results[num].append({'alpha': alpha, 'c': c, 'sigma': sigma, 'beta': beta})

    # # Plot samples with fitted estimators.
    # x = np.linspace(min_cardinality, max_cardinality, 1000)
    # y = c / x ** alpha
    # fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

    # # Fit with all samples.
    # ax = axarr[0]
    # ax.scatter(cards, delta, alpha=0.2)
    # ax.plot(x, y, label='Scaling Law', color='green')
    # ax.text((max(cards) + min(cards)) / 2, (max(delta) + min(delta)) / 2,
    #         rf'$c$ = {c:.2f}, $\alpha$ = {alpha:.2f}'
    #         + '\n'
    #         + rf'$\sigma$ = {sigma:.2f}, $\beta$ = {beta:.2f}')
    # ax.set_xlabel(r'Cardinality $|\mathcal{D}|$')
    # ax.set_ylabel(r'Marginal Contribution $\Delta(z, \mathcal{D})$')
    # ax.set_title('Scaling Law')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # # c estimates with different numbers of samples.
    # ax = axarr[1]
    # ax.plot(num_samples, c_list, marker='o', color='green')
    # ax.set_xlabel(r'# $\Delta(z, \mathcal{D})$ Samples')
    # ax.set_ylabel(r'$c$')
    # ax.set_title(r'$c$ Estimator Convergence')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # # alpha estimates with different numbers of samples.
    # ax = axarr[2]
    # ax.plot(num_samples, alpha_list, marker='o', color='green')
    # ax.set_xlabel(r'# $\Delta(z, \mathcal{D})$ Samples')
    # ax.set_ylabel(r'$\alpha$')
    # ax.set_title(r'$\alpha$ Estimator Convergence')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.tight_layout()
    # plt.savefig(f'figures/scaling-{dataset}-{model}.pdf')

    # # Plot samples with fitted estimators.
    # x = np.linspace(min_cardinality, max_cardinality, 1000)
    # y = c / x ** alpha
    # fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    # # Plot with all samples.
    # ax.scatter(cards, delta, alpha=0.2)
    # ax.plot(x, y, label='Scaling Law', color='green')
    # ax.text((max(cards) + min(cards)) / 2, (max(delta) + min(delta)) / 2,
    #         rf'$c$ = {c:.2f}, $\alpha$ = {alpha:.2f}'
    #         + '\n'
    #         + rf'$\sigma$ = {sigma:.2f}, $\beta$ = {beta:.2f}')
    # ax.set_xlabel(r'Cardinality $|\mathcal{D}|$')
    # ax.set_ylabel(r'Marginal Contribution $\Delta(z, \mathcal{D})$')
    # ax.set_title('Scaling Law')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # plt.tight_layout()
    # plt.savefig(f'figures/scaling_example-{dataset}-{model}.pdf')
    # break

# Save results.
df_dict = {num: pd.DataFrame(results[num]) for num in num_samples}
filename = f'model_results/{dataset}/scaling_validation-model={model}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(df_dict, f)
