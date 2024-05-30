import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_scaling import AggregateLikelihoodEstimator
from glob import glob


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-min_cardinality', type=int)
parser.add_argument('-max_cardinality', type=int)
parser.add_argument('-num_cardinalities', type=int)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
num_cardinalities = args.num_cardinalities
model = args.model


# Find and aggregate results.
regex = f'perf-samples=*-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}' \
        f'-model={model}-seed=*.npy'
filenames = glob(os.path.join('data_files', dataset, regex))
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
cards = results[0]['cards']
performance = {card: [] for card in cards}
for result in results:
    assert np.all(result['cards'] == cards)
    for card in cards:
        performance[card] += result['performance'][card]

# Correct loss sign.
for card in cards:
    performance[card] = [-loss for loss in performance[card]]


# Fit scaling law estimator to mean performance.
error = torch.tensor([np.mean(performance[card]) for card in cards]).float()
variance = torch.tensor([np.var(performance[card]) / np.sqrt(len(performance[card])) for card in cards]).float()
cardinalities = torch.tensor(cards).float()
estimator = AggregateLikelihoodEstimator(lr=0.05, num_steps=1000, verbose=True)
num_trials = len(performance[cards[0]])

# Populate loss grid.
alpha_grid = torch.linspace(0.01, 2.0, 100)
beta_grid = torch.linspace(0.0, 5.0, 100)
loss_grid = torch.zeros((len(alpha_grid), len(beta_grid)))
start = time.time()
for i, alpha in enumerate(alpha_grid):
    for j, beta in enumerate(beta_grid):
        eps, c, sigma = estimator.calculate_optimal_eps_c_sigma(alpha, beta, error, cardinalities)
        loss_grid[-j, i] = estimator.calculate_objective(eps, c, alpha, sigma, beta, error, cardinalities)
end = time.time()
print(f'Loss grid time: {end - start}')

# Fit estimator.
start = time.time()
eps, c, alpha, sigma, beta = estimator(error, cardinalities)
end = time.time()
print(f'Estimator time: {end - start}')


# Plot loss grid to visualize fitting accuracy.
plt.figure()
plt.imshow(loss_grid, extent=[alpha_grid.min(), alpha_grid.max(), beta_grid.min(), beta_grid.max()], aspect='auto')
plt.colorbar()
plt.plot([0.5, alpha], [3, beta], color='red', linestyle=':')
plt.scatter(alpha, beta, marker='x', color='red')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title('Loss Surface')
plt.tight_layout()
plt.savefig(f'figures/loss_grid-{dataset}-{model}.pdf')


# Plot aggregate scaling law.
fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
x = np.linspace(min_cardinality, max_cardinality, 1000)
y = eps + c / (x ** alpha)
y2 = (sigma ** 2) / (x ** beta)

# Original scale.
ax = axarr[0]
ax.plot(cards, error, marker='o', label='Empirical')
# ax.errorbar(cards, error, yerr=np.sqrt(variance), capsize=5, marker='o', label='Empirical')
ax.plot(x, y, color='green', label='Scaling Law')
ax.text((max(cards) + min(cards)) / 2, (max(error) + min(error)) / 2,
        rf'$\epsilon$ = {eps:.2f}, $c$ = {c:.2f}, $\alpha$ = {alpha:.2f}'
        + '\n'
        + rf'$\sigma$ = {sigma:.2f}, $\beta$ = {beta:.2f}')
ax.set_title(f'{dataset.upper()} Performance vs. Cardinality')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Loss')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Log-log scale.
ax = axarr[1]
ax.plot(np.log10(cards), np.log10(error - eps), marker='o', label='Empirical')
ax.plot(np.log10(x), np.log10(y - eps), color='green', label='Scaling Law')
ax.set_title(f'{dataset.upper()} Log Performance vs. Log Cardinality')
ax.set_xlabel('Log(# Datapoints)')
ax.set_ylabel(r'Log(Loss - $\epsilon$)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Log-variance.
ax = axarr[2]
ax.plot(np.log10(cards), np.log10(variance), marker='o', label='Empirical')
ax.plot(np.log10(x), np.log10(y2), color='green', label='Scaling Law')
ax.set_title(f'{dataset.upper()} Log Variance vs. Log Cardinality')
ax.set_xlabel('Log(# Datapoints)')
ax.set_ylabel('Log(Variance)')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save plot.
plt.tight_layout()
plt.savefig(f'figures/perf-{dataset}-{model}.pdf')
