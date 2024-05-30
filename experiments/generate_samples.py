'''
Script to generate marginal contribution samples. This script uses
cardinalities within a given range, whereas generate_samples_validation.py
uses a staggered set of cardinalities.

Recommended usage is to generate samples in parallel using multiple jobs, then
combine into a single file using combine_samples.py.
'''

import os
import argparse
import numpy as np
from utils import get_experiment_mediator, DistributionalSampler


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-num_inds', type=int)
parser.add_argument('-num_samples', type=int)
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])
parser.add_argument('-seed', type=int)

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
num_inds = args.num_inds
num_samples = args.num_samples
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
model = args.model
seed = args.seed

# Make sure result doesn't exist.
filename = f'data_files/{dataset}/mc-inds={num_inds}-samples={num_samples}' \
           f'-min={min_cardinality}-max={max_cardinality}' \
           f'-model={model}-seed={seed}.npy'
if os.path.exists(filename):
    exit()

# Get experiment mediator and set up marginal contribution sampler.
exper_med = get_experiment_mediator(dataset, model)
sampler = DistributionalSampler(
    num_samples=num_samples, min_cardinality=min_cardinality, max_cardinality=max_cardinality, random_state=seed)
sampler.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)

# Generate subset of relevant indices.
num_points = len(exper_med.fetcher.x_train)
print(f'Generating {num_inds}/{num_points} relevant indices...')
relevant_inds = np.arange(num_points) if (num_inds == -1) else np.arange(num_inds)

# Generate samples.
sampler.compute_marginal_contribution(relevant_inds=relevant_inds, **exper_med.train_kwargs)

# Save results for later aggregation.
results = {
    'estimates': sampler.total_contribution[relevant_inds] / sampler.total_count[relevant_inds],
    'total_contribution': sampler.total_contribution[relevant_inds],
    'total_count': sampler.total_count[relevant_inds],
    'relevant_inds': relevant_inds,
    'cards': sampler.cards[relevant_inds],
    'samples': sampler.samples[relevant_inds]
}
filename = f'data_files/{dataset}/mc-inds={num_inds}-samples={num_samples}' \
           f'-min={min_cardinality}-max={max_cardinality}' \
           f'-model={model}-seed={seed}.npy'
np.save(filename, results)
print(f'Saved results to {filename}')
