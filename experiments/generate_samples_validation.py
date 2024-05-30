'''
Script to generate marginal contribution samples. This script uses a staggered
set of cardinalities within a given range, whereas generate_samples.py uses all
cardinalities in the range.

Recommended usage is either to generate all samples in a single run, or
generate samples in parallel using multiple jobs and combine into a single file
using combine_samples.py.
'''

import argparse
import numpy as np
from utils import get_experiment_mediator, DistributionalCardinalityListSampler


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-num_inds', type=int, default=1000)
parser.add_argument('-num_samples', type=int)
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-num_cardinalities', type=int, default=10)
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
num_cardinalities = args.num_cardinalities
model = args.model
seed = args.seed

# Get experiment mediator and set up cardinalities.
exper_med = get_experiment_mediator(dataset, model)
cardinalities = np.geomspace(min_cardinality, max_cardinality, num_cardinalities).astype(int)
num_points = len(exper_med.fetcher.x_train)
# cardinalities = (
#     list(cardinalities)
#     + list(filter(lambda num: num < num_points, [2500, 5000, 7500, 10000, 15000, 20000, 25000]))
# )

# Set up marginal contribution sampler.
sampler = DistributionalCardinalityListSampler(
    num_samples=num_samples, cardinalities=cardinalities, random_state=seed)
sampler.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)

# Generate subset of relevant indices.
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
filename = f'data_files/{dataset}/mc_validation-inds={num_inds}-samples={num_samples}' \
           f'-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}' \
           f'-model={model}-seed={seed}.npy'
np.save(filename, results)
print(f'Saved results to {filename}')
