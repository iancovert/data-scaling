'''
Script to generate performance samples for aggregate scaling laws. Unlike
generate_samples.py and generate_samples_validation.py, which sample marginal
contributions, this script simply trains models with different dataset sizes.

Recommended usage is to generate all samples in a single run, because aggregate
scaling laws can be fit much more efficiently than individualized scaling laws.
'''

import argparse
import numpy as np
from utils import get_experiment_mediator, PerformanceSampler


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-num_samples', type=int)
parser.add_argument('-min_cardinality', type=int)
parser.add_argument('-max_cardinality', type=int)
parser.add_argument('-num_cardinalities', type=int)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])
parser.add_argument('-seed', type=int)

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
num_samples = args.num_samples
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
num_cardinalities = args.num_cardinalities
model = args.model
seed = args.seed

# Set up cardinality list.
cardinalities = np.geomspace(min_cardinality, max_cardinality, num_cardinalities).astype(int)

# Get experiment mediator and set up performance sampler.
exper_med = get_experiment_mediator(dataset, model)
performance_sampler = PerformanceSampler(cardinalities=cardinalities, num_samples=num_samples, random_state=seed)
performance_sampler.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)

# Generate samples.
performance_sampler.train_performance_values(**exper_med.train_kwargs)

# Save results for later aggregation.
results = {
    'cards': cardinalities,
    'performance': performance_sampler.performance
}
filename = f'data_files/{dataset}/perf-samples={num_samples}' \
           f'-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}' \
           f'-model={model}-seed={seed}.npy'
np.save(filename, results)
print(f'Saved results to {filename}')
