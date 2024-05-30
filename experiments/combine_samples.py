'''
Script to combine samples from by generate_samples.py or generate_samples_validation.py into a single file.
'''

import os
import argparse
import numpy as np
from glob import glob


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-num_inds', type=int)
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-num_cardinalities', type=int, default=None)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
num_inds = args.num_inds
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
num_cardinalities = args.num_cardinalities
model = args.model

# Find and load results.
if num_cardinalities is None:
    regex = f'mc-inds={num_inds}-samples=*-min={min_cardinality}-max={max_cardinality}' \
            f'-model={model}-seed=*.npy'
else:
    regex = f'mc_validation-inds={num_inds}-samples=*-min={min_cardinality}-max={max_cardinality}' \
            f'-num={num_cardinalities}-model={model}-seed=*.npy'
filenames = glob(os.path.join('data_files', dataset, regex))
if len(filenames) == 0:
    raise ValueError('No files found')
else:
    print(f'Aggregating {len(filenames)} files')

results = []
for filename in filenames:
    # Reduce size of combined results.
    result = np.load(filename, allow_pickle=True).item()
    relevant_inds = result['relevant_inds']
    result['cards'] = result['cards']
    result['samples'] = result['samples']
    result['total_contribution'] = result['total_contribution']
    result['total_count'] = result['total_count']
    if 'data' in result:
        del result['data']
    results.append(result)
assert len(results) > 0
assert all(all(result['relevant_inds'] == results[0]['relevant_inds']) for result in results)

# Combine results.
final_results = {
    'estimates': (sum(result['total_contribution'] for result in results)
                  / sum(result['total_count'] for result in results)),
    'total_contribution': sum(result['total_contribution'] for result in results),
    'total_count': sum(result['total_count'] for result in results),
    'relevant_inds': results[0]['relevant_inds'],
    'cards': np.concatenate([result['cards'] for result in results], axis=1),
    'samples': np.concatenate([result['samples'] for result in results], axis=1)
}
if num_cardinalities is None:
    filename = f'data_files/{dataset}/mc_combined-inds={num_inds}' \
               f'-min={min_cardinality}-max={max_cardinality}-model={model}.npy'
else:
    filename = f'data_files/{dataset}/mc_validation_combined-inds={num_inds}' \
               f'-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}-model={model}.npy'
np.save(filename, final_results)
