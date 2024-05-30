import tqdm
import argparse
import numpy as np
import pandas as pd
from utils import get_experiment_mediator, cross_entropy_loss
from opendataval.metrics import Metrics
from data_scaling import LogEstimator


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, help='Model type', default='logreg')
parser.add_argument('-dataset', type=str, help='Dataset name', default='MiniBooNE')
parser.add_argument('-subset_size', type=int, help='The size of the subset to add', default=100)
parser.add_argument('-small_sample', type=int, help='The size of the small data pool', default=500)
parser.add_argument('-large_sample', type=int, help='The size of the large data pool', default=5000)
parser.add_argument('-num_experiment', type=int, help='The number of experiment runs', default=1000)
parser.add_argument('-estimator', type=str, help='The choice of scaling law estimator', default='log', choices=['log', 'likelihood'])
parser.add_argument('-seed', type=int, help='Random seed', default=0)
parser.add_argument('-metric', type=str, help='The metric for evaluating model performance', default='accuracy')
parser.add_argument('-filter_points', type=int, help='Filter unusual alpha values or low R^2', default=1)

# Parse arguments
args = parser.parse_args()
dataset = args.dataset
model_name = args.model
subset_size = args.subset_size
small_sample = args.small_sample
large_sample = args.large_sample
num_experiment = args.num_experiment
estimator = args.estimator
filter_points = args.filter_points
metric_name = args.metric
seed = args.seed

print(f'Perform point addition experiment for {model_name} with {dataset}.')

# Random seed for reproducibility.
np.random.seed(seed)

# Load dataset.
exper_med = get_experiment_mediator(dataset, model_name)
x_train, y_train, *_, x_test, y_test = exper_med.fetcher.datapoints

# Load validation samples.
filename = f'data_files/{dataset}/mc_validation-inds=1000-samples=1000-min=100-max=1000-num=10-model={model_name}-seed=0.npy'
results_dict = np.load(filename, allow_pickle=True).item()
relevant_inds = results_dict['relevant_inds']
delta_array = results_dict['samples']
card_array = results_dict['cards']

if len(relevant_inds) <= subset_size:
    raise ValueError('Subset size is larger than the number of relevant indices.')

# Prepare scaling law estimates to score data points.
if estimator == 'log':
    # Fit scaling law using log estimator.
    c_list = []
    alpha_list = []
    r2_list = []
    estimator = LogEstimator()
    for i in tqdm.trange(len(relevant_inds)):
        card = card_array[i]
        delta = delta_array[i]
        c, alpha, r2, _, _ = estimator(delta, card)
        c_list.append(c)
        alpha_list.append(alpha)
        r2_list.append(r2)
    
    r2_list = np.array(r2_list)
    alpha_list = np.array(alpha_list)
    c_list = np.array(c_list)
    if filter_points:
        alpha_list[r2_list < 0.9] = np.nan
    
else:
    # Load saved scaling law estimates.
    filename = f'data_files/{dataset}/scaling_all-model={model_name}.csv'
    df = pd.read_csv(filename)
    alpha_list = np.array(df['alpha'])[relevant_inds]
    c_list = np.array(df['c'])[relevant_inds]
    if filter_points:
        alpha_list[alpha_list < 0.5] = np.nan

# Score data points at small and large cardinalities.
pred_small = c_list / np.power(small_sample, alpha_list)
pred_large = c_list / np.power(large_sample, alpha_list)
pred_shapley = np.mean(delta_array, axis=1)

# Prepare top indices for each scoring method.
indices_small = np.argsort(pred_small)[::-1].copy()
indices_large = np.argsort(pred_large)[::-1].copy()
indices_shapley = np.argsort(pred_shapley)[::-1].copy()
indices_small = indices_small[~np.isnan(alpha_list[indices_small])]
indices_large = indices_large[~np.isnan(alpha_list[indices_large])]


# Select highly scored subsets for each method, ensuring equal number of points per class.
subset_small = []
subset_large = []
subset_shapley = []
test = []

num_class = y_train.shape[1]
for class_label in range(num_class):
    class_indices_small = indices_small[y_train[relevant_inds[indices_small], class_label] == 1]
    class_indices_large = indices_large[y_train[relevant_inds[indices_large], class_label] == 1]
    class_indices_shapley = indices_shapley[y_train[relevant_inds[indices_shapley], class_label] == 1]
    if len(class_indices_small) > subset_size // num_class:
        subset_small += relevant_inds[class_indices_small[:subset_size // num_class]].tolist()
    if len(class_indices_large) > subset_size // num_class:
        subset_large += relevant_inds[class_indices_large[:subset_size // num_class]].tolist()
        test += class_indices_large[0: subset_size // num_class].tolist()
    if len(class_indices_shapley) > subset_size // num_class:
        subset_shapley += relevant_inds[class_indices_shapley[:subset_size // num_class]].tolist()

# Prepare indices for initial pool sampling.
total_inds = np.arange(x_train.shape[0])
total_inds = np.delete(total_inds, relevant_inds)

# Prepare model and performance metric.
pred_model = exper_med.pred_model
if metric_name == 'accuracy': 
    metric = Metrics.ACCURACY
else: 
    metric = cross_entropy_loss

result_list = np.zeros((8, num_experiment))
baseline_list = np.zeros((2, num_experiment))

for num_exp in tqdm.trange(num_experiment):
    # Prepare the small and large initial subsets.
    small_sample_inds = np.random.choice(total_inds, small_sample, replace=False)
    large_sample_inds = np.random.choice(total_inds, large_sample, replace=False)
    
    # Prepare random subset.
    subset_random = []
    for class_label in range(num_class):
        class_indices_all = relevant_inds[y_train[relevant_inds, class_label] == 1]
        subset_random += np.random.choice(class_indices_all, subset_size // num_class, replace=False).tolist()
    
    
    model = pred_model.clone()
    model.fit(x_train[small_sample_inds], y_train[small_sample_inds])
    pred = model.predict(x_test)
    baseline = metric(pred, y_test)
    baseline_list[0, num_exp] = baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_large, small_sample_inds])], y_train[np.concatenate([subset_large, small_sample_inds])])
    pred = model.predict(x_test)
    result_list[0, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_small, small_sample_inds])], y_train[np.concatenate([subset_small, small_sample_inds])])
    pred = model.predict(x_test)
    result_list[1, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_shapley, small_sample_inds])], y_train[np.concatenate([subset_shapley, small_sample_inds])])
    pred = model.predict(x_test)
    result_list[2, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_random, small_sample_inds])], y_train[np.concatenate([subset_random, small_sample_inds])])
    pred = model.predict(x_test)
    result_list[3, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[large_sample_inds], y_train[large_sample_inds])
    pred = model.predict(x_test)
    baseline = metric(pred, y_test)
    baseline_list[1, num_exp] = baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_large, large_sample_inds])], y_train[np.concatenate([subset_large, large_sample_inds])])
    pred = model.predict(x_test)
    result_list[4, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_small, large_sample_inds])], y_train[np.concatenate([subset_small, large_sample_inds])])
    pred = model.predict(x_test)
    result_list[5, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_shapley, large_sample_inds])], y_train[np.concatenate([subset_shapley, large_sample_inds])])
    pred = model.predict(x_test)
    result_list[6, num_exp] = metric(pred, y_test) - baseline
    
    model = pred_model.clone()
    model.fit(x_train[np.concatenate([subset_random, large_sample_inds])], y_train[np.concatenate([subset_random, large_sample_inds])])
    pred = model.predict(x_test)
    result_list[7, num_exp] = metric(pred, y_test) - baseline

print(f'{small_sample} sample with points selected based on {large_sample} sample contribution: mean {100*np.mean(result_list[0])}, std {100*np.std(result_list[0])}')
print(f'{small_sample} sample with points selected based on {small_sample} sample contribution: mean {100*np.mean(result_list[1])}, std {100*np.std(result_list[1])}')
print(f'{small_sample} sample with points selected based on shapley value: mean {100*np.mean(result_list[2])}, std {100*np.std(result_list[2])}')
print(f'{small_sample} sample with points randomly selected: mean {100*np.mean(result_list[3])}, std {100*np.std(result_list[3])}')
print(f'{large_sample} sample with points selected based on {large_sample} sample contribution: mean {100*np.mean(result_list[4])}, std {100*np.std(result_list[4])}')
print(f'{large_sample} sample with points selected based on {small_sample} sample contribution: mean {100*np.mean(result_list[5])}, std {100*np.std(result_list[5])}')
print(f'{large_sample} sample with points selected based on shapley value: mean {100*np.mean(result_list[6])}, std {100*np.std(result_list[6])}')
print(f'{large_sample} sample with points randomly selected: mean {100*np.mean(result_list[7])}, std {100*np.std(result_list[7])}')

result_dict = {
    'perf': result_list,
    'base': baseline_list,
}
np.save(f'data_files/{dataset}/subset_{model_name}_{subset_size}_{small_sample}_{large_sample}_{num_experiment}_{estimator}_{seed}_{filter_points}.npy', result_dict)
