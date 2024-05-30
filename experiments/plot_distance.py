import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_experiment_mediator
from sklearn.linear_model import LogisticRegression
from data_scaling import LogEstimator

# Add argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-estimator', type=str, default='likelihood', choices=['likelihood', 'log'])

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
estimator = args.estimator


# Load validation samples.
file_path = f'data_files/{dataset}/mc_validation-inds=1000-samples=1000-min=100-max=1000-num=10-model=logreg-seed=0.npy'
results_dict = np.load(file_path, allow_pickle=True).item()
relevant_inds = results_dict['relevant_inds']
delta_array = results_dict['samples']
card_array = results_dict['cards']

# Load dataset.
exper_med = get_experiment_mediator(dataset, 'logreg')
x_train, y_train, *_, x_test, y_test = exper_med.fetcher.datapoints
x_train = x_train[relevant_inds]
y_train = y_train[relevant_inds]
index1 = np.where(y_train[:,1]==0)[0]
index2 = np.where(y_train[:,1]==1)[0]

# Fit logistic regression model on all validation points.
model = LogisticRegression()
model.fit(x_train, y_train[:,1])
coef = model.coef_
intercept = model.intercept_

# Helper function to calculate distance to decision boundary.
def decision_boundary(x):
    return (-(np.dot(coef, x.T) + intercept))[0]

# Calculate decision boundary distances for all points.
distances = decision_boundary(x_train)


# Prepare scaling law parameters.
alpha_list = []
c_list = []
if estimator == 'log':
    # Fit log estimator.
    log_estimator = LogEstimator()
    for i in tqdm.trange(len(relevant_inds)):
        card = card_array[i]
        delta = delta_array[i]
        c, alpha, _, _, _ = log_estimator(delta, card)
        alpha_list.append(alpha)
        c_list.append(np.log10(np.abs(c)))
else: 
    # Use existing scaling law parameters.
    file_path = f'data_files/{dataset}/scaling_all-model=logreg.csv'
    df = pd.read_csv(file_path)
    alpha_list = np.array(df['alpha'])[relevant_inds]
    c_list = np.log10(np.abs(np.array(df['c'])[relevant_inds]))


# Plot correlation for alpha.
plt.figure()
plt.scatter(distances[index1], [alpha_list[i] for i in index1], label='Class 1', alpha=0.7)
cor1 = np.corrcoef(distances[index1], [alpha_list[i] for i in index1])[0,1]
plt.scatter(distances[index2], [alpha_list[i] for i in index2], label='Class 2', alpha=0.7)
cor2 = np.corrcoef(distances[index2], [alpha_list[i] for i in index2])[0,1]
plt.xlabel('Distance to Decision Boundary', fontsize=19)
plt.ylabel(r'$\alpha$', fontsize=19)
plt.title(f'Class 1 correlation: {round(cor1, 3)} \n Class 2 correlation: {round(cor2, 3)}', fontsize=19)
plt.legend()
plt.tick_params(labelsize=12)
plt.savefig(f'dist_{dataset}_{estimator}_alpha.pdf')

# Plot correlation for c.
plt.figure()
plt.scatter(distances[index1], [c_list[i] for i in index1], label='Class 1', alpha=0.7)
cor1 = np.corrcoef(distances[index1], [c_list[i] for i in index1])[0,1]
plt.scatter(distances[index2], [c_list[i] for i in index2], label='Class 2', alpha=0.7)
cor2 = np.corrcoef(distances[index2], [c_list[i] for i in index2])[0,1]
plt.xlabel('Distance to Decision Boundary', fontsize=19)
plt.ylabel(r'$log |c|$', fontsize=19)
plt.title(f'Class 1 correlation: {round(cor1, 3)} \n Class 2 correlation: {round(cor2, 3)}', fontsize=19)
plt.legend()
plt.tick_params(labelsize=12)
plt.savefig(f'dist_{dataset}_{estimator}_c.pdf')