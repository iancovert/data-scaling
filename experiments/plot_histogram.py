import tqdm
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from data_scaling import LogEstimator


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-plot_type', type=str, default='c_hist', help='The type of plot to generate')
parser.add_argument('-estimator', type=str, default='log', choices=['log', 'likelihood'], help='The type of estimator to use')

# Parse arguments.
args = parser.parse_args()
plot_type = args.plot_type
estimator = args.estimator


def generate_estimates(model, dataset, cards, samples, estimator_name):
    # Compute the estimates for each sample, generate a list for each quantity of interest.
    log_c_list = []
    alpha_list = []
    log_sigma2_list = []
    beta_list = []
    r2_list = []
    r2_var_list = []

    if estimator_name == 'log':
        # Fit scaling law estimates in log-space.
        estimator = LogEstimator()
        for i in tqdm.trange(len(cards)):
            # Fit for mean.
            card = cards[i]
            delta = samples[i]
            c, alpha, r2, mean_values, var_values = estimator(delta, card)
            log_c_list.append(np.log(np.abs(c) + 1e-10))
            alpha_list.append(alpha)
            r2_list.append(r2)
            
            # Fit for variance.
            unique_cardinality = np.unique(card)
            model = LinearRegression(fit_intercept=True)
            model.fit(-np.log10(unique_cardinality).reshape(-1, 1), np.log10(var_values))
            pred_var = model.predict(-np.log10(unique_cardinality).reshape(-1, 1))
            r2_var = r2_score(np.log10(var_values), pred_var)
            beta = model.coef_[0]
            sigma2 = np.power(10, model.intercept_)
            log_sigma2_list.append(np.log(sigma2))
            beta_list.append(beta)
            r2_var_list.append(r2_var)

    elif estimator_name == 'likelihood':
        # Load existing scaling law estimates.
        filename = f'estimates/{dataset}/scaling_all-model=logreg.csv'
        df = pd.read_csv(filename)
        alpha_list = np.array(df['alpha'])
        log_c_list = np.log10(np.abs(np.array(df['c'])) + 1e-10)
        beta_list = np.array(df['beta'])
        log_sigma2_list = 2 * np.log10(np.array(df['sigma']))

        # Calculate R^2 values for mean and variance.
        for i in tqdm.trange(len(cards)):
            card = cards[i]
            delta = samples[i]
            unique_cardinality = np.unique(card)
            mean_values = []
            var_values = []
            for c in unique_cardinality:
                mean = np.log10(np.abs(np.mean(delta[card == c])))
                mean_values.append(mean)
                var = np.log10(np.var(delta[card == c]))
                var_values.append(var)
            mean_values = np.array(mean_values)
            var_values = np.array(var_values)
            pred_mean = alpha_list[i] * -np.log10(unique_cardinality) + log_c_list[i]
            r2 = r2_score(mean_values, pred_mean)
            r2_list.append(r2)
            pred_var = log_sigma2_list[i] + beta_list[i] * -np.log10(unique_cardinality)
            r2_var = r2_score(var_values, pred_var)
            r2_var_list.append(r2_var)
    
    return log_c_list, alpha_list, log_sigma2_list, beta_list, r2_list, r2_var_list


def plot_line(plot_type, estimator):
    print(f'Generating figure for {plot_type}')
    models = ['logreg', 'skmlp', 'svm']
    datasets = ['cifar10', 'MiniBooNE', 'imdb']
    fig, axs = plt.subplots(len(datasets), len(models), figsize=(3 * len(models), 3 * len(datasets)))
    if len(datasets) == 1:
        axs = axs[np.newaxis, :]

    def load_result_dict(model, dataset):
        folder_path = f'data_files/{dataset}/'
        if model == 'logreg':
            # 1000 inds, 1000 samples.
            file_path = f'{folder_path}/mc_validation-inds=1000-samples=1000-min=100-max=1000-num=10-model={model}-seed=0.npy'
        else:
            # 200 inds, 5000 samples.
            file_path = f'{folder_path}mc_validation-inds=200-samples=5000-min=100-max=1000-num=10-model={model}-seed=0.npy'
        result_dict = np.load(file_path, allow_pickle=True).item()
        return result_dict

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            # Load results for the current model and dataset.
            result_dict = load_result_dict(model, dataset)
            if result_dict is None:
                continue
            cardinality = result_dict['cards']
            samples = result_dict['samples']

            # Generate the estimates for each sample
            log_c_list, alpha_list, log_sigma2_list, beta_list, r2_list, r2_var_list \
                = generate_estimates(model, dataset, cardinality, samples, estimator)
            
            # Calculate mean at some specific cardinalities, and mean across all cardinalities.
            psi_list = np.mean(samples, axis=1)
            psi_100_list = []
            psi_1000_list = []
            for n, sample in enumerate(samples):
                psi_100_list.append(np.mean(sample[cardinality[n]==100]))
                psi_1000_list.append(np.mean(sample[cardinality[n]==1000]))
            
            r2_list = np.array(r2_list)
            alpha_list = np.array(alpha_list)
            log_c_list = np.array(log_c_list)
            psi_100_list = np.array(psi_100_list)
            psi_1000_list = np.array(psi_1000_list)

            # Plot the estimates
            if plot_type == 'c_hist':
                log_c_list[r2_list < 0.8] = np.nan
                axs[i, j].hist(log_c_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'log $|c|$', fontsize=14)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'alpha_hist':
                alpha_list[r2_list < 0.8] = np.nan
                axs[i, j].hist(alpha_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'$\alpha$', fontsize=14)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'sigma2_hist':
                axs[i, j].hist(log_sigma2_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'log $\sigma^2$', fontsize=14)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'beta_hist':
                axs[i, j].hist(beta_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'$\beta$', fontsize=14)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'r2_hist':
                axs[i, j].hist(r2_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'$R^2$', fontsize=14)
                axs[i, j].set_xlim(0, 1)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'r2_var_hist':
                axs[i, j].hist(r2_var_list, bins=20, density=True)
                axs[i, j].set_xlabel(r'$R^2$', fontsize=14)
                axs[i, j].set_xlim(0, 1)
                axs[i, j].set_ylabel(r'Frequency', fontsize=14)
            elif plot_type == 'alpha_c':
                log_c_list[r2_list < 0.8] = np.nan
                alpha_list[r2_list < 0.8] = np.nan
                axs[i, j].scatter(alpha_list, log_c_list, alpha=0.7)
                axs[i, j].set_xlabel(r'$\alpha$', fontsize=14)
                axs[i, j].set_ylabel(r'log $|c|$', fontsize=14)
            elif plot_type == 'r2_psi':
                axs[i, j].scatter(r2_list, np.log10(np.abs(psi_list)), alpha=0.25)
                axs[i, j].set_xlabel(r'$R^2$', fontsize=14)
                axs[i, j].set_ylabel(r'$\log|\psi|$', fontsize=14)
            elif plot_type == 'r2_psi_100':
                axs[i, j].scatter(r2_list, np.log10(np.abs(psi_100_list)), alpha=0.25)
                axs[i, j].set_xlabel(r'$R^2$', fontsize=14)
                axs[i, j].set_ylabel(r'$\log|\psi_{100}|$', fontsize=14)
            elif plot_type == 'r2_psi_1000':
                axs[i, j].scatter(r2_list, np.log10(np.abs(psi_1000_list)), alpha=0.25)
                axs[i, j].set_xlabel(r'$R^2$', fontsize=14)
                axs[i, j].set_ylabel(r'$\log|\psi_{1000}|$', fontsize=14)
            axs[i, j].tick_params(labelsize=12)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)

    # Set titles for each row and column.
    if len(datasets) > 1:
        for i, dataset in enumerate(datasets):
            axs[i, 0].set_ylabel(dataset, fontsize=14)
    for j, model in enumerate(models):
        if model == 'logreg':
            axs[0, j].set_title('Logistic Regression', fontsize=14)
        elif model == 'skmlp':
            axs[0, j].set_title('MLP', fontsize=14)
        elif model == 'svm':
            axs[0, j].set_title('SVM', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figure/{plot_type}.pdf')


# Generate plots.
all_plot_list = ['c_hist', 'alpha_hist', 'sigma2_hist', 'beta_hist', 'r2_hist', 'r2_var_hist', 'alpha_c', 'r2_psi', 'r2_psi_100', 'r2_psi_1000']
if plot_type == 'all':
    for plot_type in all_plot_list:
        plot_line(plot_type, estimator)
elif plot_type in all_plot_list:
    plot_line(plot_type, estimator)
else:
    raise ValueError('Invalid plot type')
