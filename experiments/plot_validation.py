import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from data_scaling import LogEstimator


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('-num_inds', type=int, default=200)
parser.add_argument('-num_samples', type=int, default=5000)
parser.add_argument('-min_cardinality', type=int, default=100)
parser.add_argument('-max_cardinality', type=int, default=1000)
parser.add_argument('-num_cardinalities', type=int, default=10)
parser.add_argument('-model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])
parser.add_argument('-seed', type=int, default=0)

args = parser.parse_args()
dataset = args.dataset
num_inds = args.num_inds
num_samples = args.num_samples
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
num_cardinalities = args.num_cardinalities
model = args.model
seed = args.seed

filename = f'data_files/{dataset}/mc_validation-inds={num_inds}-samples={num_samples}' \
           f'-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}' \
           f'-model={model}-seed={seed}.npy'
# filename = f'data_files/{dataset}/mc_validation_combined-inds={num_inds}' \
#            f'-min={min_cardinality}-max={max_cardinality}-num={num_cardinalities}' \
#            f'-model={model}.npy'
result_dict = np.load(filename, allow_pickle=True).item()

cards = result_dict['cards']
samples = result_dict['samples']
estimator = LogEstimator()
mean_array = []
pred_array = []
cardinalities = np.sort(np.unique(cards[0]))
r2_points = []
alpha_list = []

for i in range(num_inds):
    card = cards[i]
    delta = samples[i]
    c, alpha, r2, mean_values, var_values = estimator(delta, card)
    mean_array.append(mean_values)
    pred_array.append(c / np.power(cardinalities, alpha))
    r2_points.append(r2)
    alpha_list.append(alpha)
mean_array = np.array(mean_array)
pred_array = np.array(pred_array)

r2_cards = []
for j in range(len(cardinalities)):
    r2_cards.append(r2_score(mean_array[:, j], pred_array[:, j]))

print(f'Overall R^2 = {r2_score(mean_array, pred_array):.4f}')

# Make validation plot.
plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(10, 9)
gs.update(wspace=0.8)

# Plot lines based on quantiles.
ax = plt.subplot(gs[:6, :])
num_points = 10
indices = np.argsort(alpha_list)[::-1]
quantiles = np.arange(num_inds // num_points, num_inds, num_inds // num_points)
indices = indices[quantiles]
for quantile, index in enumerate(indices):
    # Random selection around the quantile.
    index = index + np.random.randint(-5, 6)
    ax.plot(np.log10(cardinalities), np.log10(np.abs(mean_array[index])),
            marker='o', color=plt.cm.Blues(0.1 + 0.9 * quantile / num_points),
            label=rf'{quantile * 100 // num_points + 100 // num_points}% $\alpha$ Quantile')
ax.set_xlabel(r'Log Dataset Size $k$', fontsize=14)
ax.set_ylabel(r'Log Marginal Contribution $\psi_k$', fontsize=14)
ax.set_title(r'Marginal Contribution vs. Dataset Size', fontsize=14)
ax.tick_params(labelsize=12)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# # Plot random lines.
# ax = plt.subplot(gs[:6, :])
# num_points = 10
# for index in np.random.choice(len(mean_array), num_points, replace=False):
#     ax.plot(np.log10(cardinalities), np.log10(np.abs(mean_array[index])), marker='o')
# ax.set_xlabel(r'Log Dataset Size $k$', fontsize=14)
# ax.set_ylabel(r'Log Marginal Contribution $\psi_k$', fontsize=14)
# ax.set_title(r'Marginal Contribution vs. Dataset Size', fontsize=14)
# ax.tick_params(labelsize=12)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# Histogram of R^2.
ax = plt.subplot(gs[7:, :4])
ax.hist(r2_points, bins=20, density=True)
ax.set_xlabel(r'$R^2$', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Accuracy by Datapoint', fontsize=14)
ax.tick_params(labelsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Line plot of R^2.
ax = plt.subplot(gs[7:, 5:])
ax.plot(np.log10(cardinalities), r2_cards, marker='o')
ax.set_xlabel(r'$\log (k)$', fontsize=14)
ax.set_ylabel(r'$R^2$', fontsize=14)
ax.set_title('Accuracy vs. Cardinality', fontsize=14)
ax.tick_params(labelsize=12)
ax.set_ylim(0.5, 1.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(f'figures/validation-{dataset}-{model}.pdf')


# # Plot matrices of similarity measures.
# fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

# # R^2.
# ax = axarr[0]
# similarity_mat = np.zeros((mean_array.shape[1], mean_array.shape[1]))
# for i in range(mean_array.shape[1]):
#     for j in range(mean_array.shape[1]):
#         similarity_mat[i, j] = r2_score(mean_array[:, i], mean_array[:, j])
#         if similarity_mat[i, j] < 0:
#             similarity_mat[i, j] = np.nan
# ax.imshow(similarity_mat, cmap='viridis')
# ax.set_title(r'$R^2$')
# ax.set_xticks(np.arange(len(cardinalities)))
# ax.set_xticklabels(cardinalities.astype(int), rotation=45, ha='right')
# ax.set_yticks(np.arange(len(cardinalities)))
# ax.set_yticklabels(cardinalities.astype(int))
# for i in range(len(cardinalities)):
#     for j in range(len(cardinalities)):
#         ax.text(j, i, f'{similarity_mat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=4)

# # Pearson correlation.
# ax = axarr[1]
# similarity_mat = np.zeros((mean_array.shape[1], mean_array.shape[1]))
# for i in range(mean_array.shape[1]):
#     for j in range(mean_array.shape[1]):
#         similarity_mat[i, j] = np.corrcoef(mean_array[:, i], mean_array[:, j])[0, 1]
#         if similarity_mat[i, j] < 0:
#             similarity_mat[i, j] = np.nan
# ax.imshow(similarity_mat, cmap='viridis')
# ax.set_title('Pearson Correlation')
# ax.set_xticks(np.arange(len(cardinalities)))
# ax.set_xticklabels(cardinalities.astype(int), rotation=45, ha='right')
# ax.set_yticks(np.arange(len(cardinalities)))
# ax.set_yticklabels(cardinalities.astype(int))
# for i in range(len(cardinalities)):
#     for j in range(len(cardinalities)):
#         ax.text(j, i, f'{similarity_mat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=4)

# # Spearman correlation.
# ax = axarr[2]
# similarity_mat = np.zeros((mean_array.shape[1], mean_array.shape[1]))
# for i in range(mean_array.shape[1]):
#     for j in range(mean_array.shape[1]):
#         similarity_mat[i, j] = spearmanr(mean_array[:, i], mean_array[:, j])[0]
#         if similarity_mat[i, j] < 0:
#             similarity_mat[i, j] = np.nan
# ax.imshow(similarity_mat, cmap='viridis')
# ax.set_title('Spearman Correlation')
# ax.set_xticks(np.arange(len(cardinalities)))
# ax.set_xticklabels(cardinalities.astype(int), rotation=45, ha='right')
# ax.set_yticks(np.arange(len(cardinalities)))
# ax.set_yticklabels(cardinalities.astype(int))
# for i in range(len(cardinalities)):
#     for j in range(len(cardinalities)):
#         ax.text(j, i, f'{similarity_mat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=4)

# plt.tight_layout()
# plt.savefig(f'figures/mean_similarity-{dataset}-{model}.pdf')
