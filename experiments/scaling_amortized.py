'''
Fit individualized scaling laws using the amortized estimator.
'''

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset, Subset
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from data_scaling import LogEstimator, LikelihoodModel, Classifier, MarginalContributionStackDataset
from utils import load_dataset


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
parser.add_argument('--min_cardinality', type=int, default=100)
parser.add_argument('--max_cardinality', type=int, default=1000)
parser.add_argument('--model', type=str,
                    choices=['logreg', 'mlp', 'svm', 'tree', 'knn', 'rf', 'skmlp', 'gbm', 'linreg', 'regmlp'])

# Training parameters.
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--alpha_penalty', type=float, default=1e-5)
parser.add_argument('--mbsize', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--train_samples', type=int, default=100)

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
min_cardinality = args.min_cardinality
max_cardinality = args.max_cardinality
model = args.model

max_epochs = args.max_epochs
lr = args.lr
min_lr = args.min_lr
weight_decay = args.weight_decay
alpha_penalty = args.alpha_penalty
mbsize = args.mbsize
hidden_dim = args.hidden_dim
train_samples = args.train_samples


# Load samples for fitting estimator.
filename = f'data_files/{dataset}/mc_combined-inds=-1' \
           f'-min={min_cardinality}-max={max_cardinality}-model={model}.npy'
results = np.load(filename, allow_pickle=True).item()
deltas = torch.tensor(results['samples']).float()
cardinalities = torch.tensor(results['cards']).float()

# Load data.
x_train, y_train, x_val, y_val, _, _ = load_dataset(dataset, return_datasets=False)
input_dim = x_train.shape[1]
if y_train.dtype is torch.int64:
    # Classification task.
    output_dim = torch.max(y_train).item() + 1
else:
    # Regression task.
    raise ValueError('amortization is not currently supported for this label type')
train_dataset = TensorDataset(x_train.float(), y_train)
val_dataset = TensorDataset(x_val.float(), y_val)


# Pretrain model as classifier.
backbone = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)
classifier = Classifier(backbone, lr=1e-3, weight_decay=weight_decay)
trainer = pl.Trainer(
    max_epochs=10,
    precision='bf16-mixed',
    accelerator='gpu',
    devices=[0],
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[TQDMProgressBar(refresh_rate=10)]
)
train_loader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=mbsize)
trainer.fit(classifier, train_loader, val_loader)


# Set up dataset with marginal contributions and cardinalities.
train_contributions = TensorDataset(deltas[:, :train_samples],
                                    cardinalities[:, :train_samples])
scaling_dataset = MarginalContributionStackDataset(train_dataset, train_contributions)

# Split and set up data loaders.
all_inds = np.arange(len(train_dataset))
np.random.shuffle(all_inds)
val_cutoff = int(len(train_dataset) * 0.1)
scaling_train_dataset = Subset(scaling_dataset, all_inds[:-val_cutoff])
scaling_val_dataset = Subset(scaling_dataset, all_inds[-val_cutoff:])
scaling_train_loader = DataLoader(scaling_train_dataset, batch_size=mbsize, shuffle=True, drop_last=True)
scaling_val_loader = DataLoader(scaling_val_dataset, batch_size=mbsize)


# Set up model as amortized scaling law estimator.
backbone[-1] = nn.Linear(backbone[-1].in_features, 5 * output_dim)
nn.init.zeros_(backbone[-1].weight)
nn.init.zeros_(backbone[-1].bias)
scaling_model = LikelihoodModel(
    backbone,
    lr=lr,
    min_lr=min_lr,
    weight_decay=weight_decay,
    alpha_penalty=alpha_penalty,
    save_architecture=False)

# Train model.
best_callback = ModelCheckpoint(
    save_top_k=1,
    monitor='val_loss',
    filename='{epoch}-{val_loss:.8f}',
    verbose=True
)
epoch_callback = ModelCheckpoint(
    every_n_epochs=1,
    filename='{epoch}'
)
trainer = pl.Trainer(
    max_epochs=max_epochs,
    precision='bf16-mixed',
    accelerator='gpu',
    devices=[0],
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[TQDMProgressBar(refresh_rate=10), best_callback, epoch_callback]
)
trainer.fit(scaling_model, scaling_train_loader, scaling_val_loader)

# Provide path for best model.
best_model_path = best_callback.best_model_path
print(f'Best model path: {best_model_path}')


# Generate predictions for entire dataset.
scaling_model = LikelihoodModel.load_from_checkpoint(best_model_path, model=backbone).eval()
train_loader = DataLoader(train_dataset, batch_size=mbsize)
preds = trainer.predict(scaling_model, train_loader)
c_preds = torch.cat([pred['c'] for pred in preds]).float().squeeze().numpy()
log_c_preds = np.log(np.abs(c_preds) + 1e-10)
alpha_preds = torch.cat([pred['alpha'] for pred in preds]).float().squeeze().numpy()
sigma_preds = torch.cat([pred['sigma'] for pred in preds]).float().squeeze().numpy()
beta_preds = torch.cat([pred['beta'] for pred in preds]).float().squeeze().numpy()

# Save predictions.
df = pd.DataFrame({'alpha': alpha_preds, 'c': c_preds, 'log_c': log_c_preds, 'sigma': sigma_preds, 'beta': beta_preds})
filename = f'model_results/{dataset}/amortized-samples={train_samples}-model={model}.csv'
df.to_csv(filename)


# Compare to validation results.
filename = f'data_files/{dataset}/mc_validation-inds=1000-samples=1000' \
           f'-min={min_cardinality}-max={max_cardinality}-num=10-model={model}-seed=0.npy'
results = np.load(filename, allow_pickle=True).item()
deltas = results['samples']
cardinalities = results['cards']
relevant_inds = results['relevant_inds']
c_preds = c_preds[relevant_inds]
log_c_preds = log_c_preds[relevant_inds]
alpha_preds = alpha_preds[relevant_inds]
sigma_preds = sigma_preds[relevant_inds]
beta_preds = beta_preds[relevant_inds]

# Calculate validation ground truth.
reference_results = []
unique_cardinalities = np.unique(cardinalities)
estimator = LogEstimator()
for i in range(len(relevant_inds)):
    c, alpha, r2, mean_values, var_values = estimator(deltas[i], cardinalities[i])
    outputs = {'c': c, 'alpha': alpha, 'r2': r2}
    for cardinality, mean in zip(unique_cardinalities, mean_values):
        outputs[f'mean_{int(cardinality)}'] = mean
    reference_results.append(outputs)
reference_df = pd.DataFrame(reference_results)
reference_df['log_c'] = np.log(np.abs(reference_df['c']) + 1e-10)

# Plot various diagnostics for predictions.
fig, axarr = plt.subplots(2, 3, figsize=(15, 8))

# c scatterplot.
ax = axarr[0, 0]
ax.scatter(reference_df['c'], c_preds, alpha=0.2)
ax.plot([reference_df['c'].min(), reference_df['c'].max()],
        [reference_df['c'].min(), reference_df['c'].max()], color='black')
ax.set_xlabel(r'$c$ (Log-Estimator)')
ax.set_ylabel(r'$c$ (Amortized)')
ax.set_title(r'Scatterplot of $c$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# log c scatterplot.
ax = axarr[0, 1]
ax.scatter(reference_df['log_c'], log_c_preds, alpha=0.2)
ax.plot([reference_df['log_c'].min(), reference_df['log_c'].max()],
        [reference_df['log_c'].min(), reference_df['log_c'].max()], color='black')
ax.set_xlabel(r'$\log |c|$ (Log-Estimator)')
ax.set_ylabel(r'$\log |c|$ (Amortized)')
ax.set_title(r'Scatterplot of $\log |c|$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# alpha scatterplot.
ax = axarr[0, 2]
ax.scatter(reference_df['alpha'], alpha_preds, alpha=0.2)
ax.plot([reference_df['alpha'].min(), reference_df['alpha'].max()],
        [reference_df['alpha'].min(), reference_df['alpha'].max()], color='black')
ax.set_xlabel(r'$\alpha$ (Log-Estimator)')
ax.set_ylabel(r'$\alpha$ (Amortized)')
ax.set_title(r'Scatterplot of $\alpha$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Scatterplot of prediction vs mean @ 100 for amortized estimator.
ax = axarr[1, 0]
mean_preds = c_preds * (100 ** (- alpha_preds))
r2 = r2_score(reference_df['mean_100'], mean_preds)
ax.scatter(mean_preds, reference_df['mean_100'], alpha=0.2, label=rf'$R^2 = {r2:.2f}$')
ax.set_xlabel(r'$c / k^{\alpha}$ (Amortized Estimator)')
ax.set_ylabel(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ @ $|\mathcal{D}| = 100$')
ax.set_title(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ vs Amortized Estimator Prediction')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Scatterplot of prediction vs mean @ 599 for amortized estimator.
ax = axarr[1, 1]
mean_preds = c_preds * (599 ** (- alpha_preds))
r2 = r2_score(reference_df['mean_599'], mean_preds)
ax.scatter(mean_preds, reference_df['mean_599'], alpha=0.2, label=rf'$R^2 = {r2:.2f}$')
ax.set_xlabel(r'$c / k^{\alpha}$ (Amortized Estimator)')
ax.set_ylabel(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ @ $|\mathcal{D}| = 599$')
ax.set_title(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ vs Amortized Estimator Prediction')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Scatterplot of prediction vs mean @ 1000 for amortized estimator.
ax = axarr[1, 2]
mean_preds = c_preds * (1000 ** (- alpha_preds))
r2 = r2_score(reference_df['mean_1000'], mean_preds)
ax.scatter(mean_preds, reference_df['mean_1000'], alpha=0.2, label=rf'$R^2 = {r2:.2f}$')
ax.set_xlabel(r'$c / k^{\alpha}$ (Amortized Estimator)')
ax.set_ylabel(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ @ $|\mathcal{D}| = 1000$')
ax.set_title(r'$\mathbb{E}[\Delta(z, \mathcal{D})]$ vs Amortized Estimator Prediction')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(f'figures/amortized_hist-{dataset}-{model}.pdf')
