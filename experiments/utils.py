import os
import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Union, Optional, Callable
import daal4py as d4p
d4p.daalinit(1)
from sklearnex import patch_sklearn
patch_sklearn()  # TODO does this have to be called here?
from sklearn.utils import check_random_state
from opendataval.metrics import Metrics
from opendataval.model import ModelFactory
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import DataFetcher
from opendataval.dataval.margcontrib.sampler import Sampler
from daal4py.sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from opendataval.model.api import ClassifierUnweightedSkLearnWrapper, Model


def load_dataset(dataset_name: str, return_datasets: bool = True):
    '''Load pre-processed dataset.'''
    data_dict = torch.load(os.path.join('data_files', dataset_name, 'processed.pt'))

    if return_datasets:
        train_dataset = data_dict['train_dataset']
        val_dataset = data_dict['val_dataset']
        test_dataset = data_dict['test_dataset']
        return train_dataset, val_dataset, test_dataset
    else:
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        x_val = data_dict['x_val']
        y_val = data_dict['y_val']
        x_test = data_dict['x_test']
        y_test = data_dict['y_test']
        return x_train, y_train, x_val, y_val, x_test, y_test


def get_experiment_mediator(dataset_name: str, model: str):
    '''
    Helper function for preparing opendataval experiment mediator.

    Args:
      dataset_name:
      model:
    '''
    # Load dataset.
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(dataset_name, return_datasets=False)

    # Set up fetcher and metric.
    if y_train.dtype is torch.float32:
        # Regression task.
        fetcher = DataFetcher.from_data_splits(
            x_train.numpy(), y_train.numpy(),
            x_val.numpy(), y_val.numpy(),
            x_test.numpy(), y_test.numpy(),
            one_hot=False)
        metric = Metrics.NEG_MSE
    else:
        # Classification task.
        y_train = torch.nn.functional.one_hot(y_train)
        y_val = torch.nn.functional.one_hot(y_val)
        y_test = torch.nn.functional.one_hot(y_test)
        fetcher = DataFetcher.from_data_splits(
            x_train.numpy(), y_train.numpy(),
            x_val.numpy(), y_val.numpy(),
            x_test.numpy(), y_test.numpy(),
            one_hot=True)
        metric = cross_entropy_loss

    # Prepare data in numpy for model.
    x_train, y_train, _, _, x_test, y_test = fetcher.datapoints

    # Set up model and train as sanity check.
    if model == 'logreg':
        pred_model = ModelFactory(
            model_name='sklogreg',
            fetcher=fetcher
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'linreg':
        pred_model = ModelFactory(
            model_name='sklinreg',
            fetcher=fetcher
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'mlp':
        # Fix initialization to reduce variance.
        torch.manual_seed(0)
        pred_model = ModelFactory(
            model_name='classifiermlp',
            fetcher=fetcher,
            device=torch.device('cuda'),
            hidden_dim=32,
            layers=3,
        )
        train_kwargs = {'epochs': 25, 'batch_size': 1280}
        model = pred_model.clone()
        model.fit(x_train, y_train, **train_kwargs)
    elif model == 'regmlp':
        # Fix initialization to reduce variance.
        torch.manual_seed(0)
        pred_model = ModelFactory(
            model_name='regressionmlp',
            fetcher=fetcher,
            device=torch.device('cuda'),
            hidden_dim=32,
            layers=3,
        )
        train_kwargs = {'epochs': 25, 'batch_size': 1280}
        model = pred_model.clone()
        model.fit(x_train, y_train, **train_kwargs)
    elif model == 'skmlp':
        pred_model = ModelFactory(
            model_name='skmlp',
            fetcher=fetcher,
            hidden_layer_sizes=(32,),
            random_state=0,
            learning_rate_init=0.01,
            max_iter=200,
            alpha=0.01,
            batch_size=1280,
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'svm':
        pred_model = sklearn_model(
            model_name='sksvm',
            fetcher=fetcher,
            probability=True,
            C=1,
            random_state=0,
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'tree':
        pred_model = sklearn_model(
            model_name='sktree',
            fetcher=fetcher,
            random_state=0,
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'rf':
        pred_model = sklearn_model(
            model_name='skrf',
            fetcher=fetcher,
            n_estimators=50,
            random_state=0,
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'knn':
        pred_model = sklearn_model(
            model_name='knn',
            fetcher=fetcher,
            n_neighbors=50,
            weights='distance',
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    elif model == 'gbm':
        pred_model = sklearn_model(
            model_name='skgb',
            fetcher=fetcher,
            random_state=0,
        )
        train_kwargs = None
        model = pred_model.clone()
        model.fit(x_train, y_train)
    else:
        raise ValueError(f'Unknown model name: {model}')

    perf = metric(y_test, model.predict(x_test).cpu())
    print(f'Baseline model performance: {perf=}')

    # Return experiment mediator.
    return ExperimentMediator(fetcher=fetcher, pred_model=pred_model, train_kwargs=train_kwargs, metric_name=metric)


def sklearn_model(model_name: str,
                  fetcher: Optional[DataFetcher] = None,
                  *args,
                  **kwargs) -> Model:
    '''Prepare built-in sklearn model.'''
    _, label_dim = fetcher.covar_dim, fetcher.label_dim
    if model_name is None:
        return None
    model_name = model_name.lower()
    if model_name == 'sktree':
        return ClassifierUnweightedSkLearnWrapper(
            DecisionTreeClassifier, label_dim[0], *args, **kwargs
        )
    elif model_name == 'sksvm':
        return ClassifierUnweightedSkLearnWrapper(
            SVC, label_dim[0], *args, **kwargs
        )
    elif model_name == 'skrf':
        return ClassifierUnweightedSkLearnWrapper(
            RandomForestClassifier, label_dim[0], *args, **kwargs
        )
    elif model_name == 'skgb':
        return ClassifierUnweightedSkLearnWrapper(
            HistGradientBoostingClassifier, label_dim[0], *args, **kwargs
        )
    elif model_name == 'knn':
        return ClassifierUnweightedSkLearnWrapper(
            KNeighborsClassifier, label_dim[0], *args, **kwargs
        )
    else:
        raise ValueError(f'{model_name} is not a valid predefined model')


def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    '''Negated cross entropy loss, returns float.'''
    return - torch.nn.functional.cross_entropy(y_pred, y_true).item()


def sample_dataset(num_points: int, y: np.ndarray, random_state: Optional[np.random.RandomState] = None):
    '''
    Sample dataset without replacement, enforce class balance when applicable.

    Args:
      num_points: number of points to sample
      y: dataset labels
      random_state: random state for sampling
    '''
    # Setup.
    if random_state is None:
        random_state = np.random.RandomState()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    # Determine task type.
    if len(y.shape) == 2 and np.all(np.logical_or(y == 0, y == 1)):
        # Classification task.
        class_probs = np.mean(y, axis=0)
        class_inds = np.argmax(y, axis=1)
        class_counts = np.round(class_probs * num_points).astype(int)
        extra_points = num_points - np.sum(class_counts)
        class_counts[np.argmax(class_probs)] += extra_points
        assert np.sum(class_counts) == num_points
        assert np.all(class_counts > 0)

        # Sample points from each class.
        sampled_inds = []
        for class_ind, class_count in enumerate(class_counts):
            candidates = np.where(class_inds == class_ind)[0]
            sampled_inds.append(random_state.choice(candidates, size=class_count, replace=False))
        sampled_inds = np.concatenate(sampled_inds)
        return list(sampled_inds)
    else:
        # Regression task.
        return list(random_state.choice(len(y), size=num_points, replace=False))


class DistributionalSampler(Sampler):
    '''
    Marginal contribution sampler that generates cardinalities uniformly at random.

    Modeled on samplers from OpenDataVal:
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/margcontrib/sampler.py

    Args:
      num_samples: number of samples per data point
      min_cardinality: minimum cardinality of a training set
      max_cardinality: maximum cardinality of a training set
      random_state: random initial state
    '''

    def __init__(
        self,
        num_samples: int,
        min_cardinality: int = 100,
        max_cardinality: int = 1000,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.num_samples = num_samples
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality
        self.random_state = check_random_state(random_state)

    def setup(self, fetcher, pred_model: Model, metric):
        self.input_fetcher(fetcher)
        self.input_model(pred_model)
        self.input_metric(metric)
        self.set_coalition(torch.arange(len(self.x_train)))

    def set_coalition(self, coalition: torch.Tensor):
        '''Initialize storage to find marginal contribution of each data point.'''
        self.num_points = len(coalition)
        self.total_contribution = np.zeros(self.num_points)
        self.total_count = np.zeros(self.num_points) + 1e-8
        self.samples = np.zeros((self.num_points, self.num_samples))
        self.cards = np.zeros((self.num_points, self.num_samples))
        return self

    def input_fetcher(self, fetcher: DataFetcher):
        '''Input data from a DataFetcher object.'''
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)

    def input_data(self,
                   x_train: Union[torch.Tensor, Dataset],
                   y_train: torch.Tensor,
                   x_valid: Union[torch.Tensor, Dataset],
                   y_valid: torch.Tensor):
        '''Store and transform input data.'''
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def input_model(self, pred_model: Model):
        '''Input the prediction model.

        Args:
          pred_model: model to train repeatedly on different subsampled datasets
        '''
        self.pred_model = pred_model.clone()

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        '''
        Input the evaluation metric.

        Args:
          metric: evaluation function to determine prediction model performance
        '''
        self.metric = metric

    def compute_utility(self, subset: list[int], *args, **kwargs):
        '''
        Evaluate performance of the model on a subset of the training data set.

        Args:
          subset: indices of covariates/label to be used in training
          args: training positional args
          kwargs: training key word arguments
        '''
        curr_model = self.pred_model.clone()
        curr_model.fit(
            Subset(self.x_train, indices=subset),
            Subset(self.y_train, indices=subset),
            *args,
            **kwargs,
        )
        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.metric(self.y_valid.cpu(), y_valid_hat.cpu())
        return curr_perf

    def compute_marginal_contribution(self,
                                      relevant_inds: Union[np.ndarray, None] = None,
                                      *args,
                                      **kwargs):
        '''
        Compute marginal contributions for each point.

        Args:
          relevant_inds: indices of relevant data points, by default None
          args: training positional args
          kwargs: training key word arguments
        '''
        for epoch in tqdm.trange(self.num_samples):
            self._calculate_marginal_contributions(epoch, relevant_inds, *args, **kwargs)

    def _calculate_marginal_contributions(self,
                                          epoch: int,
                                          relevant_inds: Union[np.ndarray, None] = None,
                                          *args,
                                          **kwargs):
        '''
        Compute marginal contributions for each point.

        Args:
          epoch: current epoch number
          relevant_inds: indices of relevant data points
          args: training positional args
          kwargs: training key word arguments
        '''
        # Verify that relevant inds are valid.
        if relevant_inds is None:
            relevant_inds = list(np.arange(self.num_points))
        else:
            relevant_inds = list(relevant_inds)
            assert np.all([((0 <= ind) and (ind < self.num_points)) for ind in relevant_inds])

        # Sample preceding subset at random for each index.
        for idx in relevant_inds:
            # Sample preceding subset.
            cardinality = self.random_state.randint(self.min_cardinality, self.max_cardinality)
            coalition = sample_dataset(cardinality, self.y_train, self.random_state)

            # Calculate utility before and after adding idx.
            prev_perf = self.compute_utility(coalition, *args, **kwargs)
            curr_perf = self.compute_utility(coalition + [idx], *args, **kwargs)

            # Record marginal contribution.
            self.total_contribution[idx] += curr_perf - prev_perf
            self.total_count[idx] += 1
            self.samples[idx, epoch] = curr_perf - prev_perf
            self.cards[idx, epoch] = cardinality


class DistributionalCardinalityListSampler(Sampler):
    '''
    Marginal contribution sampler that uses a list of cardinalities.

    Modeled on samplers from OpenDataVal:
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/margcontrib/sampler.py

    Args:
      num_samples: number of samples per cardinality
      cardinalities: list of cardinalities to use for sampling marginal contributions
      random_state: random initial state
    '''

    def __init__(
        self,
        num_samples: int,
        cardinalities: Union[List, np.ndarray],
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.num_samples = num_samples
        self.cardinalities = cardinalities
        self.random_state = check_random_state(random_state)

    def setup(self, fetcher, pred_model: Model, metric):
        self.input_fetcher(fetcher)
        self.input_model(pred_model)
        self.input_metric(metric)
        self.set_coalition(torch.arange(len(self.x_train)))

    def set_coalition(self, coalition: torch.Tensor):
        '''Initialize storage to find marginal contribution of each data point.'''
        self.num_points = len(coalition)
        self.total_contribution = np.zeros(self.num_points)
        self.total_count = np.zeros(self.num_points) + 1e-8
        self.samples = np.zeros((self.num_points, self.num_samples * len(self.cardinalities)))
        self.cards = np.zeros((self.num_points, self.num_samples * len(self.cardinalities)))
        return self

    def input_fetcher(self, fetcher: DataFetcher):
        '''Input data from a DataFetcher object.'''
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)

    def input_data(self,
                   x_train: Union[torch.Tensor, Dataset],
                   y_train: torch.Tensor,
                   x_valid: Union[torch.Tensor, Dataset],
                   y_valid: torch.Tensor):
        '''Store and transform input data.'''
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def input_model(self, pred_model: Model):
        '''Input the prediction model.

        Args:
          pred_model: model to train repeatedly on different subsampled datasets
        '''
        self.pred_model = pred_model.clone()

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        '''
        Input the evaluation metric.

        Args:
          metric: evaluation function to determine model performance
        '''
        self.metric = metric

    def compute_utility(self, subset: list[int], *args, **kwargs):
        '''
        Evaluate performance of the model on a subset of the training data set.

        Args:
          subset: indices of covariates/label to be used in training
          args: training positional args
          kwargs: training key word arguments
        '''
        curr_model = self.pred_model.clone()
        curr_model.fit(
            Subset(self.x_train, indices=subset),
            Subset(self.y_train, indices=subset),
            *args,
            **kwargs,
        )
        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.metric(self.y_valid.cpu(), y_valid_hat.cpu())
        return curr_perf

    def compute_marginal_contribution(self,
                                      relevant_inds: Union[np.ndarray, None] = None,
                                      *args,
                                      **kwargs):
        '''
        Compute marginal contributions for each point.

        Args:
          relevant_inds: indices of relevant data points
          args: training positional args
          kwargs: training key word arguments
        '''
        for epoch in tqdm.trange(self.num_samples):
            self._calculate_marginal_contributions(epoch, relevant_inds, *args, **kwargs)

    def _calculate_marginal_contributions(self,
                                          epoch: int,
                                          relevant_inds: Union[np.ndarray, None] = None,
                                          *args,
                                          **kwargs):
        '''
        Compute marginal contributions for each point.

        Args:
          epoch: current epoch number
          relevant_inds: indices of relevant data points
          args: training positional args
          kwargs: training key word arguments
        '''
        # Verify that relevant inds are valid.
        if relevant_inds is None:
            relevant_inds = list(np.arange(self.num_points))
        else:
            relevant_inds = list(relevant_inds)
            assert np.all([((0 <= ind) and (ind < self.num_points)) for ind in relevant_inds])

        # Sample preceding subset at random for each index.
        for idx in relevant_inds:
            for i, cardinality in enumerate(self.cardinalities):
                # Sample preceding subset.
                coalition = sample_dataset(cardinality, self.y_train, self.random_state)

                # Calculate utility before and after adding idx.
                prev_perf = self.compute_utility(coalition, *args, **kwargs)
                curr_perf = self.compute_utility(coalition + [idx], *args, **kwargs)

                # Record marginal contribution.
                self.total_contribution[idx] += curr_perf - prev_perf
                self.total_count[idx] += 1
                self.samples[idx, epoch * len(self.cardinalities) + i] = curr_perf - prev_perf
                self.cards[idx, epoch * len(self.cardinalities) + i] = cardinality


class PerformanceSampler:
    '''
    Generate curve of performance vs. number of data points.

    Args:
      cardinalities: list of cardinalities to use for sampling performance
      num_samples: number of samples per cardinality
      random_state: random initial state
    '''

    def __init__(self,
                 cardinalities: Union[List, np.ndarray],
                 num_samples: int,
                 random_state: Optional[np.random.RandomState] = None):
        self.cardinalities = cardinalities
        self.num_samples = num_samples
        self.random_state = check_random_state(random_state)

    def setup(self, fetcher, pred_model: Model, metric):
        self.input_fetcher(fetcher)
        self.input_model(pred_model)
        self.input_metric(metric)

    def input_fetcher(self, fetcher: DataFetcher):
        '''Input data from a DataFetcher object. Alternative way of adding data.'''
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)

    def input_data(self,
                   x_train: Union[torch.Tensor, Dataset],
                   y_train: torch.Tensor,
                   x_valid: Union[torch.Tensor, Dataset],
                   y_valid: torch.Tensor):
        '''Store and transform input data.'''
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def input_model(self, pred_model: Model):
        '''Input the prediction model.

        Args:
          pred_model: model to train repeatedly on different subsampled datasets
        '''
        self.pred_model = pred_model.clone()

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        '''
        Input the evaluation metric.

        Args:
          metric: evaluation function to determine model performance
        '''
        self.metric = metric

    def compute_utility(self, subset: list[int], *args, **kwargs):
        '''
        Evaluate performance of the model on a subset of the training data set.

        Args:
          subset: indices of covariates/label to be used in training
          args: training positional args
          kwargs: training key word arguments
        '''
        curr_model = self.pred_model.clone()
        curr_model.fit(
            Subset(self.x_train, indices=subset),
            Subset(self.y_train, indices=subset),
            *args,
            **kwargs,
        )
        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.metric(self.y_valid.cpu(), y_valid_hat.cpu())
        return curr_perf

    def train_performance_values(self, *args, **kwargs):
        '''
        Args:
          args: training positional args
          kwargs: training key word arguments
        '''
        # Setup.
        self.num_points = len(self.x_train)
        self.performance = {cardinality: [] for cardinality in self.cardinalities}

        for _ in tqdm.trange(self.num_samples):
            for cardinality in self.cardinalities:
                # Sample coalition and train model.
                coalition = sample_dataset(cardinality, self.y_train, self.random_state)
                perf = self.compute_utility(coalition, *args, **kwargs)
                self.performance[cardinality].append(perf)
