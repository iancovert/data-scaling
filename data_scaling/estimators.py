import torch
import numpy as np
from typing import Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.distributions import Normal


class LogEstimator:
    '''
    Estimate c, alpha parameters by solving linear regression problem in log-space.
    '''

    def __init__(self):
        # No initialization needed.
        pass

    def __call__(self,
                 delta: Union[torch.Tensor, np.ndarray],
                 cardinalities: Union[torch.Tensor, np.ndarray]):
        # Setup.
        if isinstance(delta, torch.Tensor):
            delta = delta.numpy()
        if isinstance(cardinalities, torch.Tensor):
            cardinalities = cardinalities.numpy()

        # Prepare mean and variance at each cardinality.
        unique_cardinality = np.unique(cardinalities)
        mean_values = []
        log_mean_values = []
        var_values = []
        for c in unique_cardinality:
            mean = np.mean(delta[cardinalities == c])
            mean_values.append(mean)
            log_mean_values.append(np.log10(np.abs(mean) + 1e-12))
            var = np.var(delta[cardinalities == c])
            var_values.append(var)

        # Fit alpha in log space.
        model = LinearRegression(fit_intercept=True)
        model.fit(-np.log10(unique_cardinality).reshape(-1, 1), log_mean_values)
        alpha = model.coef_[0]

        # Calculate R^2 for model fit.
        pred = model.predict(-np.log10(unique_cardinality).reshape(-1, 1))
        r2 = r2_score(log_mean_values, pred)

        # Fit c in original space.
        model = LinearRegression(fit_intercept=False)
        model.fit(np.power(cardinalities, -alpha).reshape(-1, 1), delta)
        c = model.coef_[0]

        return c, alpha, r2, mean_values, var_values


class LikelihoodEstimator:
    '''
    Estimate c, alpha, sigma, beta parameters by solving joint MLE problem.

    Args:
      lr: learning rate.
      num_steps: number of optimization steps.
      verbose: whether to print final NLL and results.
    '''

    def __init__(self,
                 lr: float = 0.05,
                 num_steps: int = 1000,
                 verbose: bool = False):
        self.lr = lr
        self.num_steps = num_steps
        self.verbose = verbose

    def calculate_optimal_c_sigma(self,
                                  alpha: torch.Tensor,
                                  beta: torch.Tensor,
                                  delta: torch.Tensor,
                                  cardinalities: torch.Tensor):
        # Calculate optimal c.
        c = (
            torch.mean(delta * cardinalities ** (beta - alpha))
            / torch.mean(cardinalities ** (beta - 2 * alpha))
        )

        # Calculate optimal sigma.
        e = (delta - c * (cardinalities ** (- alpha))) ** 2
        w = cardinalities ** beta
        sigma = torch.sqrt(torch.mean(e * w))

        return c, sigma

    def calculate_objective(self,
                            c: torch.Tensor,
                            alpha: torch.Tensor,
                            sigma: torch.Tensor,
                            beta: torch.Tensor,
                            delta: torch.Tensor,
                            cardinalities: torch.Tensor):
        # Calculate negative log-likelihood under normal distributions.
        mean = c * (cardinalities ** (- alpha))
        std = sigma * (cardinalities ** (- beta / 2))
        dist = Normal(loc=mean, scale=std)
        return - torch.mean(dist.log_prob(delta))

    def __call__(self,
                 delta: Union[torch.Tensor, np.ndarray],
                 cardinalities: Union[torch.Tensor, np.ndarray]):
        # Setup.
        if isinstance(delta, np.ndarray):
            delta = torch.tensor(delta).float()
        if isinstance(cardinalities, np.ndarray):
            cardinalities = torch.tensor(cardinalities).float()

        # Initialize.
        log_alpha = torch.tensor(np.float32(np.log(1)), requires_grad=True)
        log_beta = torch.tensor(np.float32(np.log(2)), requires_grad=True)
        param_list = [log_alpha, log_beta]

        # Setup.
        min_nll = np.inf
        best_params = None
        optimizer = torch.optim.Adam(param_list, lr=self.lr)
        bad_iters = 0
        max_bad_iters = 100

        # Optimize.
        for _ in range(self.num_steps):
            # Calculate loss.
            optimizer.zero_grad()
            alpha = torch.exp(log_alpha)
            beta = torch.exp(log_beta)
            c, sigma = self.calculate_optimal_c_sigma(alpha, beta, delta, cardinalities)
            loss = self.calculate_objective(c, alpha, sigma, beta, delta, cardinalities)

            # Check for best parameters.
            # print(f'NLL = {loss.item()}')
            if loss.item() < min_nll:
                min_nll = loss.item()
                best_params = (c.item(), alpha.item(), sigma.item(), beta.item())
                bad_iters = 0
            else:
                bad_iters += 1
                if bad_iters >= max_bad_iters:
                    break

            # Take gradient step.
            loss.backward()
            optimizer.step()

            # Projection step to avoid large values.
            log_alpha.data = torch.clamp(log_alpha.data, max=2)
            log_beta.data = torch.clamp(log_beta.data, max=2)

        # Extract parameters.
        c, alpha, sigma, beta = best_params
        if self.verbose:
            print(f'NLL = {min_nll}')
            print(f'c={c:.2f}, alpha={alpha:.2f}, sigma={sigma:.2f}, beta={beta:.2f}')
        return c, alpha, sigma, beta


class AggregateLikelihoodEstimator:
    '''
    Estimate eps, c, alpha, sigma, beta parameters by solving joint MLE problem.

    Args:
      lr: learning rate.
      num_steps: number of optimization steps.
      verbose: whether to print final NLL and results.
    '''

    def __init__(self,
                 lr: float = 0.05,
                 num_steps: int = 100,
                 verbose: bool = False):
        self.lr = lr
        self.num_steps = num_steps
        self.verbose = verbose

    def calculate_optimal_eps_c_sigma(self,
                                      alpha: torch.Tensor,
                                      beta: torch.Tensor,
                                      error: torch.Tensor,
                                      cardinalities: torch.Tensor):
        # Calculate optimal eps, c.
        w = cardinalities ** beta
        y = error.unsqueeze(1)
        x = torch.stack([torch.ones(len(cardinalities)), (cardinalities ** (- alpha))]).T
        coef = torch.linalg.solve(x.T @ torch.diag(w) @ x, x.T @ torch.diag(w) @ y)
        eps, c = coef[0, 0], coef[1, 0]
        eps = torch.clamp(eps, min=0, max=min(error) - 1e-6)

        # Calculate optimal sigma.
        e = (error - eps - c * (cardinalities ** (- alpha))) ** 2
        sigma = torch.sqrt(torch.mean(e * w))

        return eps, c, sigma

    def calculate_objective(self,
                            eps: torch.Tensor,
                            c: torch.Tensor,
                            alpha: torch.Tensor,
                            sigma: torch.Tensor,
                            beta: torch.Tensor,
                            error: torch.Tensor,
                            cardinalities: torch.Tensor):
        # Calculate negative log-likelihood under normal distributions.
        mean = c * (cardinalities ** (- alpha))
        std = sigma * (cardinalities ** (- beta / 2))
        dist = Normal(loc=mean, scale=std)
        return - torch.mean(dist.log_prob(error - eps))

    def __call__(self,
                 error: Union[torch.Tensor, np.ndarray],
                 cardinalities: Union[torch.Tensor, np.ndarray]):
        # Setup.
        if isinstance(error, np.ndarray):
            error = torch.tensor(error).float()
        if isinstance(cardinalities, np.ndarray):
            cardinalities = torch.tensor(cardinalities).float()

        # Initialize.
        log_alpha = torch.tensor(np.float32(np.log(0.5)), requires_grad=True)
        log_beta = torch.tensor(np.float32(np.log(3)), requires_grad=True)
        param_list = [log_alpha, log_beta]

        # Setup.
        min_nll = np.inf
        best_params = None
        optimizer = torch.optim.Adam(param_list, lr=self.lr)

        # Optimize.
        for _ in range(self.num_steps):
            # Calculate loss.
            optimizer.zero_grad()
            alpha = torch.exp(log_alpha)
            beta = torch.exp(log_beta)
            eps, c, sigma = self.calculate_optimal_eps_c_sigma(alpha, beta, error, cardinalities)
            loss = self.calculate_objective(eps, c, alpha, sigma, beta, error, cardinalities)

            # Check for best parameters.
            # print(f'NLL = {loss.item()}')
            if loss.item() < min_nll:
                min_nll = loss.item()
                best_params = (eps.item(), c.item(), alpha.item(), sigma.item(), beta.item())

            # Take gradient step.
            loss.backward()
            optimizer.step()

        # Extract parameters.
        eps, c, alpha, sigma, beta = best_params
        if self.verbose:
            print(f'NLL = {min_nll}')
            print(f'eps={eps:.2f}, c={c:.2f}, alpha={alpha:.2f}, sigma={sigma:.2f}, beta={beta:.2f}')
        return eps, c, alpha, sigma, beta
