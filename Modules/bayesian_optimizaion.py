"""
Bayesian Optimization Wrapper
=============================

This module provides a simple wrapper around ``skopt.gp_minimize`` to perform
Bayesian Optimization on a black-box objective function. It supports custom
search spaces, acquisition functions, and exposes convenient access to the
best parameters and best score.

Classes
-------
BayesianOptimization
    High-level interface for running Bayesian Optimization using scikit-optimize.

Dependencies
------------
- scikit-optimize (skopt)
- numpy

Example
-------
>>> from skopt.space import Integer, Real
>>> def objective(params):
...     x, y = params
...     return (x - 2)**2 + (y + 1)**2
>>> space = [Integer(-5, 5, name="x"), Real(-3.0, 3.0, name="y")]
>>> opt = BayesianOptimization(objective, space, n_calls=30)
>>> opt.run_optimization()
>>> opt.best_params
{'x': 2, 'y': -1.01}
>>> opt.best_score
-0.0003
"""

from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective


class BayesianOptimization:
    """
    Bayesian Optimization using ``skopt.gp_minimize``.

    Parameters
    ----------
    objective_func : callable
        Objective function to minimize. Must accept a list-like structure
        representing a point in the search space.
    space : list of skopt.space.Dimension
        Search space defining the domain of each parameter.
    n_calls : int, optional (default=50)
        Total number of evaluations of the objective function.
    n_initial_points : int, optional (default=10)
        Number of random initial evaluations before the acquisition function
        guides the search.
    acq_func : str, optional (default="EI")
        Acquisition function to use. Common choices include:
        - ``"EI"`` : Expected Improvement
        - ``"PI"`` : Probability of Improvement
        - ``"LCB"`` : Lower Confidence Bound
    random_state : int, optional (default=42)
        Seed for reproducibility.

    Attributes
    ----------
    results : skopt.OptimizeResult or None
        Optimization results returned by ``gp_minimize``. Contains:
        - ``x`` : best parameters found
        - ``fun`` : best objective value
        - ``x_iters`` : all evaluated points
        - ``func_vals`` : objective values at each point

    Notes
    -----
    The optimizer minimizes the objective function. If the user wants to
    *maximize* a function, they should pass ``lambda x: -f(x)`` instead.

    The ``best_score`` property returns the *negated* value of ``results.fun``
    to allow users to interpret it as a maximization score if needed.
    """

    def __init__(
        self,
        objective_func,
        space,
        n_calls=50,
        n_initial_points=10,
        acq_func="EI",
        random_state=42,
    ):
        self.objective_func = objective_func
        self.space = space
        self.n_calls = n_calls
        self._n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.random_state = random_state
        self.results = None

    def run_optimization(self):
        """
        Run the Bayesian Optimization process.

        Returns
        -------
        self : BayesianOptimization
            The instance itself, with ``results`` populated.

        Raises
        ------
        Exception
            If ``gp_minimize`` fails internally.

        Notes
        -----
        This method calls ``gp_minimize`` with the provided objective function,
        search space, acquisition function, and number of calls.
        """
        self.results = gp_minimize(
            func=self.objective_func,
            dimensions=self.space,
            n_calls=self.n_calls,
            n_initial_points=self._n_initial_points,
            acq_func=self.acq_func,
            random_state=self.random_state,
        )
        return self

    # ---------- Résultats ----------

    @property
    def best_params(self):
        """
        Return the best parameters found by the optimizer.

        Returns
        -------
        dict
            Mapping from parameter names to their optimal values.

        Raises
        ------
        RuntimeError
            If optimization has not been run yet.
        """
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return dict(zip([dim.name for dim in self.space], self.results.x, strict=True))

    @property
    def best_score(self):
        """
        Return the best (maximized) score.

        Returns
        -------
        float
            The negated objective value at the best point.

        Raises
        ------
        RuntimeError
            If optimization has not been run yet.

        Notes
        -----
        ``gp_minimize`` minimizes the objective function. This property returns
        ``-results.fun`` so that users can interpret the score as a maximization
        metric if desired.
        """
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return -self.results.fun

    @property
    def history(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return {
            "params": self.results.x_iters,
            "scores": -self.results.func_vals,
        }

    # ---------- Visualisation ----------

    def plot_objective_func(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        plot_objective(self.results)

    def plot_convergence(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        plot_convergence(self.results)
