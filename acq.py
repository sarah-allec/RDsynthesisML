import jax.random as jra
import numpy as np
import random
import gpax

def UCB(y_pred, unc_pred, opt, kappa=0.5):
    """
    Computes upper confidence bound acquisition function.

    Args:
        y_pred (np.ndarray): Array of ML predictions
        unc_pred (np.ndarray): Array of ML prediction uncertainties
        opt (str | int, float): Optimization strategy - if driving toward a specific output value, provide value as int or float; otherwise, specify 'min' for minimization or 'max' for maximization.
        kappa (float): Weight for influence of prediction uncertainties on decision-making 
    """
    if isinstance(opt, (int, float)):
        return -(y_pred - opt) ** 2 + kappa * unc_pred
    if opt == 'min':
        return -y_pred + kappa * unc_pred
    if opt == 'max':
        return y_pred + kappa * unc_pred

# adapted from gpax
def thompson_sampling(rng_key, model, X):
    """
    Performs Thompson sampling.

    Args:
        rng_key (jnp.ndarray): Random number generator key
        model ( BNN | partialBNN ): Trained BNN or partialBNN model
        X (jnp.ndarray | np.ndarray): Array of ML inputs
    """
    posterior_samples = model.get_samples()
    idx = jra.randint(rng_key, (1,), 0, len(posterior_samples["mu"]))
    samples = {k: v[idx] for (k, v) in posterior_samples.items()} # gets a single sample of NN parameters
    mean, var = model.predict(rng_key=rng_key, X_new=X, samples=samples) # foward pass with the sampled NN parameters
    return mean

def Thompson(model, X, y_scaler, opt, N=1):
    """
    Thompson-sampling based acquisition function.

    Args:
        model ( BNN | partialBNN ): Trained BNN or partialBNN model
        X (jnp.ndarray | np.ndarray): Array of ML inputs
        opt (str | int, float): Optimization strategy - if driving toward a specific output value, provide value as int or float; otherwise, specify 'min' for minimization or 'max' for maximization.
        N ( int ): batch size (i.e., number of Thompson samples per x)
    """
    t_samples = np.zeros( (len(X), N) )
    for i in range(N):
        _, rng_key = gpax.utils.get_keys(random.randint(0,999))
        t_sample_scaled = thompson_sampling(rng_key, model=model, X=X)
        t_sample = y_scaler.inverse_transform( t_sample_scaled.reshape(-1, 1) )
        if isinstance(opt, (int, float)):
            t_samples[:,i] = (-(t_sample - opt) ** 2).reshape((len(X), ))
        if opt == 'min':
            t_samples[:,i] = (-t_sample).reshape((len(X), ))
        if opt == 'max':
            t_samples[:,i] = t_sample.reshape((len(X), ))
    return t_samples
