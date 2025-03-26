import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple
from numpyro.infer import MCMC, NUTS


def run_mcmc(
    model: Callable[..., None],
    data: Dict[str, jnp.ndarray],
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains
    )
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, **data)
    posterior_samples = mcmc.get_samples()
    posterior_mean = jnp.mean(posterior_samples["player_latents"], axis=0)
    posterior_std = jnp.std(posterior_samples["player_latents"], axis=0)
    return posterior_mean, posterior_std, posterior_samples
