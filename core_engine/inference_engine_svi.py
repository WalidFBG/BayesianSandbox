import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam


def run_svi(
    model: Callable[..., None],
    data: Dict[str, jnp.ndarray],
    num_steps: int = 200,
    lr: float = 1e-2,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    guide = AutoNormal(model)
    optimizer = Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key, **data)

    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, **data)
        if step == num_steps - 1:
            print(f"Step {step + 1}, ELBO loss: {loss}")

    posterior_params = svi.get_params(svi_state)
    posterior_samples = guide.sample_posterior(
        rng_key, posterior_params, sample_shape=(1000,)
    )
    posterior_mean = jnp.mean(posterior_samples["player_latents"], axis=0)
    posterior_std = jnp.std(posterior_samples["player_latents"], axis=0)

    return posterior_mean, posterior_std, posterior_samples
