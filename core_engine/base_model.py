import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Callable


def build_model(
    prior_means: jnp.ndarray,
    prior_stds: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
    def model(
        generic_logits: jnp.ndarray,
        player_idxs: jnp.ndarray,
        weight_vectors: jnp.ndarray,
        outcomes: jnp.ndarray,
    ) -> None:
        player_latents = numpyro.sample(
            "player_latents", dist.Normal(prior_means, prior_stds)
        )

        selected_latents = player_latents[player_idxs.clip(0)]
        contribs = jnp.einsum("nrd,nrd->n", weight_vectors, selected_latents)
        logits = generic_logits + contribs
        probs = jax.nn.sigmoid(logits)

        numpyro.sample("obs", dist.Bernoulli(probs=probs), obs=outcomes)

    return model
