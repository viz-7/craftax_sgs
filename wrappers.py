import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any, NamedTuple

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, None, 0, None, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 6))
    def step(self, rng, state, sgs_state, action, random_frac, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, sgs_state, action, random_frac, params)

        return obs, state, reward, done, info

SGS_EPS = 1e-8
SGS_TEMPERATURE = 2.0

def sample_sgs_seed(rng, sgs_state, fallback_key):
    # A buffer has seeds if it has wrapped or its cursor has advanced
    valid = (sgs_state.gen > 0) | (sgs_state.cursors > 0)                # (A,)

    # Paper's logits l_i = log(w_i + eps); empties get -inf (probability exactly 0)
    logits = jnp.where(valid, jnp.log(sgs_state.weights + SGS_EPS), -jnp.inf)

    k_ach, k_slot = jax.random.split(rng)
    ach = jax.random.categorical(k_ach, logits / SGS_TEMPERATURE)        # softmax sampling

    # Number of filled slots: full buffer if it has wrapped, else the cursor
    cap = sgs_state.storage.shape[1]
    n_valid = jnp.where(sgs_state.gen[ach] > 0, cap, sgs_state.cursors[ach])
    slot = jax.random.randint(k_slot, (), 0, jnp.maximum(n_valid, 1))

    seed = sgs_state.storage[ach, slot]                                  # (2,) uint32
    ok = valid.any()                                                     # False while the pool is empty
    return jnp.where(ok, seed, fallback_key), ok

class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 6))
    def step(self, rng, state, sgs_state, action, random_frac, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, k_reset, k_choice, k_sgs = jax.random.split(rng, 4)
        use_random = jax.random.uniform(k_choice) < random_frac
        sgs_key, sgs_ok = sample_sgs_seed(k_sgs, sgs_state, k_reset)
        from_sgs  = (~use_random) & sgs_ok                     # only true if the pool was actually used
        reset_key = jnp.where(from_sgs, sgs_key, k_reset)
        obs_re, state_re = self._env.reset(reset_key, params, from_sgs)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    seed: chex.PRNGKey
    from_sgs: bool

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None, from_sgs=False):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0, key, jnp.asarray(from_sgs, dtype=bool))
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            seed=state.seed,
            from_sgs=state.from_sgs
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
