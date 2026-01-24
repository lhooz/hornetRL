import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Any
from functools import partial

# ==============================================================================
# 1. PBT STATE DEFINITION
# ==============================================================================
class PBTHyperparams(NamedTuple):
    """
    State container for Population Based Training.
    Is JAX-compatible (registered as a PyTree node automatically via NamedTuple).
    """
    # Reward Weights: [pos, ang_th, ang_ab, lin_vel, ang_vel, eff]
    # Shape: (BATCH_SIZE, 6)
    weights: jnp.ndarray 
    
    # Performance metric for ranking (e.g., smoothed reward)
    # Shape: (BATCH_SIZE,)
    running_reward: jnp.ndarray 

def init_pbt_state(key: jax.Array, batch_size: int, base_weights: jnp.ndarray) -> PBTHyperparams:
    """
    Initializes the PBT state with log-uniform noise around baseline defaults.
    """
    
    # Tile to match batch size: (Batch, 6)
    weights = jnp.tile(base_weights, (batch_size, 1))
    
    # Add initial diversity (+/- 20%)
    # We use log-uniform noise so we don't accidentally make weights negative
    noise_key, _ = jax.random.split(key)
    noise = jax.random.uniform(noise_key, weights.shape, minval=0.8, maxval=1.2)
    weights = weights * noise
    
    return PBTHyperparams(
        weights=weights,
        running_reward=jnp.zeros(batch_size)
    )

# ==============================================================================
# 2. EVOLUTION LOGIC (JIT COMPATIBLE)
# ==============================================================================
@partial(jax.jit, static_argnames=("perturb_factor", "truncate_fraction"))
def pbt_evolve(
    key: jax.Array,
    params: Any,
    opt_state: Any,
    pbt_state: PBTHyperparams,
    perturb_factor: float = 1.2,
    truncate_fraction: float = 0.2
) -> Tuple[Any, Any, PBTHyperparams]:
    """
    Performs the Exploit (Copy) and Explore (Mutate) steps of PBT.
    
    Args:
        key: JAX RNG key.
        params: Model parameters (PyTree).
        opt_state: Optimizer state (PyTree).
        pbt_state: Current PBT hyperparameters.
        perturb_factor: How much to mutate weights (e.g., 1.2 = +/- 20%).
        truncate_fraction: Fraction of population to replace (e.g., bottom 20%).
        
    Returns:
        (new_params, new_opt_state, new_pbt_state)
    """
    batch_size = pbt_state.running_reward.shape[0]
    n_replace = int(batch_size * truncate_fraction)
    
    # ---------------------------------------------------------
    # 1. RANKING
    # ---------------------------------------------------------
    # Get indices of sorted rewards (Ascending: Worst -> Best)
    sorted_idx = jnp.argsort(pbt_state.running_reward)
    
    # Indices of the agents to overwrite (Losers) -> Bottom N
    loser_idx = sorted_idx[:n_replace]
    
    # Indices of the agents to copy from (Winners) -> Top N
    # We select randomly from the top N to maintain diversity
    winner_pool = sorted_idx[-n_replace:]
    
    k1, k2 = jax.random.split(key)
    # For every loser, pick a random winner to copy
    chosen_winners = jax.random.choice(k1, winner_pool, shape=(n_replace,), replace=True)

    # ---------------------------------------------------------
    # 2. EXPLOIT (Copy Winners -> Losers)
    # ---------------------------------------------------------
    def copy_subset(array_leaf):
        # Array shape is (Batch, ...)
        
        # FIX: Check if the leaf is a scalar (0-dimensional). 
        # Global optimizer counts or stats don't have a batch dimension.
        if array_leaf.ndim == 0:
            return array_leaf
            
        # Extract winner values
        replacements = array_leaf[chosen_winners]
        # Update loser positions with winner values
        return array_leaf.at[loser_idx].set(replacements)

    # Apply to Model Params (Neural Network Weights)
    new_params = jax.tree.map(copy_subset, params)
    
    # Apply to Optimizer State (Momentum, etc.)
    new_opt_state = jax.tree.map(copy_subset, opt_state)
    
    # Apply to Reward Weights
    new_weights = copy_subset(pbt_state.weights)

    # ---------------------------------------------------------
    # 3. EXPLORE (Mutate Losers)
    # ---------------------------------------------------------
    # We only mutate the weights of the *newly copied* agents (the former losers)
    
    # Generate mutation noise
    # Range: [1/factor, factor] (e.g., 0.83 to 1.2)
    noise_shape = (n_replace, new_weights.shape[1])
    mutation_noise = jax.random.uniform(
        k2, 
        shape=noise_shape, 
        minval=1.0/perturb_factor, 
        maxval=perturb_factor
    )
    
    # Apply mutation only to the specific rows corresponding to losers
    # 1. Extract the current (just copied) weights of losers
    loser_weights = new_weights[loser_idx]
    
    # 2. Multiply by noise
    mutated_loser_weights = loser_weights * mutation_noise
    
    # 3. Put back into the main weight matrix
    final_weights = new_weights.at[loser_idx].set(mutated_loser_weights)

    # ---------------------------------------------------------
    # 4. SMART RESET METRICS
    # ---------------------------------------------------------
    # We want to keep the score for the "Winners" (since they are stable),
    # but wipe the score for the "Losers" (since they are now new/mutated agents).
    # This forces the new agents to "prove themselves" from scratch.
    
    # Create a mask: 1.0 for losers (mutated), 0.0 for winners (kept)
    reset_mask = jnp.zeros(batch_size)
    reset_mask = reset_mask.at[loser_idx].set(1.0)
    
    # If mask is 1, set reward to 0. If mask is 0, keep existing running_reward.
    # Note: We use the *current* pbt_state.running_reward, not new_running_reward
    smart_running_reward = pbt_state.running_reward * (1.0 - reset_mask)

    return new_params, new_opt_state, PBTHyperparams(final_weights, smart_running_reward)
