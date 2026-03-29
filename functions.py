import jax
import jax.numpy as jnp
import itertools
import numpy  as np
def get_reward(state, side):
    ax = jnp.abs(state.astype(jnp.float32) / (side - 1) - 0.5)
    reward_corners = jnp.prod(ax > 0.25, axis=-1) * 0.5
    reward_ring = jnp.prod((ax < 0.4) & (ax > 0.3), axis=-1) * 2.0
    return reward_corners + reward_ring + 1e-3
def compute_init_loss(log_Z, model_apply, params, state, dim, side):
    s_ohe = jax.nn.one_hot(state, side).reshape(-1)
    logits = model_apply({'params': params}, s_ohe)
    
    log_F_s = logits[-1]
    log_PF_s0 = -dim * jnp.log(side)
    
    
    log_pbs = jax.nn.log_softmax(logits[2*dim+1 : 4*dim+2])
    log_PB_s0 = log_pbs[-1] # The 'Stop' action in backward policy

    error = (log_Z + log_PF_s0) - (log_F_s + log_PB_s0)
    return jnp.square(error)
def step_frog(state, action, dim, side):
    """Transition logic: 0 to dim-1: dec, dim to 2*dim-1: inc, 2*dim: exit"""
    move_vecs = jnp.eye(dim, dtype=jnp.int32)

    dec_state = jnp.clip(state - move_vecs[action % dim], 0, side - 1)
    inc_state = jnp.clip(state + move_vecs[action % dim], 0, side - 1)
    
    new_state = jnp.where(action < dim, dec_state, state)
    new_state = jnp.where((action >= dim) & (action < 2 * dim), inc_state, new_state)
    return new_state

def compute_db_loss(params, model_fn, s, s_prime, action, done, reward, reg_coef, dim, side):
    s_ohe = jax.nn.one_hot(s, side).reshape(-1)
    sp_ohe = jax.nn.one_hot(s_prime, side).reshape(-1)
    
    logits = model_fn({'params': params}, s_ohe)
    logits_next = model_fn({'params': params}, sp_ohe)
    
    log_PF = jax.nn.log_softmax(logits[:2*dim+1])[action]
    
    log_F_s = logits[-1]
    
    log_F_sp = jnp.where(done, jnp.log(reward), logits_next[-1])
    
    log_PB = jnp.where(done, 0.0, -jnp.log(2 * dim))
    
    error = (log_F_s + log_PF) - (log_F_sp + log_PB)
    
    return jnp.square(error) + reg_coef * jnp.exp(log_F_s)

# plottin functions 
def compute_true_distribution(dim, side):
    """Calculates the target probability for every single cell."""
    size = [side] * dim
    probs = np.zeros(tuple(size))
    
    # Iterate through every possible coordinate in the 20x20 grid
    for state in itertools.product(range(side), repeat=dim):
        # We use the reward function we already wrote
        r = get_reward(jnp.array(state), side)
        probs[state] = r
        
    true_Z = probs.sum()
    return probs / true_Z, true_Z

def print_ascii_map(samples, side):
    """Prints a 20x20 grid in terminal showing where the Frog exited."""
    grid = np.zeros((side, side))
    for s in samples[-500:]: 
        grid[int(s[0]), int(s[1])] += 1
    
    print("\n--- Current Sample Heatmap (Last 500) ---")
    for row in grid:
        line = ""
        for cell in row:
            if cell > 5: line += "█" 
            elif cell > 0: line += "·" 
            else: line += " " 
        print(line)
    print("------------------------------------------\n")
