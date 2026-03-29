import jax
import jax.numpy as jnp
import optax
import numpy as np
import argparse
from tqdm import tqdm
from model import GFlowNetModel
from functions import get_reward, step_frog, compute_db_loss, compute_true_distribution, print_ascii_map, compute_init_loss
from functools import partial

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--side", type=int, default=20)
    parser.add_argument("--reg_coef", type=float, default=1e-4) # Start with smaller lambda
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    model = GFlowNetModel(dim=args.dim, side=args.side, hidden_size=256)
    
    dummy_input = jnp.zeros(args.dim * args.side)
    key, subkey = jax.random.split(key)
    initial_params = model.init(subkey, dummy_input)['params']
    

    params = {'nn': initial_params, 'log_Z': jnp.zeros(1)}
    
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @partial(jax.jit, static_argnums=(7,))
    def train_step(params, opt_state, s, sp, action, done, reward, is_init):
        def loss_fn(p):
            if is_init:
                loss = compute_init_loss(p['log_Z'], model.apply, p['nn'], s, args.dim, args.side)
            else:
                loss = compute_db_loss(p['nn'], model.apply, s, sp, action, done, reward, args.reg_coef, args.dim, args.side)
            
            return jnp.sum(loss) 
        
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    print(f"Training non-acyclic GFlowNet | Lambda: {args.reg_coef}")
    true_dist, true_Z = compute_true_distribution(args.dim, args.side)
    print(f"True Log Z: {np.log(true_Z):.4f}")
    
    l1_history = []
    traj_len_history = []
    loss_history = [] 
    exits = []

    for i in tqdm(range(args.steps)):
        key, subkey = jax.random.split(key)
        state = jax.random.randint(subkey, (args.dim,), 0, args.side)
        
        params, opt_state, init_loss = train_step(params, opt_state, state, state, 0, False, 0.0, True)
        
        done = False
        steps = 0
        total_traj_loss = init_loss

        while not done:
            s_ohe = jax.nn.one_hot(state, args.side).reshape(-1)
            logits = model.apply({'params': params['nn']}, s_ohe)
            
            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, jax.nn.log_softmax(logits[:2*args.dim+1]))
            
            if action == 2 * args.dim: 
                reward = get_reward(state, args.side)
                params, opt_state, step_loss = train_step(params, opt_state, state, state, action, True, reward, False)
                exits.append(np.array(state))
                done = True
            else:
                next_state = step_frog(state, action, args.dim, args.side)
                params, opt_state, step_loss = train_step(params, opt_state, state, next_state, action, False, 0.0, False)
                state = next_state
            
            total_traj_loss += step_loss
            steps += 1
        
        traj_len_history.append(steps)
        loss_history.append(total_traj_loss)

        if i % 500 == 0 and i > 0:
            recent_exits = np.array(exits[-500:])
            emp_dist = np.zeros_like(true_dist)
            for e in recent_exits: emp_dist[tuple(e)] += 1
            emp_dist /= emp_dist.sum()
            
            l1 = np.sum(np.abs(emp_dist - true_dist))
            l1_history.append(l1)
            
            avg_loss = np.mean(np.array(loss_history[-500:]))
            avg_traj = np.mean(traj_len_history[-500:])
            est_log_Z = params['log_Z'][0]

            print(f"\n[Iter {i}] L1: {l1:.4f} | Loss: {avg_loss:.4f} | Path: {avg_traj:.1f} | LogZ: {est_log_Z:.2f}")
            print_ascii_map(exits, args.side)


    prefix = f"lam_{args.reg_coef}"
    np.save(f"{prefix}_l1.npy", np.array(l1_history))
    np.save(f"{prefix}_traj.npy", np.array(traj_len_history))
    np.save(f"{prefix}_loss.npy", np.array(loss_history))
    print("Дело сделано")

if __name__ == "__main__":
    main()
