import jax
import jax.numpy as jnp
import optax
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import GFlowNetModel
from functions import get_reward, step_frog, compute_db_loss, compute_true_distribution, print_ascii_map, compute_init_loss
from functools import partial


def make_rollout_fn(model, optimizer, dim, side, reg_coef, max_steps=200):
    n_actions = 2 * dim + 1  

    def sample_action(key, logits):
        return jax.random.categorical(key, jax.nn.log_softmax(logits[:n_actions]))

    def env_step(state, action):
        next_state = step_frog(state, action, dim, side)
        done = action == (2 * dim)
        reward = jnp.where(done, get_reward(state, side), 0.0)
        next_state = jnp.where(done, state, next_state)
        return next_state, done, reward

    def update_step(carry, transition):
        params, opt_state = carry
        s, sp, action, done, reward, is_init = transition

        def loss_fn(p):
            init_loss = compute_init_loss(p['log_Z'], model.apply, p['nn'], s, dim, side)
            step_loss = compute_db_loss(p['nn'], model.apply, s, sp, action, done, reward, reg_coef, dim, side)
            return jnp.where(is_init, jnp.sum(init_loss), jnp.sum(step_loss))

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), loss_val

    def rollout_single(key, params, start_state):
        def step_fn(carry, _):
            state, done, key = carry
            key, sk1, sk2 = jax.random.split(key, 3)
            s_ohe = jax.nn.one_hot(state, side).reshape(-1)
            logits = model.apply({'params': params['nn']}, s_ohe)
            action = sample_action(sk1, logits)

            next_state, step_done, reward = env_step(state, action)

            active = ~done
            transition = (
                state,        
                next_state,   
                action,
                step_done & active,
                reward * active,
                jnp.array(False),  
            )
            new_done = done | step_done
            return (next_state, new_done, key), (transition, active, next_state * active)

        
        init_transition = (
            start_state, start_state,
            jnp.array(0), jnp.array(False), jnp.array(0.0),
            jnp.array(True),   
        )

        (final_state, _, _), (transitions, active_mask, exit_states) = jax.lax.scan(
            step_fn,
            (start_state, jnp.array(False), key),
            None,
            length=max_steps,
        )

        all_s      = jnp.concatenate([start_state[None], transitions[0]], axis=0)
        all_sp     = jnp.concatenate([start_state[None], transitions[1]], axis=0)
        all_a      = jnp.concatenate([jnp.array([0]),    transitions[2]], axis=0)
        all_done   = jnp.concatenate([jnp.array([False]),transitions[3]], axis=0)
        all_rew    = jnp.concatenate([jnp.array([0.0]),  transitions[4]], axis=0)
        all_init   = jnp.concatenate([jnp.array([True]), transitions[5]], axis=0)
        all_active = jnp.concatenate([jnp.array([True]), active_mask],    axis=0)

        traj_len = active_mask.sum() + 1
        exit_state = final_state

        return (all_s, all_sp, all_a, all_done, all_rew, all_init, all_active), traj_len, exit_state

    @jax.jit
    def batched_train_step(params, opt_state, key, start_states):
        B = start_states.shape[0]
        keys = jax.random.split(key, B)

        all_transitions, traj_lens, exit_states = jax.vmap(
            rollout_single, in_axes=(0, None, 0)
        )(keys, params, start_states)

        
        def flatten(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        flat = jax.tree_util.tree_map(flatten, all_transitions)
        s, sp, a, done, rew, is_init, active = flat

        def masked_update(carry, idx):
            params, opt_state = carry
            transition = (s[idx], sp[idx], a[idx], done[idx], rew[idx], is_init[idx])
            (new_params, new_opt_state), loss = update_step((params, opt_state), transition)

            params_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(active[idx], new, old), new_params, params
            )
            opt_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(active[idx], new, old), new_opt_state, opt_state
            )
            return (params_out, opt_out), loss * active[idx]

        total_steps = B * (max_steps + 1)
        (new_params, new_opt_state), losses = jax.lax.scan(
            masked_update, (params, opt_state), jnp.arange(total_steps)
        )

        return new_params, new_opt_state, losses.sum(), traj_lens, exit_states

    return batched_train_step



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",       type=int,   default=2)
    parser.add_argument("--side",      type=int,   default=20)
    parser.add_argument("--reg_coef",  type=float, default=1e-4)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--steps",     type=int,   default=10000)
    parser.add_argument("--batch_size",type=int,   default=32,   help="Trajectories per update")
    parser.add_argument("--max_traj",  type=int,   default=200,  help="Max steps per trajectory")
    parser.add_argument("--dir",       type=str,   default=".",  help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    key = jax.random.PRNGKey(0)
    model = GFlowNetModel(dim=args.dim, side=args.side, hidden_size=256)

    key, subkey = jax.random.split(key)
    dummy_input   = jnp.zeros(args.dim * args.side)
    initial_params = model.init(subkey, dummy_input)['params']
    params         = {'nn': initial_params, 'log_Z': jnp.zeros(1)}

    optimizer  = optax.adam(args.lr)
    opt_state  = optimizer.init(params)

    batched_train_step = make_rollout_fn(
        model, optimizer, args.dim, args.side, args.reg_coef, args.max_traj
    )

    true_dist, true_Z = compute_true_distribution(args.dim, args.side)
    print(f"Training GFlowNet | λ={args.reg_coef} | batch={args.batch_size} | True LogZ={np.log(true_Z):.4f}")

    l1_history       = []
    traj_len_history = []
    loss_history     = []
    exits            = []

    for i in tqdm(range(args.steps)):
        key, sk_states, sk_train = jax.random.split(key, 3)
        start_states = jax.random.randint(sk_states, (args.batch_size, args.dim), 0, args.side)

        params, opt_state, total_loss, traj_lens, exit_states = batched_train_step(
            params, opt_state, sk_train, start_states
        )

        traj_len_history.extend(np.array(traj_lens).tolist())
        loss_history.append(float(total_loss))
        exits.extend(np.array(exit_states).tolist())

        if i % 500 == 0 and i > 0:
            recent_exits = np.array(exits[-500:])
            emp_dist = np.zeros_like(true_dist)
            for e in recent_exits:
                emp_dist[tuple(int(x) for x in e)] += 1
            emp_dist /= emp_dist.sum() + 1e-9

            l1 = float(np.sum(np.abs(emp_dist - true_dist)))
            l1_history.append(l1)

            avg_loss  = float(np.mean(loss_history[-500:]))
            avg_traj  = float(np.mean(traj_len_history[-500:]))
            est_log_Z = float(params['log_Z'][0])

            print(f"\n[Iter {i}] L1: {l1:.4f} | Loss: {avg_loss:.4f} | "
                  f"Path: {avg_traj:.1f} | LogZ: {est_log_Z:.2f}")
            print_ascii_map(exits, args.side)

    prefix = os.path.join(args.dir, f"lam_{args.reg_coef}")
    np.save(f"{prefix}_l1.npy",   np.array(l1_history))
    np.save(f"{prefix}_traj.npy", np.array(traj_len_history))
    np.save(f"{prefix}_loss.npy", np.array(loss_history))
    print(f"Дело сделано — результаты в {args.dir}/")


if __name__ == "__main__":
    main()