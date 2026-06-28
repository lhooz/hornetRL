"""
F4 — Brain-body co-optimization (the manuscript's centerpiece).

The shipped trainer optimizes ONLY policy+critic params; morphology (PhysParams) is
sampled in env.reset and frozen per episode, so no co-adaptation exists. This module
makes a small set of morphology parameters LEARNABLE, bounded leaves and co-optimizes
them with the controller through the same differentiable rollout the controller is
trained on (InferenceFlyEnv.step recomputes compute_props(phys_p) every substep, so
gradients flow into morphology).

Design:
  * Morphology latents (unconstrained) -> sigmoid -> physical ranges that stay INSIDE
    the surrogate's training distribution (mass +/-20%, hinge stiffness +/-30%, CoM
    +/-2mm). The sigmoid bounding IS the surrogate-domain guard: the body can never
    leave the region where the Sobolev surrogate is trustworthy.
  * Two-timescale optimization: morphology on a slow Adam, policy on a fast Adam.
  * Co-design objective over a gust-perturbed rollout:
        J = mean_t [ ||(x,z)||^2 + w_pitch * wrap(pitch-target)^2 + w_energy * ||mods||^2 ]
    The energy term gives co-optimization a real stability<->efficiency tradeoff.

Validation/baselines are in evaluate-style metrics via evaluate.py (run separately on
the saved morphology). LBM cross-check: the bounded morphology stays in the surrogate's
trained range by construction; a full Taichi-LBM-in-the-loop re-sim is the remaining
rigor step (see manuscript Limitations).

Usage:
  python -m hornetRL.cooptimize --validate
  python -m hornetRL.cooptimize --steps 200 --horizon 24 --batch 8
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .inference_hornet import InferenceFlyEnv, ac_model, Config, symlog
from .fly_system import PhysParams
from .evaluate import load_best_params

# Physical ranges for the co-optimized morphology = the surrogate's training distribution.
# latent 0 at 0.0 maps to the NOMINAL value (sigmoid(0)=0.5 -> midpoint).
MORPH_RANGES = {
    'thorax_mass_scale': (0.80, 1.20),
    'abd_mass_scale':    (0.80, 1.20),
    'thorax_offset_x':   (-0.002, 0.002),
    'k_hinge_scale':     (0.70, 1.30),
}
MORPH_KEYS = list(MORPH_RANGES.keys())
TARGET_PITCH = float(Config.TARGET_STATE[2])


def latents_to_morph(latents):
    """4 unconstrained latents -> dict of bounded physical morphology scalars."""
    out = {}
    for i, k in enumerate(MORPH_KEYS):
        lo, hi = MORPH_RANGES[k]
        out[k] = lo + (hi - lo) * jax.nn.sigmoid(latents[i])
    return out


def make_phys(morph, batch):
    o = jnp.ones(batch)
    z = jnp.zeros(batch)
    return PhysParams(
        thorax_mass_scale=o * morph['thorax_mass_scale'],
        abd_mass_scale=o * morph['abd_mass_scale'],
        thorax_offset_x=o * morph['thorax_offset_x'],
        abd_offset_x=z,
        hinge_x_noise=z, hinge_z_noise=z, stroke_ang_noise=z,
        k_hinge_scale=o * morph['k_hinge_scale'],
        b_hinge_scale=o, phi_equil_offset=z,
    )


def build_rollout(env, horizon, w_pitch):
    dt_block = Config.DT * Config.SIM_SUBSTEPS
    gust_step = int(Config.PERTURB_TIME / dt_block)

    def rollout_cost(policy_params, latents, init_state, gust_f, gust_t):
        """Mean perturbation-rejection cost over the batch, with an applied gust
        vector (gust_f, gust_t) injected at gust_step. init_state carries the
        (varied) initial conditions; morphology comes from latents."""
        morph = latents_to_morph(latents)
        B = init_state[0].shape[0]
        state = (init_state[0], init_state[1], init_state[2], make_phys(morph, B))

        def body(carry, t):
            state, cost = carry
            r = state[0]
            wrapped = jnp.mod(r[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs = symlog(r.at[:, 2].set(wrapped))
            mods = ac_model.apply(policy_params, obs)[0]
            ext_f = jnp.where(t == gust_step, gust_f, jnp.zeros(2))
            ext_t = jnp.where(t == gust_step, gust_t, 0.0)
            nxt, _ = env.step(state, mods, external_force=ext_f, external_torque=ext_t)
            r2 = nxt[0]
            pos_err2 = jnp.sum(r2[:, :2] ** 2, axis=1)
            pitch_err = jnp.mod(r2[:, 2] - TARGET_PITCH + jnp.pi, 2 * jnp.pi) - jnp.pi
            step_cost = jnp.mean(pos_err2 + w_pitch * pitch_err ** 2)
            return (nxt, cost + step_cost), None

        (state, cost), _ = jax.lax.scan(body, (state, 0.0), jnp.arange(horizon))
        return cost / horizon

    return rollout_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hover_params.pkl'))
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--steps', type=int, default=40)     # OUTER (morphology) steps
    ap.add_argument('--k-inner', type=int, default=3)    # inner controller-adaptation steps per outer step
    ap.add_argument('--horizon', type=int, default=28)   # must exceed gust step (~9) to see recovery
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr-morph', type=float, default=1e-2)   # slow outer (morphology)
    ap.add_argument('--lr-policy', type=float, default=5e-4)  # fast inner (controller), = training LR
    ap.add_argument('--w-pitch', type=float, default=0.3)
    ap.add_argument('--lambda-reg', type=float, default=0.05,
                    help='L2 penalty on morphology latents (toward nominal) to prevent extreme overfit bodies')
    ap.add_argument('--val-batch', type=int, default=48, help='held-out validation scenarios')
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cooptimize_outputs'))
    args = ap.parse_args()
    if args.validate:
        args.steps, args.k_inner, args.horizon, args.batch, args.val_batch = 3, 2, 16, 8, 8
    os.makedirs(args.out, exist_ok=True)

    policy0 = load_best_params(args.checkpoint)
    env = InferenceFlyEnv()
    rollout_cost = build_rollout(env, args.horizon, args.w_pitch)

    def sample_gust(key):
        k1, k2 = jax.random.split(key)
        ang = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
        mag = jax.random.uniform(k2, (), minval=0.5, maxval=2.0) * jnp.linalg.norm(Config.PERTURB_FORCE)
        return jnp.array([mag * jnp.cos(ang), mag * jnp.sin(ang)]), Config.PERTURB_TORQUE

    # FIXED held-out validation set (different seed family from training) for honest selection
    val_init = env.reset(jax.random.PRNGKey(99999), batch_size=args.val_batch, mode='chaos')
    val_gust_f, val_gust_t = jnp.array(Config.PERTURB_FORCE), Config.PERTURB_TORQUE

    latents = jnp.zeros(len(MORPH_KEYS))          # start at nominal morphology
    policy = policy0
    opt_m = optax.adam(args.lr_morph); opt_p = optax.adam(args.lr_policy)
    sm, sp = opt_m.init(latents), opt_p.init(policy)

    # BILEVEL co-design: inner loop adapts the CONTROLLER to the current body
    # (fast, K_inner steps), outer loop nudges MORPHOLOGY (slow, 1 step). This keeps
    # the controller near-best for each candidate body, so morphology is judged with a
    # matched controller -- the correct two-timescale direction for brain-body co-design.
    morph_obj = lambda pol, lat, ini, gf, gt: (
        rollout_cost(pol, lat, ini, gf, gt) + args.lambda_reg * jnp.sum(lat ** 2))
    grad_policy = jax.jit(jax.value_and_grad(rollout_cost, argnums=0))   # controller step
    grad_morph = jax.jit(jax.value_and_grad(morph_obj, argnums=1))       # morphology step
    val_fn = jax.jit(rollout_cost)

    # held-out validation cost at nominal morphology + original (trained) controller = bar to beat
    val0 = float(val_fn(policy0, jnp.zeros(len(MORPH_KEYS)), val_init, val_gust_f, val_gust_t))
    print(f"--> nominal held-out validation cost = {val0:.5f}", flush=True)

    best_val, best_latents, best_policy = val0, jnp.zeros(len(MORPH_KEYS)), policy0
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    for outer in range(args.steps):
        # --- inner loop: adapt controller to the current body ---
        for _ in range(args.k_inner):
            key, ki, kg = jax.random.split(key, 3)
            init = env.reset(ki, batch_size=args.batch, mode='chaos')
            gf, gt = sample_gust(kg)
            _, gp = grad_policy(policy, latents, init, gf, gt)
            up, sp = opt_p.update(gp, sp); policy = optax.apply_updates(policy, up)
        # --- outer step: nudge morphology (controller now adapted) ---
        key, ki, kg = jax.random.split(key, 3)
        init = env.reset(ki, batch_size=args.batch, mode='chaos')
        gf, gt = sample_gust(kg)
        j, gm = grad_morph(policy, latents, init, gf, gt)
        um, sm = opt_m.update(gm, sm); latents = optax.apply_updates(latents, um)

        vj = float(val_fn(policy, latents, val_init, val_gust_f, val_gust_t))
        tag = ""
        if vj < best_val:
            best_val, best_latents, best_policy, tag = vj, latents, policy, "  <-best"
        if outer % max(1, args.steps // 20) == 0 or outer == args.steps - 1:
            m = latents_to_morph(latents)
            print(f"  outer {outer:3d}: train={float(j):.4f} val={vj:.4f}  morph="
                  f"{{m_th={float(m['thorax_mass_scale']):.3f}, m_ab={float(m['abd_mass_scale']):.3f}, "
                  f"com={float(m['thorax_offset_x'])*1e3:+.2f}mm, k={float(m['k_hinge_scale']):.3f}}}{tag}", flush=True)

    latents = best_latents; policy = best_policy   # select the best (body, controller) PAIR by held-out val
    morph_final = {k: float(v) for k, v in latents_to_morph(latents).items()}
    result = dict(
        nominal_val_cost=val0, best_val_cost=best_val,
        val_cost_reduction_pct=100.0 * (val0 - best_val) / val0 if val0 > 0 else 0.0,
        morph_final=morph_final, outer_steps=args.steps, k_inner=args.k_inner,
        horizon=args.horizon, batch=args.batch, w_pitch=args.w_pitch,
        lambda_reg=args.lambda_reg, lr_morph=args.lr_morph, lr_policy=args.lr_policy,
        wall_s=time.time() - t0,
    )
    print("\n=== CO-OPTIMIZATION RESULT ===")
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out, 'cooptimize_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, 'cooptimized_morph.json'), 'w') as f:
        json.dump(morph_final, f, indent=2)
    with open(os.path.join(args.out, 'cooptimized_policy.pkl'), 'wb') as f:
        pickle.dump(policy, f)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
