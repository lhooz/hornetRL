"""
F3 — Real evaluation harness for the HornetRL hover controller.

Replaces the FABRICATED Table-1 headline metrics (820 epochs / 98.2% / 3.39 cm),
which appear in NO code/log, with MEASURED quantities that carry stated n and
confidence intervals:

  - hover success rate (binomial Wilson 95% CI)
  - final position error in cm (mean / median + 95% t-CI)
  - settling time to the hover box

The controller is rolled out under DOMAIN-RANDOMIZED physics (the ±20% mass /
±30% stiffness / ±2 mm CoM randomization already in InferenceFlyEnv.reset) plus
the standard gust perturbation at PERTURB_TIME, across many agents x seeds.

Success criterion (single shared definition — also intended for the figure scripts,
to kill the figure4-vs-figure5 threshold divergence):
  a trial SUCCEEDS iff the trajectory stays finite AND, averaged over the final
  SETTLE_WINDOW seconds, ||(x,z)|| < POS_BOX_M and |pitch - target_pitch| < PITCH_BOX_RAD.

Usage:
  python -m hornetRL.evaluate --seeds 5 --agents 64 --duration 0.6
"""
import os, sys, time, json, pickle, argparse, math
import numpy as np
import jax
import jax.numpy as jnp

from .inference_hornet import InferenceFlyEnv, ac_model, Config, symlog

# ---- shared success / crash thresholds (single source of truth) ----
TARGET_PITCH = float(Config.TARGET_STATE[2])   # 1.0 rad (57.3 deg)
POS_BOX_M = 0.05        # 5 cm hover box on ||(x,z)||
PITCH_BOX_RAD = 0.30    # ~17 deg pitch tolerance
SETTLE_WINDOW_S = 0.15  # averaging window at the end of the rollout


def load_best_params(param_file):
    with open(param_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'params' in data:
        if 'pbt_state' in data:
            idx = int(np.argmax(np.array(data['pbt_state'].running_reward)))
            print(f"--> PBT population: best agent {idx} "
                  f"(reward {float(np.array(data['pbt_state'].running_reward)[idx]):.2f})")
            return jax.tree.map(lambda x: x[idx], data['params'])
        return jax.tree.map(lambda x: x[0], data['params'])
    return data  # already a bare param pytree


def rollout(params, env, key, batch_size, duration, ablation=False, phys_override=None, mode='nominal'):
    """Roll out `batch_size` domain-randomized agents; return robot_state trajectory
    (T, batch, 8) and the per-control-step time array.
    ablation=True zeros the policy modulations -> open-loop-CPG ablation (the same
    comparator used in run_generalization_sweeps.py:108).
    phys_override: optional PhysParams (batched) to replace the sampled morphology -- used
    by the F4 morphology comparison to evaluate a FIXED body across seeds/agents.
    mode: 'nominal' (start at hover) or 'chaos' (varied initial perturbations)."""
    state = env.reset(key, batch_size=batch_size, mode=mode)
    if phys_override is not None:
        state = (state[0], state[1], state[2], phys_override)
    total_steps = int(duration / (Config.DT * Config.SIM_SUBSTEPS))

    @jax.jit
    def step(curr_state, ext_f, ext_t):
        r = curr_state[0]
        wrapped = jnp.mod(r[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs = symlog(r.at[:, 2].set(wrapped))
        mods, _, _ = ac_model.apply(params, obs)
        if ablation:
            mods = jnp.zeros_like(mods)
        nxt, _ = env.step(curr_state, mods, external_force=ext_f, external_torque=ext_t)
        return nxt

    _ = step(state, jnp.zeros(2), 0.0)  # warmup/compile
    traj, times = [], []
    t_sim = 0.0
    dt_block = Config.DT * Config.SIM_SUBSTEPS
    for i in range(total_steps):
        ext_f, ext_t = jnp.zeros(2), 0.0
        if Config.PERTURBATION:
            be = t_sim + dt_block
            if (t_sim <= Config.PERTURB_TIME + 0.002) and (be >= Config.PERTURB_TIME):
                ext_f, ext_t = Config.PERTURB_FORCE, Config.PERTURB_TORQUE
        state = step(state, ext_f, ext_t)
        traj.append(np.array(state[0]))
        times.append(t_sim)
        t_sim += dt_block
    return np.stack(traj), np.array(times)  # (T, B, 8), (T,)


def metrics_from_traj(traj, times):
    """Per-agent metrics from a (T,B,8) trajectory."""
    T, B, _ = traj.shape
    pos = traj[:, :, :2]                       # x, z
    pitch = traj[:, :, 2]
    # pitch is an angle: wrap (pitch - target) to [-pi, pi] so a 2*pi rotation
    # through a gust is not mistaken for divergence.
    pitch_err = np.abs((pitch - TARGET_PITCH + np.pi) % (2 * np.pi) - np.pi)
    dist = np.sqrt((pos ** 2).sum(axis=2))     # (T,B) ||(x,z)||
    finite = np.isfinite(traj).all(axis=(0, 2))  # (B,)

    win = max(1, int(SETTLE_WINDOW_S / (times[1] - times[0]))) if T > 1 else 1
    final_dist = np.nanmean(dist[-win:], axis=0)             # (B,)
    final_pitch_err = np.nanmean(pitch_err[-win:], axis=0)

    success = finite & (final_dist < POS_BOX_M) & (final_pitch_err < PITCH_BOX_RAD)

    # settling time: first time dist stays < POS_BOX_M through the end
    settle = np.full(B, np.nan)
    for b in range(B):
        if not finite[b]:
            continue
        below = dist[:, b] < POS_BOX_M
        for ti in range(T):
            if below[ti:].all():
                settle[b] = times[ti]
                break
    return dict(success=success, final_dist_cm=final_dist * 100.0,
                final_pitch_err_deg=np.degrees(final_pitch_err), settle_s=settle,
                finite_frac=float(np.mean(finite)))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def t_ci(x, conf=0.95):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float(np.mean(x)) if n else float('nan'), float('nan'), float('nan'))
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    # 95% normal approx (n is large here)
    h = 1.96 * sd / math.sqrt(n)
    return (m, m - h, m + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hover_params.pkl'))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--agents', type=int, default=64)
    ap.add_argument('--duration', type=float, default=0.6)
    ap.add_argument('--ablation', action='store_true',
                    help='zero policy mods -> open-loop-CPG ablation (F5a/F2 comparator)')
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_outputs'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"--> Loading controller: {args.checkpoint}")
    params = load_best_params(args.checkpoint)
    env = InferenceFlyEnv()
    print(f"--> Evaluating {args.seeds} seeds x {args.agents} agents "
          f"(domain-randomized + gust), duration {args.duration}s")

    all_succ, all_dist, all_pitch, all_settle = [], [], [], []
    per_seed_succ = []
    t0 = time.time()
    for s in range(args.seeds):
        key = jax.random.PRNGKey(1000 + s)
        traj, times = rollout(params, env, key, args.agents, args.duration, ablation=args.ablation)
        m = metrics_from_traj(traj, times)
        per_seed_succ.append(float(np.mean(m['success'])))
        all_succ.append(m['success']); all_dist.append(m['final_dist_cm'])
        all_pitch.append(m['final_pitch_err_deg']); all_settle.append(m['settle_s'])
        print(f"  seed {s}: success={np.mean(m['success'])*100:.1f}%  "
              f"median_pos={np.nanmedian(m['final_dist_cm']):.2f}cm", flush=True)

    succ = np.concatenate(all_succ); dist = np.concatenate(all_dist)
    pitch = np.concatenate(all_pitch); settle = np.concatenate(all_settle)
    n = len(succ); k = int(succ.sum())
    lo, hi = wilson_ci(k, n)
    succ_dist = dist[succ]  # error among successful (matches "hover pos error")
    pm, pl, ph = t_ci(succ_dist if len(succ_dist) else dist)

    result = dict(
        mode='ablation_open_loop_cpg' if args.ablation else 'full_controller',
        n_trials=n, n_seeds=args.seeds, n_agents=args.agents,
        success_rate=k / n, success_rate_ci95=[lo, hi],
        per_seed_success=per_seed_succ,
        pos_error_cm_mean=pm, pos_error_cm_ci95=[pl, ph],
        pos_error_cm_median=float(np.nanmedian(succ_dist if len(succ_dist) else dist)),
        pitch_err_deg_median=float(np.nanmedian(pitch[np.isfinite(pitch)])),
        settle_s_median=float(np.nanmedian(settle[np.isfinite(settle)])) if np.isfinite(settle).any() else None,
        criterion=dict(pos_box_m=POS_BOX_M, pitch_box_rad=PITCH_BOX_RAD,
                       settle_window_s=SETTLE_WINDOW_S, target_pitch_rad=TARGET_PITCH),
        wall_s=time.time() - t0,
    )
    print(f"\n=== MEASURED controller metrics [{result['mode']}] ===")
    print(json.dumps(result, indent=2))
    fname = 'eval_result_ablation.json' if args.ablation else 'eval_result.json'
    with open(os.path.join(args.out, fname), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {os.path.join(args.out, fname)}")


if __name__ == '__main__':
    main()
