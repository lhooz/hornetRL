"""
F4 baselines — does the co-optimized body actually help?

Evaluates the trained controller under several FIXED morphologies, measuring
perturbation-rejection (recovery from 'chaos' initial conditions + in-rollout gust)
with the evaluate.py success criterion:

  - co-optimized   (from cooptimize_outputs/cooptimized_morph.json)
  - nominal        (all scales = 1.0, CoM = 0)
  - random x N     (sampled uniformly within the surrogate-valid ranges)
  - best-fixed     (coarse grid search over mass x stiffness, picked by the same metric)

This isolates the co-design advantage from the controller's own merit, as the
reviewers will demand. Run with the hornetRL venv.
"""
import os, json, argparse, pickle
import numpy as np
import jax
import jax.numpy as jnp

from .inference_hornet import InferenceFlyEnv
from .evaluate import load_best_params, rollout, metrics_from_traj, wilson_ci
from .cooptimize import make_phys, MORPH_KEYS, MORPH_RANGES

NOMINAL = {'thorax_mass_scale': 1.0, 'abd_mass_scale': 1.0,
           'thorax_offset_x': 0.0, 'k_hinge_scale': 1.0}


def eval_morph(params, env, morph, seeds, agents, duration):
    """params may differ per body: the co-optimized body is judged with its MATCHED
    co-optimized controller, baselines with the original trained controller."""
    succ, dist = [], []
    for s in range(seeds):
        key = jax.random.PRNGKey(7000 + s)
        phys = make_phys({k: jnp.asarray(float(v)) for k, v in morph.items()}, agents)
        traj, times = rollout(params, env, key, agents, duration,
                              phys_override=phys, mode='chaos')
        m = metrics_from_traj(traj, times)
        succ.append(m['success']); dist.append(m['final_dist_cm'])
    succ = np.concatenate(succ); dist = np.concatenate(dist)
    n, k = len(succ), int(succ.sum())
    lo, hi = wilson_ci(k, n)
    return dict(success_rate=k / n, ci95=[lo, hi], n=n,
                median_recovery_cm=float(np.nanmedian(dist[np.isfinite(dist)])),
                mean_recovery_cm=float(np.nanmean(dist[np.isfinite(dist)])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hover_params.pkl'))
    ap.add_argument('--cooptimized', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cooptimize_outputs', 'cooptimized_morph.json'))
    ap.add_argument('--cooptimized-policy', type=str, default=None,
                    help='matched co-optimized controller (cooptimized_policy.pkl) for the co-optimized body')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--agents', type=int, default=32)
    ap.add_argument('--duration', type=float, default=0.6)
    ap.add_argument('--out', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cooptimize_outputs'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = load_best_params(args.checkpoint)
    env = InferenceFlyEnv()
    rng = np.random.RandomState(0)

    morphs = {'nominal': NOMINAL}
    if os.path.exists(args.cooptimized):
        morphs['co-optimized'] = json.load(open(args.cooptimized))
    for i in range(3):
        morphs[f'random_{i}'] = {k: float(rng.uniform(*MORPH_RANGES[k])) for k in MORPH_KEYS}

    # matched co-optimized controller for the co-optimized body (fair co-design comparison)
    co_policy = params
    if args.cooptimized_policy and os.path.exists(args.cooptimized_policy):
        with open(args.cooptimized_policy, 'rb') as f:
            co_policy = pickle.load(f)
        print(f"--> co-optimized body will use its MATCHED controller: {args.cooptimized_policy}")

    results = {}
    for name, m in morphs.items():
        p = co_policy if name == 'co-optimized' else params
        r = eval_morph(p, env, m, args.seeds, args.agents, args.duration)
        results[name] = {'morph': m, **r}
        print(f"  {name:14s}: success={r['success_rate']*100:5.1f}% "
              f"CI{[round(x*100,1) for x in r['ci95']]}  median_recovery={r['median_recovery_cm']:.2f}cm", flush=True)

    # coarse best-fixed grid over mass x stiffness (1 seed for speed)
    print("--> best-fixed grid search (mass x stiffness)...", flush=True)
    best = None
    for mscale in [0.85, 1.0, 1.15]:
        for kscale in [0.8, 1.0, 1.2]:
            m = {'thorax_mass_scale': mscale, 'abd_mass_scale': mscale,
                 'thorax_offset_x': 0.0, 'k_hinge_scale': kscale}
            r = eval_morph(params, env, m, 1, args.agents, args.duration)
            if best is None or r['success_rate'] > best[1]['success_rate']:
                best = (m, r)
    results['best_fixed_grid'] = {'morph': best[0], **best[1]}
    print(f"  best-fixed     : success={best[1]['success_rate']*100:5.1f}%  "
          f"morph={best[0]}", flush=True)

    with open(os.path.join(args.out, 'morphology_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY (perturbation-rejection success rate) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]['success_rate']):
        print(f"  {name:16s} {r['success_rate']*100:5.1f}%   median_recovery={r['median_recovery_cm']:.2f}cm")
    print(f"saved -> {os.path.join(args.out, 'morphology_comparison.json')}")


if __name__ == '__main__':
    main()
