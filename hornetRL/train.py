import os
import time
# Force CPU for stability/Memory 
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from typing import NamedTuple
from functools import partial

# --- USER MODULES ---
from .fluid_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem, PhysParams
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action, ScaleConfig
from .pbt_manager import init_pbt_state, pbt_evolve
from .env import FlyEnv

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """
    Hyperparameters and physical constraints for the training pipeline.
    """
    SEED = 42

    # --- Master Frequency Setting ---
    # Base operating frequency for the Central Pattern Generator (Hz).
    BASE_FREQ = 115.0  
    
    # Target State: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    # Goal: Stationary hover at specific altitude with upright posture.
    TARGET_STATE = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    # --- Arena Boundaries ---
    # Absolute displacement limit (meters) before episode termination.
    # 0.45m corresponds to a 90cm total workspace width.
    ARENA_W = 0.45 

    # --- Time Scales ---
    DT = 3e-5               # Physics integration timestep (s)
    SIM_SUBSTEPS = 20       # Physics steps per Control Step
                            # Effective Control Freq: ~1666 Hz (0.6ms)
                            
    HORIZON = 64            # Trajectory horizon for Back-propagation Through Time (BPTT).
                            # Duration: ~0.038s (~4.4 wingbeats), sufficient for stability convergence.
    RESET_INTERVAL = 50     # Forced reset interval (epochs) to enforce takeoff robustness.
    PBT_INTERVAL = 1000      # Evolution interval (Survival of the fittest)

    BATCH_SIZE = 32          # Number of parallel environments
    LR_ACTOR = 5e-4         # Learning Rate
    MAX_GRAD_NORM = 1.0     # Gradient Clipping threshold
    GAMMA = 0.99            # Discount Factor
    OBS_NOISE_SIGMA = 0.01  # Observation noise sigma
    TOTAL_UPDATES = 100000   # Total Gradient Steps

    # % Nominal (Hover), % Chaos (Recovery)
    # Set to 1.0 for easier initial training, 0.8 for robustness
    CURRICULUM_RATIO = 0.5
    
    # --- PBT Hyperparameters ---
    # Initial Reward Weights, act as the "center" of the search distribution.
    PBT_BASE_WEIGHTS = jnp.array([
        200.0,    # Pos (The "Pot of Gold" max value)
        10.0,     # Th_Ang (Orientation penalty)
        1.0,     # Ab_Ang (Abdomen stability)
        0.1,     # Lin_Vel (Drift damping)
        0.001,     # Ang_Vel (Vibration damping)
        0.1      # Eff (Force efficiency)
    ])
    
    # Evolution Dynamics
    PBT_PERTURB_FACTOR = 1.2       # Mutation strength (+/- 20%)
    PBT_TRUNCATE_FRACTION = 0.2    # Kill bottom 20%

    CKPT_DIR = "checkpoints_shac"
    VIS_DIR = "checkpoints_shac"
    AUX_LOSS_WEIGHT = 10.0   
    VIS_INTERVAL = 200      
    
    WARMUP_STEPS = 1        # Control steps to pin the fly before releasing dynamics.

    FORCE_NORMALIZER = ScaleConfig.CONTROL_SCALE

# --- Observation Scaling SYMLOG---
def symlog(x):
    """
    Symmetric Log scaling.
    Compresses large magnitudes while preserving small differences near zero.
    Range: Real Numbers -> [-inf, inf] (but mostly compressed to [-10, 10])
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state, force_noise=None):
    """
    Defines the Actor-Critic architecture.
    
    Actor: Input Convex Neural Network (ICNN) wrapped in IDA-PBC logic.
    Critic: Standard MLP Value Function.
    """
    
    # 1. Prepare Target in SymLog Space
    target_sym = symlog(Config.TARGET_STATE)

    # 2. Actor (Brain + Muscles)
    # Pass None to obs_scale because we already handled scaling via SymLog
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=target_sym,
        force_noise=force_noise
    )
    
    # 3. Critic
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(robot_state)
    
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. VISUALIZATION ENGINE
# ==============================================================================
def run_visualization(env, params, update_idx, vis_step_fn):
    """
    Generates a GIF visualization of the flight trajectory and aerodynamic forces.
    """
    print(f"--> Generatng Visualization for Step {update_idx}...")

    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 4

    sim_data = {'states': [], 'wing_pose': [], 'nodal_forces': [], 'le_marker': [], 'hinge_marker': [], 't': []}
    
    rng = jax.random.PRNGKey(update_idx)
    state = env.reset(rng, 1) 
    
    # Extract parameters for the specific fly being visualized (Index 0)
    active_props_batch = state[3] 
    
    # Take the 0-th element of the batch
    # This IS the real geometry. No need to call compute_props again.
    real_props = jax.tree.map(lambda x: x[0], active_props_batch)
    
    # Extract the 0-th Brain from the population to visualize
    params_single = jax.tree.map(lambda x: x[0], params)

    current_step_counter = 0

    for i in range(total_visual_frames):
        # --- Visualization Loop ---
        for _ in range(steps_per_frame):
            r_st = state[0]
            
            # NaN Safety Check
            r_cpu = np.array(r_st[0])
            if np.isnan(r_cpu).any():
                print(f"!!! Visualization stopped early due to NaN !!!")
                break
            
            # Environment Step
            state, f_nodal, w_pose, h_marker = vis_step_fn(env, state, params_single, current_step_counter)
            current_step_counter += 1

        # Record Frame Data
        sim_data['states'].append(np.array(state[0][0])) 
        sim_data['t'].append(current_step_counter * Config.DT)
        
        f_st = state[1]
        sim_data['le_marker'].append(np.array(f_st.marker_le[0]))
        sim_data['wing_pose'].append(np.array(w_pose[0]))
        sim_data['nodal_forces'].append(np.array(f_nodal[0]))
        sim_data['hinge_marker'].append(np.array(h_marker[0]))

    # --- Matplotlib Animation Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # Body Segments
    patch_thorax = patches.Ellipse((0,0), linewidth=1.0, width=0.012, height=0.006, facecolor='#333333', edgecolor='black', zorder=10)
    patch_head = patches.Circle((0,0), linewidth=1.0, radius=0.0025, facecolor='#00FF00', edgecolor='black', zorder=10)
    patch_abd = patches.Ellipse((0,0), linewidth=1.0, width=0.018, height=0.008, facecolor='#1f77b4', edgecolor='black', alpha=0.8, zorder=9)
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)

    real_line, = ax.plot([], [], 'k-', linewidth=1.0, alpha=0.8, zorder=12)
    patch_le = patches.Circle((0,0), radius=0.001, color='red', zorder=15, label='Leading Edge')
    ax.add_patch(patch_le)

    # Hinge Marker
    patch_hinge = patches.Circle((0,0), radius=0.001, color='orange', zorder=15, label='Hinge')
    ax.add_patch(patch_hinge)

    # Force Vectors (Quiver)
    dummy = np.zeros(20)
    quiver = ax.quiver(dummy, dummy, dummy, dummy, color='red', scale=3.0, scale_units='xy', zorder=20, width=0.0002)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')
    ax.grid(True, linestyle=':', alpha=0.3)

    def update(frame):
        if frame >= len(sim_data['states']): return
        
        r_state = sim_data['states'][frame]
        w_pose = sim_data['wing_pose'][frame]
        f_nodal = sim_data['nodal_forces'][frame]
        le_pos = sim_data['le_marker'][frame]
        hinge_pos = sim_data['hinge_marker'][frame]
        t = sim_data['t'][frame]
        
        rx, rz = r_state[0], r_state[1]
        r_th, r_phi = r_state[2], r_state[3]
        
        # Follow Camera
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        
        d1 = real_props.d1
        d2 = real_props.d2

        # 1. Update Thorax
        patch_thorax.set_center((rx, rz))
        patch_thorax.set_angle(np.degrees(r_th))
        
        # 2. Update Head
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        
        # 3. Update Abdomen (Kinematic Chain)
        joint_x = rx - d1 * np.cos(r_th)
        joint_z = rz - d1 * np.sin(r_th)
        
        abd_ang = r_th + r_phi
        patch_abd.set_center((joint_x - d2 * np.cos(abd_ang), joint_z - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))
        
        # 4. Update Wing
        wx, wz, wang = w_pose
        wing_len = env.phys.fluid.WING_LEN
        N_pts = env.phys.fluid.N_PTS
        x_local = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        wing_x = wx + x_local * c_w
        wing_z = wz + x_local * s_w
        real_line.set_data(wing_x, wing_z)
        
        patch_le.set_center((le_pos[0], le_pos[1]))
        patch_hinge.set_center((hinge_pos[0], hinge_pos[1]))
        
        pts = np.stack([wing_x, wing_z], axis=1)
        quiver.set_offsets(pts)
        quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
        
        time_text.set_text(f"T: {t:.4f}s | Y: {rz:.3f}")
        return patch_thorax, patch_le, patch_hinge

    ani = animation.FuncAnimation(fig, update, frames=len(sim_data['states']), interval=20, blit=False)
    out_file = os.path.join(Config.VIS_DIR, f"epoch_{update_idx}.gif")
    ani.save(out_file, writer='pillow', fps=60)
    plt.close(fig)
    print(f"--> Saved Viz: {out_file}")

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train():
    """
    Main training loop implementing Short Horizon Actor-Critic (SHAC).
    Performs trajectory rollouts, computes gradients via BPTT, and updates policies.
    """
    
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    
    env = FlyEnv(Config)
    rng = jax.random.PRNGKey(Config.SEED)
    
    dummy_input = jnp.zeros((1, 8))

    # ---------------------------------------------------------
    # 1. DEFINE LOADING HIERARCHY
    # ---------------------------------------------------------
    checkpoints = glob.glob(os.path.join(Config.CKPT_DIR, "*.pkl"))
    hornet_path = "hornet_brain.pkl" # <--- Put your expert brain file here
    
    start_step = 0
    
    # --- SCENARIO A: RESUME TRAINING (Highest Priority) ---
    if checkpoints:
        checkpoints.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        last_ckpt = checkpoints[-1]
        print(f"--> [RESUME] Found checkpoint: {last_ckpt}")
        
        with open(last_ckpt, "rb") as f:
            data = pickle.load(f)
        
        # Load Population
        params = data['params'] 
        opt_state = data['opt_state']
        
        # Load PBT State (Critical for persistence)
        if 'pbt_state' in data:
            pbt_state = data['pbt_state']
            print("    -> PBT State loaded.")
        else:
            # Fallback for old checkpoints
            pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)
            print("    -> WARNING: No PBT state found. Resetting PBT curriculum.")

        match = re.search(r"params_(\d+)", last_ckpt)
        if match: start_step = int(match.group(1)) + 1
        
        # Re-create optimizer structure (stateless) to match loaded state
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )

    # --- SCENARIO B: NEW RUN WITH HORNET BRAIN (Transfer Learning) ---
    elif os.path.exists(hornet_path):
        print(f"--> [TRANSFER] No checkpoint found. Loading Expert: {hornet_path}")
        
        with open(hornet_path, "rb") as f:
            expert_data = pickle.load(f)
            
        single_params = expert_data['params']
        
        # TILE THE BRAIN: (Layer) -> (Batch, Layer)
        # This creates 32 independent copies of the expert
        params = jax.tree.map(lambda x: jnp.stack([x] * Config.BATCH_SIZE), single_params)
        
        # Create fresh optimizer for this population
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )
        opt_state = optimizer.init(params)
        
        # Start PBT from scratch
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    # --- SCENARIO C: FRESH START (Random Init) ---
    else:
        print("--> [SCRATCH] No checkpoint or expert found. Initializing random population.")
        
        # Init single random brain
        single_params = ac_model.init(rng, dummy_input)
        
        # Tile it (so code structure is consistent with Scenario B)
        params = jax.tree.map(lambda x: jnp.stack([x] * Config.BATCH_SIZE), single_params)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )
        opt_state = optimizer.init(params)
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    print(f"--> Initialization Complete. Params Batch Shape: {params['linear']['w'].shape}")

    def loss_fn(params, start_state, pbt_weights, key):
        """
        Computes the total loss over the trajectory horizon.
        Includes policy gradient, value function loss, and auxiliary force matching loss.
        """
        # Generate temporal keys for noise injection
        # 1. Rollout Index: 0..H (Used for Gradient Masking)
        rollout_indices = jnp.arange(Config.HORIZON)
        
        # 2. Physics Index: H..2H (Used for Physics Dynamics)
        # By adding a large offset (WARMUP_STEPS + 1), we ensure the physics engine 
        # always sees a value > WARMUP_STEPS, preventing it from triggering the 
        # "Velocity Pinning" or "Force Ramping" logic during continuous flight.
        phys_indices = rollout_indices + Config.WARMUP_STEPS + 5
        
        scan_keys = jax.random.split(key, Config.HORIZON)
        
        # Pass BOTH indices to the scan
        scan_inputs = (rollout_indices, phys_indices, scan_keys)

        def scan_fn(carry, xs): 
            curr_full = carry
            r_idx, p_idx, step_key = xs
            
            curr_robot = curr_full[0]
            # 1. Wrap Theta (Index 2) to [-pi, pi]
            wrapped_theta = jnp.mod(curr_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs_robot = curr_robot.at[:, 2].set(wrapped_theta)

            # 2. Normalize Observation
            scaled_obs = symlog(obs_robot)
            
            # --- Noise Injection (Sensor Model) ---
            # Simulates sensor noise to promote robust control policies.
            noise_sigma = Config.OBS_NOISE_SIGMA
            obs_noise = jax.random.normal(step_key, shape=scaled_obs.shape) * noise_sigma
            noisy_obs = scaled_obs + obs_noise
            
            # --- GENERATE ACTION NOISE ---
            # 1. Split key
            key_noise, key_step = jax.random.split(step_key)
            
            # 2. Define Noise Scale (e.g., 25% of max force)
            # Use Config.FORCE_NORMALIZER so the noise is in Newtons
            noise_sigma = Config.FORCE_NORMALIZER * 0.25
            
            # 3. Sample Gaussian Noise
            # Shape should match u_forces: [Batch, 4]
            force_noise = jax.random.normal(key_noise, shape=(Config.BATCH_SIZE, 4)) * noise_sigma

            # 3. Policy Inference (Noisy Input -> Smooth Action)
            # This allows Agent 1 to use Brain 1 on Obs 1, Agent 2 on Brain 2, etc.
            batched_network = jax.vmap(ac_model.apply)
            
            # Apply: Brain[i] sees Observation[i]
            mods, u_brain, _ = batched_network(params, noisy_obs, force_noise)
            
            # 4. Environment Step (Physics uses raw actions)
            next_full, f_actual, _, _, _ = env.step_batch(curr_full, mods, step_idx=p_idx)
            
            # 5. Reward Calculation
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain, pbt_weights)
            
            force_err = jnp.mean((u_brain[:, :3] - f_actual[:, :3])**2)
            loss_t = -rew_scaled + (Config.AUX_LOSS_WEIGHT * force_err)
            
            step_metrics = (
                loss_t, rew_scaled, force_err, 
                met['rew'], met['pos'], 
                met['ang_th'], met['ang_ab'], 
                met['vel_lin'], met['vel_ang']
            )
            return next_full, step_metrics

        final_full, step_results = jax.lax.scan(
            scan_fn, start_state, scan_inputs
        )
        
        # Unpack scan results
        (losses, rewards_scaled, f_errs, real_rews, m_pos, 
         m_ang_th, m_ang_ab, m_vel_lin, m_vel_ang) = step_results

        # Mask losses during Warmup
        warmup_mask = rollout_indices >= Config.WARMUP_STEPS
        losses = jnp.where(warmup_mask[:, None], losses, 0.0)

        # 1. Discounted Loss (Actor)
        discounts = Config.GAMMA ** jnp.arange(Config.HORIZON)
        weighted_loss = jnp.dot(discounts, losses)
        
        # 2. Terminal Value Bootstrap
        final_robot = final_full[0]

        # Wrap and Scale terminal state (Raw -> Obs)
        f_wrapped_th = jnp.mod(final_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        final_obs = symlog(final_robot.at[:, 2].set(f_wrapped_th))

        _, _, final_val_actor = jax.vmap(ac_model.apply)(params, final_obs)
        final_val_actor = jnp.squeeze(final_val_actor)
        
        actor_term = jnp.mean(weighted_loss - (Config.GAMMA**Config.HORIZON * final_val_actor))

        # 3. Critic Loss (Huber)
        final_val_target = jax.lax.stop_gradient(final_val_actor)
        discounted_return = jnp.dot(discounts, rewards_scaled) + (Config.GAMMA**Config.HORIZON * final_val_target)
        
        # Wrap and Scale start state (Raw -> Obs)
        start_robot = start_state[0]
        s_wrapped_th = jnp.mod(start_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        start_obs = symlog(start_robot.at[:, 2].set(s_wrapped_th))

        _, _, start_val = jax.vmap(ac_model.apply)(params, start_obs)
        start_val = jnp.squeeze(start_val)
        
        critic_loss = optax.huber_loss(start_val, discounted_return, delta=1.0)
        critic_loss = jnp.mean(critic_loss)
        
        total_loss = actor_term + (0.5 * critic_loss)
        
        logs = {
            'rew': jnp.mean(real_rews),
            'rew_per_agent': jnp.mean(real_rews, axis=0),
            'ferr': jnp.mean(f_errs),
            'pos': jnp.mean(m_pos),
            # NEW: Keep batch dimension for PBT ranking
            'pos_per_agent': jnp.mean(m_pos, axis=0), 
            'ang_th': jnp.mean(m_ang_th),
            'ang_ab': jnp.mean(m_ang_ab),
            'vel_lin': jnp.mean(m_vel_lin),
            'vel_ang': jnp.mean(m_vel_ang),
            'act_loss': actor_term,
            'crit_loss': critic_loss
        }
        return total_loss, (logs, final_full)

    @jax.jit
    def update(params, opt_state, full_state, pbt_state, key): 
        # Split key: one for loss noise, one for next step exploration
        key_loss, key_next = jax.random.split(key)
        
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, full_state, pbt_state.weights, key_loss
        )
        
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Update PBT Running Reward (EMA)
        # Use the raw position error from the logs. 
        # We negate it so that "Small Error" becomes "High Score" (e.g. -0.01 > -5.0)
        current_score = -1.0 * (logs['pos_per_agent'] * 100.0)

        new_running = 0.8 * pbt_state.running_reward + 0.2 * current_score
        new_pbt_state = pbt_state._replace(running_reward=new_running)

        return new_params, new_opt, loss, logs, next_state, new_pbt_state, key_next

    print(f"=== Starting Training: Step {start_step} to {Config.TOTAL_UPDATES} ===")
    rng, key_reset = jax.random.split(rng)
    curr_state = env.reset(key_reset, Config.BATCH_SIZE)
    
    # --- JIT Compilation ---
    print("--> Compiling JAX Update...")
    t0 = time.time()
    key_compile = jax.random.PRNGKey(0)
    _ = update(params, opt_state, curr_state, pbt_state, key_compile) 
    print(f"--> Compilation Finished in {time.time() - t0:.2f}s")

    # --- Main Loop ---
    key_explore = jax.random.PRNGKey(999) 
    t_start = time.time()
    
    for i in range(start_step, Config.TOTAL_UPDATES):
        t0 = time.time() 

        if i == start_step:
            print(f"DEBUG: Env Target Theta: {env.target[2]:.4f}")
            print(f"DEBUG: Spawn Theta: {curr_state[0][0, 2]:.4f}")

        # 1. Update Step
        params, opt_state, loss, logs, next_state, pbt_state, key_explore = update(params, opt_state, curr_state, pbt_state, key_explore)
    
        # 2. Stability Checks
        if jnp.isnan(loss):
            print(f"!!! CRITICAL: NaN detected at step {i} !!! Reseting batch.")
            rng, k_res = jax.random.split(rng)
            curr_state = env.reset(k_res, Config.BATCH_SIZE)
            continue
        
        r_state = next_state[0]
        
        # --- Environment Reset Logic ---
        # A. Detect Crashes & NaNs
        is_nan = jnp.isnan(r_state).any(axis=1)
        is_crashed = (jnp.abs(r_state[:, 0]) > Config.ARENA_W) | (jnp.abs(r_state[:, 1]) > Config.ARENA_W)
        
        reset_mask = is_nan | is_crashed
    
        rng, k_res = jax.random.split(rng)
        fresh_state = env.reset(k_res, Config.BATCH_SIZE)
    
        curr_state = jax.tree.map(
            lambda x, y: jnp.where(jnp.reshape(reset_mask, (-1,) + (1,)*(x.ndim-1)), y, x),
            next_state, fresh_state
        )

        # 3. Telemetry & Logging
        dt_epoch = time.time() - t0              
        total_elapsed = time.time() - t_start    

        sample_robot = next_state[0][0] 
        sample_osc = jax.tree.map(lambda x: x[0], next_state[2])

        # Diagnostic: Check Brain's perspective
        wrapped_theta = jnp.mod(sample_robot[2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_sample = sample_robot.at[2].set(wrapped_theta)
        scaled_sample = symlog(obs_sample)

        params_sample = jax.tree.map(lambda x: x[0], params)
        
        # Now apply single brain to single observation
        sample_mods, _, _ = ac_model.apply(params_sample, scaled_sample)

        angles, _, _, _ = get_wing_kinematics(sample_osc, unpack_action(sample_mods))
        str_angle, dev_angle, pit_angle = angles
    
        mean_x = jnp.mean(curr_state[0][:, 0])
        mean_z = jnp.mean(curr_state[0][:, 1])
    
        # Velocity Breakdown
        lin_vels = next_state[0][:, 4:6]
        lin_mag = jnp.mean(jnp.sqrt(jnp.sum(lin_vels**2, axis=1)))

        ang_vels = next_state[0][:, 6:8]
        
        # Energy Diagnostics
        raw_hum_energy = jnp.mean(jnp.sum(ang_vels**2, axis=1))
        thorax_vel = next_state[0][:, 6]
        thorax_mag = jnp.mean(jnp.abs(thorax_vel))

        # --- A. PBT EVOLUTION (Slow Cycle) ---
        # Triggered less frequently to allow learning/exploration
        performed_pbt = False
        if i % Config.PBT_INTERVAL == 0 and i > 0:
            print(f"--> PBT EVOLUTION (Step {i})")
            rng, k_pbt = jax.random.split(rng)

            # Print best weights
            best_idx = jnp.argmax(pbt_state.running_reward)
            best_score = pbt_state.running_reward[best_idx]
            print(f"    Best Score: {best_score:.2f} cm | Weights: {pbt_state.weights[best_idx]}")
        
            params, opt_state, pbt_state = pbt_evolve(
                k_pbt, 
                params, 
                opt_state, 
                pbt_state,
                perturb_factor=Config.PBT_PERTURB_FACTOR,
                truncate_fraction=Config.PBT_TRUNCATE_FRACTION
            )
            performed_pbt = True # Mark that we just changed brains

        # --- B. FORCED RESET (Fast Cycle) ---
        # Triggered frequently to practice takeoff robustness
        # OR triggered immediately if PBT happened (Mutants need a fresh start)
        if (i % Config.RESET_INTERVAL == 0 and i > 0) or performed_pbt:
            if performed_pbt:
                print("    -> PBT Mutation applied. Forcing Environment Reset.")
            
            # Force Environment Reset
            curr_state = env.reset(rng, Config.BATCH_SIZE)

        print(f"Step {i:04d} | Epoch: {dt_epoch:.2f}s | Total: {total_elapsed/60:.1f}min | "
              f"Loss: {loss:.1e} (A:{logs['act_loss']:.1e} C:{logs['crit_loss']:.1e}) | "
              f"Rew: {logs['rew']:.1f}\n"
              f"    -> Errs[Pos:{logs['pos']:.2f} Th:{logs['ang_th']:.2f} Ab:{logs['ang_ab']:.2f} "
              f"LVel:{logs['vel_lin']:.2f} AVel:{logs['vel_ang']:.2f} Frc:{logs['ferr']:.4f}] | "
              f"MeanPos: [{mean_x:+.2f}, {mean_z:+.2f}]\n"
              f"    -> Phys: [Hum:{raw_hum_energy:.0f} | ThVel:{thorax_mag:.2f} rad/s]")
    
        if i % Config.VIS_INTERVAL == 0:
            ckpt_path = os.path.join(Config.CKPT_DIR, f"shac_params_{i}.pkl")
            with open(ckpt_path, "wb") as f:
                # UPDATED: Save pbt_state too
                pickle.dump({
                    'params': params, 
                    'opt_state': opt_state,
                    'pbt_state': pbt_state 
                }, f)
            print(f"--> Saved Checkpoint: {ckpt_path}")
            run_visualization(env, params, i, vis_step_fn)

# Global Visualization Step for JIT efficiency
# static_argnums=(0,) tells JAX: "The 0th argument (env) is a Python object/class, 
# not an array. Bake it into the compiled code as a constant."
@partial(jax.jit, static_argnums=(0,))
def vis_step_fn(env, curr_state, curr_params, step_idx):
    r_st = curr_state[0]
    
    # 1. Prepare Observation (Same logic as before)
    wrapped_th = jnp.mod(r_st[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
    obs_v = r_st.at[:, 2].set(wrapped_th)
    scaled_obs = symlog(obs_v)
    
    # 2. Policy Inference (Uses global ac_model)
    mods, _, _ = ac_model.apply(curr_params, scaled_obs)
    
    # 3. Env Step (Uses the passed 'env' object)
    next_state, _, f_nodal, w_pose, h_marker = env.step_batch(curr_state, mods, step_idx=step_idx)
    
    return next_state, f_nodal, w_pose, h_marker

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="checkpoints_shac", 
                        help="Directory to save checkpoints and visuals")
    parser.add_argument("--gpu", action="store_true", 
                        help="Enable GPU (Removes CPU forcing)")
    args = parser.parse_args()

    # 1. Handle Path Config
    abs_dir = os.path.abspath(args.dir)
    Config.CKPT_DIR = abs_dir
    Config.VIS_DIR = abs_dir
    
    print(f"--> OUTPUT DIRECTORY: {Config.CKPT_DIR}")

    # 2. Handle GPU Config
    if args.gpu:
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]
        print("--> MODE: GPU Enabled (JAX Default)")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("--> MODE: Force CPU (Use --gpu to enable GPU)")

    train()