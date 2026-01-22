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

# --- USER MODULES ---
from .environment_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

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
    TARGET_STATE = jnp.array([0.0, 0.0, 1.08, 0.3, 0.0, 0.0, 0.0, 0.0])

    # --- Arena Boundaries ---
    # Absolute displacement limit (meters) before episode termination.
    # 0.45m corresponds to a 90cm total workspace width.
    ARENA_W = 0.45 

    # --- Observation Scaling ---
    # Normalization constants to map raw physics states to Neural Network friendly ranges [-1, 1].
    # Indices: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    OBS_SCALE = jnp.array([
        0.45,   # x: Arena boundary
        0.45,   # z: Arena boundary
        3.14,   # theta: Full rotation normalization
        1.50,   # phi: Abdomen joint limit
        5.00,   # vx: Max expected flight velocity
        5.00,   # vz: Max expected flight velocity
        50.0,   # w_theta: High-frequency body oscillation scale
        50.0    # w_phi: High-frequency abdomen oscillation scale
    ])
    
    # --- Time Scales ---
    DT = 3e-5               # Physics integration timestep (s)
    SIM_SUBSTEPS = 20       # Physics steps per Control Step
                            # Effective Control Freq: ~1666 Hz (0.6ms)
                            
    HORIZON = 64            # Trajectory horizon for Back-propagation Through Time (BPTT).
                            # Duration: ~0.038s (~4.4 wingbeats), sufficient for stability convergence.

    BATCH_SIZE = 5          # Number of parallel environments
    LR_ACTOR = 5e-4         # Learning Rate
    MAX_GRAD_NORM = 1.0     # Gradient Clipping threshold
    GAMMA = 0.99            # Discount Factor
    TOTAL_UPDATES = 10000   # Total Gradient Steps
    RESET_INTERVAL = 20     # Forced reset interval (epochs) to enforce takeoff robustness.
    
    CKPT_DIR = "checkpoints_shac"
    VIS_DIR = "checkpoints_shac"
    AUX_LOSS_WEIGHT = 100.0   
    VIS_INTERVAL = 200      
    
    WARMUP_STEPS = 1        # Control steps to pin the fly before releasing dynamics.

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state):
    """
    Defines the Actor-Critic architecture.
    
    Actor: Input Convex Neural Network (ICNN) wrapped in IDA-PBC logic.
           Injects OBS_SCALE to ensure target and current states share the same coordinate space.
    Critic: Standard MLP Value Function.
    """
    

    # 1. Actor (Brain + Muscles)
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=Config.TARGET_STATE,
        obs_scale=Config.OBS_SCALE
    )
    
    # 2. Critic (Value Function estimation)
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(robot_state)
    
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. ROBUST ENVIRONMENT
# ==============================================================================
class FlyEnv:
    """
    JAX-based environment handling the coupled Rigid Body and Fluid Surrogate dynamics.
    Implements SHAC-compatible step functions with automatic differentiation support.
    """
    def __init__(self):
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=Config.BASE_FREQ 
        )
        self.target = Config.TARGET_STATE

    def reset(self, key, batch_size):
        """
        Resets the environment state.
        
        Strategy:
            Uses a mixed initialization curriculum (80% Nominal Hover, 20% Chaotic Perturbation)
            to robustify the policy against disturbances.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        # =========================================================
        # 1. INIT ROBOT STATE (Curriculum Strategy)
        # =========================================================
        n_nominal = int(batch_size * 1.0)
        n_chaos = batch_size - n_nominal
        
        # --- A. Nominal Group (Stable Hover Conditions) ---
        k1_n, k2_n = jax.random.split(k1)
        
        # Position: Tight window (+/- 5cm)
        q_pos_nom = jax.random.uniform(k1_n, (n_nominal, 2), minval=-0.05, maxval=0.05)
        
        # Angle Setup:
        # 1. Pitch: Upright (~1.08 rad)
        theta_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        theta_nom = theta_nom + 1.08
        
        # 2. Abdomen: Equilibrium (~0.3 rad) to minimize initial internal stress
        phi_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        phi_nom = phi_nom + 0.3
        
        q_ang_nom = jnp.concatenate([theta_nom, phi_nom], axis=-1)
        
        # --- B. Chaos Group (Recovery Training) ---
        k1_c, k2_c = jax.random.split(k2)
        k_theta, k_phi = jax.random.split(k2_c)
        
        # Position: Wide window (+/- 30cm)
        q_pos_chaos = jax.random.uniform(k1_c, (n_chaos, 2), minval=-0.30, maxval=0.30)
        
        # Pitch: Full 360-degree randomization
        theta_chaos = jax.random.uniform(k_theta, (n_chaos, 1), minval=-jnp.pi, maxval=jnp.pi)
        
        # Abdomen: Randomized within physical limits (-0.6 to 1.4)
        phi_chaos = jax.random.uniform(k_phi, (n_chaos, 1), minval=-0.6, maxval=1.4)
        
        q_ang_chaos = jnp.concatenate([theta_chaos, phi_chaos], axis=-1)
        
        # --- C. Combine & Velocity ---
        q_pos = jnp.concatenate([q_pos_nom, q_pos_chaos], axis=0)
        q_ang = jnp.concatenate([q_ang_nom, q_ang_chaos], axis=0)
        
        # Velocity: Initialize at zero
        v = jnp.zeros((batch_size, 4))
        
        # State Vector: [Batch, 8]
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # =========================================================
        # 2. INIT OSCILLATOR (Random Phase)
        # =========================================================
        osc_state_single = OscillatorState.init(base_freq=Config.BASE_FREQ) 
        def stack_batch(x): return jnp.stack([x] * batch_size)
        osc_state = jax.tree.map(stack_batch, osc_state_single)
        
        rand_phase = jax.random.uniform(k3, (batch_size,), minval=0.0, maxval=2*jnp.pi)
        osc_state = osc_state._replace(phase=rand_phase)

        # =========================================================
        # 3. CALCULATE WING POSE (Global -> Centered Local)
        # =========================================================
        # Initialize surrogate in local frame relative to the stroke plane
        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles = ret[0]
        k_rates  = ret[1]
        
        robot_state_p_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_p_dummy, k_angles, k_rates)
        
        # --- Center Pose Transformation ---
        # The surrogate model inference assumes the wing is at (0,0).
        # We subtract (Body Pos + Hinge Offset + Bias Offset) to move to the local inference frame.
        
        def get_centered_pose(r_state, w_pose_glob, bias_val):
            q = r_state[:4]
            theta = q[2]
            
            # A. Global Hinge Offset (Rotated by Body)
            h_x = self.phys.robot.hinge_offset_x
            h_z = self.phys.robot.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            # B. Global Bias Offset (Rotated by Stroke Plane)
            total_st_ang = theta + self.phys.robot.stroke_plane_angle
            c_st, s_st = jnp.cos(total_st_ang), jnp.sin(total_st_ang)
            bias_glob_x = bias_val * c_st
            bias_glob_z = bias_val * s_st
            
            # C. Total Offset Vector
            off_x = hinge_glob_x + bias_glob_x
            off_z = hinge_glob_z + bias_glob_z
            
            # D. Centered Coordinates
            p_x = w_pose_glob[0] - (q[0] + off_x)
            p_y = w_pose_glob[1] - (q[1] + off_z)
            
            return jnp.array([p_x, p_y, w_pose_glob[2]])

        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias)

        # =========================================================
        # 4. INIT FLUID STATE
        # =========================================================
        def init_fluid_fn(wp):
            return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
            
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state)

    def step_batch(self, full_state, action_mods, step_idx=100):
        """
        Advances the simulation by one control step (Config.SIM_SUBSTEPS physics steps).
        Includes warmup ramping and velocity clamping for numerical stability.
        """
        robot_st, fluid_st, osc_st = full_state
        
        # Define single agent step function for vmap/scan
        def single_agent_step(r, f, o, a):
            
            # --- Sub-stepping Loop (Physics Integration) ---
            def sub_step_fn(carry, _):
                curr_r, curr_f, curr_o = carry
                
                # 1. Oscillator Update (Steps by DT)
                o_next = step_oscillator(curr_o, unpack_action(a), Config.DT)
                k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
                
                action_data = (k_angles, k_rates, tau_abd, bias)

                # 2. Physics Update (Rigid Body + Fluid)
                (r_next_v, f_next), f_wing, f_nodal, wing_pose, hinge_marker = self.phys.step(
                    self.phys.fluid.params, (curr_r, curr_f), action_data, 0.0, Config.DT
                )
                
                # --- Warmup Ramp & Stability ---
                # Applies a linear ramp to forces during the first WARMUP_STEPS
                ramp = jnp.clip(step_idx / Config.WARMUP_STEPS, 0.0, 1.0)
                
                # 1. Velocity Reset: Pin fly during warmup
                v_reset = jnp.zeros(4) 
                r_next_v = jnp.where(step_idx < Config.WARMUP_STEPS, r_next_v.at[4:].set(v_reset), r_next_v)
                
                # 2. Force Ramp: Scale nodal forces
                f_nodal_ramped = f_nodal * ramp
                
                # 3. Velocity Saturation: Safety clamp to prevent physics explosion
                # Limits: Linear (20 m/s), Angular (200 rad/s)
                v_limits = jnp.array([20.0, 20.0, 50.0, 50.0])
                v_current = r_next_v[4:]
                v_clamped = jnp.clip(v_current, -v_limits, v_limits)
                
                r_next_v = r_next_v.at[4:].set(v_clamped)
                
                # 4. Aux Loss Calculation Data
                tau_actual = f_wing[2] * ramp
                f_actual = jnp.array([f_wing[0]*ramp, f_wing[1]*ramp, tau_actual, 0.0])
                
                return (r_next_v, f_next, o_next), (f_actual, f_nodal_ramped, wing_pose, hinge_marker)

            # --- Execute Scan ---
            init_carry = (r, f, o)
            (final_r, final_f, final_o), (stacked_forces, stacked_nodals, stacked_poses, stacked_hinges) = jax.lax.scan(
                sub_step_fn, init_carry, None, length=Config.SIM_SUBSTEPS
            )
            
            # --- Post-Processing ---
            # 1. Mean Aerodynamic Force (for Aux Loss)
            avg_f_actual = jnp.mean(stacked_forces, axis=0)
            
            # 2. Visualization Data (Last Step)
            last_f_nodal = stacked_nodals[-1]
            last_wing_pose = stacked_poses[-1]
            last_hinge_marker = stacked_hinges[-1] 
            
            return final_r, final_f, final_o, avg_f_actual, last_f_nodal, last_wing_pose, last_hinge_marker

        # --- Checkpointing Optimization ---
        # Rematerializes the physics loop during backprop to save memory (~40x reduction).
        single_agent_step_remat = jax.checkpoint(single_agent_step)

        # Vectorize over batch
        r_n, f_n, o_n, f_act, f_nodal_b, w_pose_b, h_marker_b = jax.vmap(single_agent_step_remat)(robot_st, fluid_st, osc_st, action_mods)
        
        return (r_n, f_n, o_n), f_act, f_nodal_b, w_pose_b, h_marker_b

    def get_reward_metrics(self, robot_state, u_forces):
        """
        Calculates the scalar reward and detailed cost breakdown.
        Prioritizes position holding while allowing necessary body inclination for movement.
        """
        err = robot_state - self.target
        
        # Wrap Thorax Angle error to [-pi, pi]
        err_theta = jnp.mod(err[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi

        # 1. Squared Errors
        loss_pos = jnp.sum(err[:, :2]**2, axis=1)    
        loss_ang_thorax = err_theta**2             
        loss_ang_abdomen = err[:, 3]**2
        loss_lin_vel = jnp.sum(err[:, 4:6]**2, axis=1)
        loss_ang_vel = jnp.sum(err[:, 6:8]**2, axis=1)
        loss_eff = jnp.sum(u_forces**2, axis=1)        
        
        # 2. Weighted Cost Function (Agility Tuned)
        # Weights allow body tilt (low penalty) to facilitate lateral corrections (high penalty).
        cost = (100000.0 * loss_pos + 
                80.0    * loss_ang_thorax + 
                1.0     * loss_ang_abdomen + 
                0.1     * loss_lin_vel + 
                0.00001 * loss_ang_vel + 
                0.05    * loss_eff)
        
        # 3. Soft Fence Constraint
        dist_from_center = jnp.sqrt(loss_pos)
        out_of_bounds_cost = jnp.where(dist_from_center > 0.20, 100.0, 0.0)
        cost = cost + out_of_bounds_cost 

        # 4. Proximity Bonuses
        is_close = loss_pos < 0.002 # ~4.5cm
        bonus = is_close * 5.0
        is_close2 = loss_pos < 0.0004 # ~2cm
        bonus2 = is_close2 * 25.0

        # 5. Survival Bonus
        # Constant reward accumulated every timestep the agent remains valid.
        alive_reward = 1.0 
        
        raw_reward = alive_reward + bonus + bonus2 - cost
        scaled_reward = raw_reward * 0.02 
        
        metrics = {
            'rew': raw_reward,
            'pos': loss_pos,
            'ang_th': loss_ang_thorax, 
            'ang_ab': loss_ang_abdomen,
            'vel_lin': loss_lin_vel,  
            'vel_ang': loss_ang_vel,  
            'ferr': loss_eff,
            'ang': loss_ang_thorax + loss_ang_abdomen
        }
        return scaled_reward, metrics

# ==============================================================================
# 4. VISUALIZATION ENGINE
# ==============================================================================
def run_visualization(env, params, update_idx):
    """
    Generates a GIF visualization of the flight trajectory and aerodynamic forces.
    """
    print(f"--> Generatng Visualization for Step {update_idx}...")
    sim_data = {'states': [], 'wing_pose': [], 'nodal_forces': [], 'le_marker': [], 'hinge_marker': [], 't': []}
    
    rng = jax.random.PRNGKey(update_idx)
    state = env.reset(rng, 1) 
    
    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 1 
    
    current_step_counter = 0
    
    # JIT-COMPILED STEP FUNCTION
    # Compile Brain + Physics into one kernel to prevent OOM and speed up rendering.
    @jax.jit
    def vis_step(curr_state, curr_params, step_idx):
        r_st = curr_state[0]
        
        # 1. Prepare Observation
        wrapped_th = jnp.mod(r_st[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_v = r_st.at[:, 2].set(wrapped_th)
        scaled_obs = obs_v / Config.OBS_SCALE
        
        # 2. Policy Inference
        mods, _, _ = ac_model.apply(curr_params, scaled_obs)
        
        # 3. Environment Step
        next_state, _, f_nodal, w_pose, h_marker = env.step_batch(curr_state, mods, step_idx=step_idx)
        
        return next_state, f_nodal, w_pose, h_marker

    # Warmup JIT (Critical to prevent lag/crash on first frame)
    print("--> Compiling Visualization JIT...")
    _ = vis_step(state, params, 0)
    print("--> Compilation Complete!")

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
            state, f_nodal, w_pose, h_marker = vis_step(state, params, current_step_counter)
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
        
        # 1. Update Thorax
        patch_thorax.set_center((rx, rz))
        patch_thorax.set_angle(np.degrees(r_th))
        
        # 2. Update Head
        d1 = env.phys.robot.d1
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        
        # 3. Update Abdomen (Kinematic Chain)
        joint_x = rx - d1 * np.cos(r_th)
        joint_z = rz - d1 * np.sin(r_th)
        
        d2 = env.phys.robot.d2
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
# 5. TRAINING LOOP
# ==============================================================================
def train():
    """
    Main training loop implementing Short Horizon Actor-Critic (SHAC).
    Performs trajectory rollouts, computes gradients via BPTT, and updates policies.
    """
    

    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    
    env = FlyEnv()
    rng = jax.random.PRNGKey(Config.SEED)
    
    dummy_input = jnp.zeros((1, 8))
    params = ac_model.init(rng, dummy_input)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
        optax.adam(Config.LR_ACTOR)
    )
    opt_state = optimizer.init(params)
    
    start_step = 0
    # --- Checkpoint Resuming Logic ---
    checkpoints = glob.glob(os.path.join(Config.CKPT_DIR, "*.pkl"))
    
    if checkpoints:
        # Sort by step number extracted from filename
        checkpoints.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        
        last_ckpt = checkpoints[-1]
        print(f"--> Resuming from {last_ckpt}")
        
        with open(last_ckpt, "rb") as f:
            data = pickle.load(f)
            params = data['params']
            opt_state = data['opt_state']
            
            match = re.search(r"params_(\d+)", last_ckpt)
            if match: 
                start_step = int(match.group(1)) + 1

    def loss_fn(params, start_state, key):
        """
        Computes the total loss over the trajectory horizon.
        Includes policy gradient, value function loss, and auxiliary force matching loss.
        """
        # Generate temporal keys for noise injection
        scan_keys = jax.random.split(key, Config.HORIZON)
        scan_inputs = (jnp.arange(Config.HORIZON), scan_keys)

        def scan_fn(carry, xs): 
            curr_full = carry
            step_idx, step_key = xs 
            
            curr_robot = curr_full[0]
            # 1. Wrap Theta (Index 2) to [-pi, pi]
            wrapped_theta = jnp.mod(curr_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs_robot = curr_robot.at[:, 2].set(wrapped_theta)

            # 2. Normalize Observation
            scaled_obs = obs_robot / Config.OBS_SCALE
            
            # --- Noise Injection (Sensor Model) ---
            # Simulates sensor noise to promote robust control policies.
            noise_sigma = 0.05
            obs_noise = jax.random.normal(step_key, shape=scaled_obs.shape) * noise_sigma
            noisy_obs = scaled_obs + obs_noise
            
            # 3. Policy Inference (Noisy Input -> Smooth Action)
            mods, u_brain, _ = ac_model.apply(params, noisy_obs)
            
            # 4. Environment Step (Physics uses raw actions)
            next_full, f_actual, _, _, _ = env.step_batch(curr_full, mods, step_idx=step_idx)
            
            # 5. Reward Calculation
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain)
            
            # Auxiliary Loss: Match ICNN forces to Physics forces
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
        warmup_mask = jnp.arange(Config.HORIZON) >= Config.WARMUP_STEPS
        losses = jnp.where(warmup_mask[:, None], losses, 0.0)

        # 1. Discounted Loss (Actor)
        discounts = Config.GAMMA ** jnp.arange(Config.HORIZON)
        weighted_loss = jnp.dot(discounts, losses)
        
        # 2. Terminal Value Bootstrap
        final_robot = final_full[0]
        _, _, final_val_actor = ac_model.apply(params, final_robot)
        final_val_actor = jnp.squeeze(final_val_actor)
        
        actor_term = jnp.mean(weighted_loss - (Config.GAMMA**Config.HORIZON * final_val_actor))

        # 3. Critic Loss (Huber)
        final_val_target = jax.lax.stop_gradient(final_val_actor)
        discounted_return = jnp.dot(discounts, rewards_scaled) + (Config.GAMMA**Config.HORIZON * final_val_target)
        
        _, _, start_val = ac_model.apply(params, start_state[0])
        start_val = jnp.squeeze(start_val)
        
        critic_loss = optax.huber_loss(start_val, discounted_return, delta=1.0)
        critic_loss = jnp.mean(critic_loss)
        
        total_loss = actor_term + (0.5 * critic_loss)
        
        logs = {
            'rew': jnp.mean(real_rews),
            'ferr': jnp.mean(f_errs),
            'pos': jnp.mean(m_pos),
            'ang_th': jnp.mean(m_ang_th),
            'ang_ab': jnp.mean(m_ang_ab),
            'vel_lin': jnp.mean(m_vel_lin),
            'vel_ang': jnp.mean(m_vel_ang),
            'act_loss': actor_term,
            'crit_loss': critic_loss
        }
        return total_loss, (logs, final_full)

    @jax.jit
    def update(params, opt_state, full_state, key): 
        # Split key: one for loss noise, one for next step exploration
        key_loss, key_next = jax.random.split(key)
        
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, full_state, key_loss)
        
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt, loss, logs, next_state, key_next

    print(f"=== Starting Training: Step {start_step} to {Config.TOTAL_UPDATES} ===")
    rng, key_reset = jax.random.split(rng)
    curr_state = env.reset(key_reset, Config.BATCH_SIZE)
    
    # --- JIT Compilation ---
    print("--> Compiling JAX Update...")
    t0 = time.time()
    key_compile = jax.random.PRNGKey(0)
    _ = update(params, opt_state, curr_state, key_compile) 
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
        params, opt_state, loss, logs, next_state, key_explore = update(params, opt_state, curr_state, key_explore)
    
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
        
        # B. Periodic Timeout (Infinite Horizon Guard)
        # Forces the agent to practice takeoff and stabilization repeatedly.
        is_timeout = (i > 0) & (i % Config.RESET_INTERVAL == 0)
        
        reset_mask = is_nan | is_crashed | is_timeout
    
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
        scaled_sample = obs_sample / Config.OBS_SCALE
        sample_mods, _, _ = ac_model.apply(params, scaled_sample)

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

        print(f"Step {i:04d} | Epoch: {dt_epoch:.2f}s | Total: {total_elapsed/60:.1f}min | "
              f"Loss: {loss:.1e} (A:{logs['act_loss']:.1e} C:{logs['crit_loss']:.1e}) | "
              f"Rew: {logs['rew']:.1f}\n"
              f"    -> Errs[Pos:{logs['pos']:.2f} Th:{logs['ang_th']:.2f} Ab:{logs['ang_ab']:.2f} "
              f"LVel:{logs['vel_lin']:.2f} AVel:{logs['vel_ang']:.2f} Frc:{logs['ferr']:.1f}] | "
              f"MeanPos: [{mean_x:+.2f}, {mean_z:+.2f}]\n"
              f"    -> Phys: [Hum:{raw_hum_energy:.0f} | ThVel:{thorax_mag:.2f} rad/s]")
    
        if i % Config.VIS_INTERVAL == 0:
            ckpt_path = os.path.join(Config.CKPT_DIR, f"shac_params_{i}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({'params': params, 'opt_state': opt_state}, f)
            print(f"--> Saved Checkpoint: {ckpt_path}")
            run_visualization(env, params, i)

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