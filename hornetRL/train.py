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
    SEED = 42

    # --- MASTER FREQUENCY SETTING ---
    BASE_FREQ = 115.0  # Change this to whatever you want (e.g. 150.0)
    TARGET_STATE = jnp.array([0.0, 0.0, 1.08, 0.3, 0.0, 0.0, 0.0, 0.0])

    # --- ARENA BOUNDARIES ---
    # The absolute limit before the robot is considered "crashed"
    # 0.30m = 30cm radius. (Total workspace width = 60cm)
    ARENA_W = 0.45  # Max displacement

    # --- OBSERVATION SCALING ---
    # Indices: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    # These map the raw physics to roughly [-1, 1] for the neural network
    OBS_SCALE = jnp.array([
        0.45,   # x: Arena boundary
        0.45,   # z: Arena boundary
        3.14,   # theta: Full rotation
        1.50,   # phi: Abdomen limit
        5.00,   # vx: Reasonable flight speed
        5.00,   # vz: Reasonable flight speed
        50.0,  # w_theta: Wing-beat vibration truth
        50.0   # w_phi: Wing-beat vibration truth
    ])
    
    # --- TIME SCALES ---
    DT = 3e-5               # Physics Timestep (High freq)
    SIM_SUBSTEPS = 20       # Physics steps per 1 Control Step (Brain update)
                            # Effective Control DT = 3e-5 * 20 = 0.6ms (1666 Hz)
                            
    HORIZON = 64            # Control Steps per Episode. 
                            # 64 * 0.6ms = 0.038 seconds (~4.4 wingbeats)
                            # This is plenty for learning hover stability.

    BATCH_SIZE = 4         # Parallel Environments
    LR_ACTOR = 5e-4         # Learning Rate
    MAX_GRAD_NORM = 1.0     # Gradient Clipping
    GAMMA = 0.99            # Discount Factor
    TOTAL_UPDATES = 10000   # Total Gradient Steps
    RESET_INTERVAL = 20     # Reset every N epochs to force takeoff practice.
    
    CKPT_DIR = "checkpoints_shac"
    VIS_DIR = "checkpoints_shac"
    AUX_LOSS_WEIGHT = 100.0   
    VIS_INTERVAL = 100      
    
    WARMUP_STEPS = 1        # Control steps (1 * 40 = 40 physics steps) to pin the fly

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state):
    # 1. Actor (Brain + Muscles)
    # [UPDATED]: Pass OBS_SCALE so the brain can normalize the target internally
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=Config.TARGET_STATE,
        obs_scale=Config.OBS_SCALE
    )
    
    # 2. Critic (Value Function)
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
    def __init__(self):
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=Config.BASE_FREQ  # <--- LINKED
        )
        
        # Target: Hover Upright
        self.target = Config.TARGET_STATE

    def reset(self, key, batch_size):
        """
        Generates fresh states with SYNCHRONIZED Fluid/Robot history.
        Uses Mixed Initialization (80% Nominal, 20% Chaos) to train recovery.
        Corrected to initialize Surrogate in Local Frame.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        
        # =========================================================
        # 1. INIT ROBOT STATE (Mixed Strategy)
        # =========================================================
        n_nominal = int(batch_size * 1.0)
        n_chaos = batch_size - n_nominal
        
        # --- A. Nominal Group (Upright Hover) ---
        k1_n, k2_n = jax.random.split(k1)
        
        # Position: Tight window (+/- 15cm)
        q_pos_nom = jax.random.uniform(k1_n, (n_nominal, 2), minval=-0.05, maxval=0.0)
        
        # Angle Setup:
        # 1. Pitch (Theta): Upright (pi/2)
        theta_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        theta_nom = theta_nom + 1.08
        
        # 2. Abdomen (Phi): [FIXED] Spawn near equilibrium (~0.3 rad)
        # Range: 0.1 to 0.5 (Relaxed, natural droop). 
        # Prevents starting with internal tension.
        phi_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        phi_nom = phi_nom + 0.3
        
        q_ang_nom = jnp.concatenate([theta_nom, phi_nom], axis=-1)
        
        # --- B. Chaos Group (Recovery Training) ---
        k1_c, k2_c = jax.random.split(k2)
        k_theta, k_phi = jax.random.split(k2_c)
        
        # Position: Wider window (+/- 30cm) - simulating being blown off course
        q_pos_chaos = jax.random.uniform(k1_c, (n_chaos, 2), minval=-0.30, maxval=0.30)
        
        # Pitch (Theta): FULL 360 degree randomization (-pi to pi)
        theta_chaos = jax.random.uniform(k_theta, (n_chaos, 1), minval=-jnp.pi, maxval=jnp.pi)
        
        # Abdomen (Phi): [FIXED] Spawn within VALID Asymmetric Limits (-0.2 to +1.4)
        # -0.2 (Hyperextension Limit) to 1.4 (Stinging Limit)
        # Prevents spawning inside the "Hard Wall" which would cause immediate explosion.
        phi_chaos = jax.random.uniform(k_phi, (n_chaos, 1), minval=-0.6, maxval=1.4)
        
        q_ang_chaos = jnp.concatenate([theta_chaos, phi_chaos], axis=-1)
        
        # --- C. Combine & Velocity ---
        q_pos = jnp.concatenate([q_pos_nom, q_pos_chaos], axis=0)
        q_ang = jnp.concatenate([q_ang_nom, q_ang_chaos], axis=0)
        
        # Velocity: Start at zero (or small noise) for simplicity
        v = jnp.zeros((batch_size, 4))
        
        # [Batch, 8] State Vector
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # =========================================================
        # 2. INIT OSCILLATOR (Random Phase)
        # =========================================================
        osc_state_single = OscillatorState.init(base_freq=Config.BASE_FREQ) # <--- LINKED
        def stack_batch(x): return jnp.stack([x] * batch_size)
        osc_state = jax.tree.map(stack_batch, osc_state_single)
        
        rand_phase = jax.random.uniform(k3, (batch_size,), minval=0.0, maxval=2*jnp.pi)
        osc_state = osc_state._replace(phase=rand_phase)

        # =========================================================
        # 3. CALCULATE WING POSE (Global -> Centered Local)
        # =========================================================
        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles = ret[0]
        k_rates  = ret[1]
        
        robot_state_p_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
        # This returns the wing tip position in GLOBAL coordinates
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_p_dummy, k_angles, k_rates)
        
        # --- CENTER THE POSE FOR SURROGATE INIT ---
        # The surrogate expects the wing to be at (0,0) in its inference frame.
        # We must subtract: Body Pos + Hinge Offset + Bias Offset
        
        def get_centered_pose(r_state, w_pose_glob, bias_val):
            # Unpack Robot State
            q = r_state[:4]
            theta = q[2]
            
            # A. Calculate Global Hinge Offset (Rotated by Body)
            h_x = self.phys.robot.hinge_offset_x
            h_z = self.phys.robot.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            # B. Calculate Global Bias Offset (Rotated by Stroke Plane)
            # The bias shifts the center of oscillation along the stroke plane
            total_st_ang = theta + self.phys.robot.stroke_plane_angle
            c_st, s_st = jnp.cos(total_st_ang), jnp.sin(total_st_ang)
            bias_glob_x = bias_val * c_st
            bias_glob_z = bias_val * s_st
            
            # C. Total Offset Vector (Body CoM -> Instantaneous Stroke Center)
            off_x = hinge_glob_x + bias_glob_x
            off_z = hinge_glob_z + bias_glob_z
            
            # D. Center the Pose
            # Local_Pos = Global_Wing - (Body_Pos + Total_Offset)
            p_x = w_pose_glob[0] - (q[0] + off_x)
            p_y = w_pose_glob[1] - (q[1] + off_z)
            
            # Return [x_centered, z_centered, angle_global]
            return jnp.array([p_x, p_y, w_pose_glob[2]])

        # Apply centering transformation
        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias)

        # =========================================================
        # 4. INIT FLUID STATE
        # =========================================================
        def init_fluid_fn(wp):
            return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
            
        # Pass the centered pose, not the global one
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state)

    def step_batch(self, full_state, action_mods, step_idx=100):
        robot_st, fluid_st, osc_st = full_state
        
        # We define a function for a SINGLE agent
        def single_agent_step(r, f, o, a):
            
            # --- SUB-STEPPING LOOP ---
            # This inner function runs 'Config.SIM_SUBSTEPS' times
            def sub_step_fn(carry, _):
                curr_r, curr_f, curr_o = carry
                
                # 1. Oscillator (Steps by DT)
                o_next = step_oscillator(curr_o, unpack_action(a), Config.DT)
                k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
                
                # Pack stroke_bias (o.bias) into action_data
                action_data = (k_angles, k_rates, tau_abd, bias)

                # 2. Physics (Steps by DT)
                (r_next_v, f_next), f_wing, f_nodal, wing_pose, hinge_marker = self.phys.step(
                    self.phys.fluid.params, (curr_r, curr_f), action_data, 0.0, Config.DT
                )
                
                # RAMP calculation (Based on the MACRO step_idx)
                ramp = jnp.clip(step_idx / Config.WARMUP_STEPS, 0.0, 1.0)
                
                # 1. Reset robot velocity logic
                # Applies if the macro step is < Config.WARMUP_STEPS
                v_reset = jnp.zeros(4) 
                r_next_v = jnp.where(step_idx < Config.WARMUP_STEPS, r_next_v.at[4:].set(v_reset), r_next_v)
                
                # 2. Apply linear ramp to nodal forces
                f_nodal_ramped = f_nodal * ramp
                
                # [SAFETY CLAMP] Prevent Physics Explosion
                # Define limits: [Vx, Vz, W_theta, W_phi]
                # Linear: 20 m/s (plenty of room for diving)
                # Angular: 200 rad/s (allows snap turns/saccades)
                v_limits = jnp.array([20.0, 20.0, 50.0, 50.0])
                
                # Apply vector clamp
                v_current = r_next_v[4:]
                v_clamped = jnp.clip(v_current, -v_limits, v_limits)
                
                r_next_v = r_next_v.at[4:].set(v_clamped)
                
                # 3. Torque Check (Aux Loss) calculation
                tau_actual = f_wing[2] * ramp
                f_actual = jnp.array([f_wing[0]*ramp, f_wing[1]*ramp, tau_actual, 0.0])
                
                # Return: (New State Carry), (Data to accumulate)
                return (r_next_v, f_next, o_next), (f_actual, f_nodal_ramped, wing_pose, hinge_marker)

            # --- EXECUTE SCAN ---
            init_carry = (r, f, o)
            
            # Loop for SIM_SUBSTEPS iterations
            (final_r, final_f, final_o), (stacked_forces, stacked_nodals, stacked_poses, stacked_hinges) = jax.lax.scan(
                sub_step_fn, init_carry, None, length=Config.SIM_SUBSTEPS
            )
            
            # --- POST-PROCESSING ---
            # 1. Average the actual forces (This gives the "Mean Aerodynamic Force" for the loss)
            avg_f_actual = jnp.mean(stacked_forces, axis=0)
            
            # 2. Take the LAST nodal force and pose (For visualization consistency)
            last_f_nodal = stacked_nodals[-1]
            last_wing_pose = stacked_poses[-1]
            last_hinge_marker = stacked_hinges[-1] # [FIX] Get last hinge pos
            
            # [FIX IS HERE]: Added last_hinge_marker to the return tuple
            return final_r, final_f, final_o, avg_f_actual, last_f_nodal, last_wing_pose, last_hinge_marker

        # --- MEMORY OPTIMIZATION: CHECKPOINTING ---
        # Wraps the single agent logic to prevent storing the 40 sub-steps in RAM.
        # This reduces memory usage by ~40x, fixing the OOM crash.
        single_agent_step_remat = jax.checkpoint(single_agent_step)

        # Vectorize the CHECKPOINTED function over the batch
        # Now the unpacking works because single_agent_step returns 7 items
        r_n, f_n, o_n, f_act, f_nodal_b, w_pose_b, h_marker_b = jax.vmap(single_agent_step_remat)(robot_st, fluid_st, osc_st, action_mods)
        
        # Re-group the state tuple so it matches the input structure for the next step
        return (r_n, f_n, o_n), f_act, f_nodal_b, w_pose_b, h_marker_b

    # --- Metrics-Aware Reward Function ---
    def get_reward_metrics(self, robot_state, u_forces):
        """Returns Reward AND breakdown of costs"""
        err = robot_state - self.target
        
        # --- SHORTEST PATH FOR THETHA ONLY ---
        # Abdomen (phi) is already bounded by physics [-0.6, 1.4]
        err_theta = jnp.mod(err[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi

        # 1. Calculate raw squared errors
        loss_pos = jnp.sum(err[:, :2]**2, axis=1)    
        
        # Index 2: Thorax (Theta)
        loss_ang_thorax = err_theta**2             # WRAPPED
        # Index 3: Abdomen (Phi)
        loss_ang_abdomen = err[:, 3]**2
        
        # Indices 4,5 are Linear Velocity (vx, vz)
        loss_lin_vel = jnp.sum(err[:, 4:6]**2, axis=1)
        # Indices 6,7 are Angular Velocity (w_theta, w_phi)
        loss_ang_vel = jnp.sum(err[:, 6:8]**2, axis=1)

        loss_eff = jnp.sum(u_forces**2, axis=1)        
        
        # 2. Weighted Cost (OPTIMIZED FOR AGILITY)
        # ---------------------------------------------------------
        # Pos:        1000.0 (Primary Goal)
        # Ang Thorax: 80.0   (WAS 500.0). Reduced drastically. 
        #             Why? To move, the fly MUST tilt. 
        #             If penalty is too high, it refuses to tilt to correct position errors.
        #             Now: Cost of Tilt < Cost of Position Error.
        # Ang Abdomen: 1.0   (Loose leash - allow pendulum swinging)
        # LinVel:     0.1    (WAS 0.01). Increased. 
        #             Why? Prevents "Drive-by" hovering. Forces it to actually PARK at the target.
        # AngVel:     0.01   (WAS 0.001). Slight bump to reduce jitter.
        # Eff:        0.1    (WAS 1.0). Reduced. 
        #             High effort penalty discourages "punchy" agile corrections.
        # ---------------------------------------------------------
        
        cost = (5000.0 * loss_pos + 
                80.0   * loss_ang_thorax + 
                1.0    * loss_ang_abdomen + 
                0.1    * loss_lin_vel + 
                0.00001   * loss_ang_vel + 
                0.05    * loss_eff)
        
        # 3. "Soft Fence" (Unchanged)
        dist_from_center = jnp.sqrt(loss_pos)
        out_of_bounds_cost = jnp.where(dist_from_center > 0.20, 100.0, 0.0)
        cost = cost + out_of_bounds_cost 

        # 4. Bonus (Unchanged)
        is_close = loss_pos < 0.002 # Within ~4.5cm
        bonus = is_close * 5.0
        is_close2 = loss_pos < 0.0004 # Within ~2cm
        bonus2 = is_close2 * 25.0

        # [ALIVE REWARD EXPLANATION]
        # This works because when the agent dies (hit_ground in step function),
        # the episode terminates. The agent stops receiving this +1.0.
        # Therefore, staying alive longer = More +1.0s.
        alive_reward = 1.0 
        
        raw_reward = alive_reward + bonus + bonus2 - cost
        
        # REWARD SCALING
        scaled_reward = raw_reward * 0.02 
        
        # 5. Return metrics dictionary for logging
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
    print(f"--> Generatng Visualization for Step {update_idx}...")
    # [UPDATED] Added 'hinge_marker' to data storage
    sim_data = {'states': [], 'wing_pose': [], 'nodal_forces': [], 'le_marker': [], 'hinge_marker': [], 't': []}
    
    rng = jax.random.PRNGKey(update_idx)
    state = env.reset(rng, 1) 
    
    # Configuration for sub-stepping
    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 1  # Adjust as needed for GIF length
    
    current_step_counter = 0
    
    for i in range(total_visual_frames):
        # --- SUB-STEPPING LOOP ---
        # Run physics multiple times per one visual frame
        for _ in range(steps_per_frame):
            r_st = state[0]
            
            # Check for NaNs (using the first agent in batch)
            r_cpu = np.array(r_st[0])
            if np.isnan(r_cpu).any():
                print(f"!!! Visualization stopped early due to NaN !!!")
                break

            # --- ADD WRAP HERE ---
            wrapped_th = jnp.mod(r_st[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs_v = r_st.at[:, 2].set(wrapped_th)
            
            scaled_obs = obs_v / Config.OBS_SCALE
            
            # Model sees scaled, but env.step_batch takes raw mods
            mods, _, _ = ac_model.apply(params, scaled_obs)
            
            # Use current_step_counter for correct ramp/warmup index
            # [UPDATED] Unpack the 5th return value (h_marker)
            state, _, f_nodal, w_pose, h_marker = env.step_batch(state, mods, step_idx=current_step_counter)
            current_step_counter += 1

        # Record data only for the visual frame
        sim_data['states'].append(np.array(state[0][0])) 
        sim_data['t'].append(current_step_counter * Config.DT)
        
        f_st = state[1]
        sim_data['le_marker'].append(np.array(f_st.marker_le[0]))
        sim_data['wing_pose'].append(np.array(w_pose[0]))
        sim_data['nodal_forces'].append(np.array(f_nodal[0]))
        # [UPDATED] Store hinge marker data
        sim_data['hinge_marker'].append(np.array(h_marker[0]))

    # Matplotlib Animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    patch_thorax = patches.Ellipse((0,0), linewidth=1.0, width=0.012, height=0.006, facecolor='#333333', edgecolor='black', zorder=10)
    patch_head = patches.Circle((0,0), linewidth=1.0, radius=0.0025, facecolor='#00FF00', edgecolor='black', zorder=10)
    patch_abd = patches.Ellipse((0,0), linewidth=1.0, width=0.018, height=0.008, facecolor='#1f77b4', edgecolor='black', alpha=0.8, zorder=9)
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)

    real_line, = ax.plot([], [], 'k-', linewidth=1.0, alpha=0.8, zorder=12)
    patch_le = patches.Circle((0,0), radius=0.001, color='red', zorder=15, label='Leading Edge')
    ax.add_patch(patch_le)

    # [UPDATED] Create the Hinge Marker Patch (Orange)
    patch_hinge = patches.Circle((0,0), radius=0.001, color='orange', zorder=15, label='Hinge')
    ax.add_patch(patch_hinge)

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
        # [UPDATED] Retrieve hinge marker data
        hinge_pos = sim_data['hinge_marker'][frame]
        t = sim_data['t'][frame]
        
        rx, rz = r_state[0], r_state[1]
        r_th, r_phi = r_state[2], r_state[3]
        
        # Follow Camera
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        
        # 1. Thorax
        patch_thorax.set_center((rx, rz))
        patch_thorax.set_angle(np.degrees(r_th))
        
        # 2. Head
        d1 = env.phys.robot.d1
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        
        # 3. Abdomen (Calculated AFTER d1/d2/thorax pose)
        joint_x = rx - d1 * np.cos(r_th)
        joint_z = rz - d1 * np.sin(r_th)
        
        d2 = env.phys.robot.d2
        abd_ang = r_th + r_phi
        patch_abd.set_center((joint_x - d2 * np.cos(abd_ang), joint_z - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))
        
        # 4. Wing
        wx, wz, wang = w_pose
        wing_len = env.phys.fluid.WING_LEN
        N_pts = env.phys.fluid.N_PTS
        x_local = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        wing_x = wx + x_local * c_w
        wing_z = wz + x_local * s_w
        real_line.set_data(wing_x, wing_z)
        
        patch_le.set_center((le_pos[0], le_pos[1]))
        
        # [UPDATED] Update Hinge Position
        patch_hinge.set_center((hinge_pos[0], hinge_pos[1]))
        
        pts = np.stack([wing_x, wing_z], axis=1)
        quiver.set_offsets(pts)
        quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
        
        time_text.set_text(f"T: {t:.4f}s | Y: {rz:.3f}")
        # [UPDATED] Return new patch
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
    # Change this part in your train() function:
    checkpoints = glob.glob(os.path.join(Config.CKPT_DIR, "*.pkl"))
    
    if checkpoints:
        # NEW: Sort by extracting the integer from the filename
        # This uses a lambda to find the number in 'shac_params_XXXX.pkl'
        checkpoints.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        last_ckpt = checkpoints[-1]
        print(f"--> Resuming from {last_ckpt}")
        
        with open(last_ckpt, "rb") as f:
            data = pickle.load(f)
            params = data['params']
            opt_state = data['opt_state']
            
            # Extract the actual step number to resume the loop correctly
            match = re.search(r"params_(\d+)", last_ckpt)
            if match: 
                start_step = int(match.group(1)) + 1

    def loss_fn(params, start_state, key):  # <--- UPDATED: Added 'key' argument
        
        # 1. Generate random keys for the entire rollout at once
        # We need one key per timestep for the scan loop
        scan_keys = jax.random.split(key, Config.HORIZON)
        
        # <--- UPDATED: Pass keys alongside step_idx
        scan_inputs = (jnp.arange(Config.HORIZON), scan_keys)

        def scan_fn(carry, xs): 
            curr_full = carry
            step_idx, step_key = xs  # Unpack key
            
            curr_robot = curr_full[0]
            # 1. WRAP THETA ONLY (Index 2)
            wrapped_theta = jnp.mod(curr_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            
            # 2. Reconstruct the state with the wrapped angle
            obs_robot = curr_robot.at[:, 2].set(wrapped_theta)

            # 3. Use the config to normalize the WRAPPED observation
            scaled_obs = obs_robot / Config.OBS_SCALE
            
            # --- NOISE INJECTION START (MOVED TO OBSERVATION) ---
            # Define noise scale (std dev). 
            # 0.05 is standard. Now applied to SENSORS, not MUSCLES.
            noise_sigma = 0.05
            
            # Generate noise for the OBSERVATION
            obs_noise = jax.random.normal(step_key, shape=scaled_obs.shape) * noise_sigma
            
            # Add noise to observation (Simulate blurry vision/sensors)
            noisy_obs = scaled_obs + obs_noise
            # --- NOISE INJECTION END ---

            # 4. Pass NOISY obs to the actor-critic model
            # The network will naturally filter this noise, outputting smooth 'mods'
            mods, u_brain, _ = ac_model.apply(params, noisy_obs)
            
            # 5. Use CLEAN 'mods' for the environment step
            # We no longer add noise here. The "Variance" comes from the 
            # network reacting to the noisy input.
            next_full, f_actual, _, _, _ = env.step_batch(curr_full, mods, step_idx=step_idx)
            
            # Get SCALED reward and metrics dict
            # We compare u_brain (what the brain commanded based on noisy info)
            # against f_actual (what physics actually did).
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain)
            
            force_err = jnp.mean((u_brain[:, :3] - f_actual[:, :3])**2)
            
            # Apply scaling to the auxiliary loss too, to match magnitude
            loss_t = -rew_scaled + (Config.AUX_LOSS_WEIGHT * force_err)
            
            # Pack all metrics for the scan
            step_metrics = (
                loss_t, rew_scaled, force_err, 
                met['rew'], met['pos'], 
                met['ang_th'], met['ang_ab'], 
                met['vel_lin'], met['vel_ang']
            )

            return next_full, step_metrics

        final_full, step_results = jax.lax.scan(
            scan_fn, start_state, scan_inputs # <--- UPDATED: inputs now include keys
        )
        
        # Unpack the scan results
        (losses, rewards_scaled, f_errs, real_rews, m_pos, 
         m_ang_th, m_ang_ab,   # <--- NEW VARIABLES
         m_vel_lin, m_vel_ang) = step_results

        # Warmup Mask
        warmup_mask = jnp.arange(Config.HORIZON) >= Config.WARMUP_STEPS
        losses = jnp.where(warmup_mask[:, None], losses, 0.0)

        # 1. Discounted Loss
        discounts = Config.GAMMA ** jnp.arange(Config.HORIZON)
        weighted_loss = jnp.dot(discounts, losses)
        
        # 2. Terminal Value for Actor
        final_robot = final_full[0]
        _, _, final_val_actor = ac_model.apply(params, final_robot)
        final_val_actor = jnp.squeeze(final_val_actor)
        
        # Actor Optimization Target (using scaled values)
        actor_term = jnp.mean(weighted_loss - (Config.GAMMA**Config.HORIZON * final_val_actor))

        # 3. Critic Target
        final_val_target = jax.lax.stop_gradient(final_val_actor)
        discounted_return = jnp.dot(discounts, rewards_scaled) + (Config.GAMMA**Config.HORIZON * final_val_target)
        
        _, _, start_val = ac_model.apply(params, start_state[0])
        start_val = jnp.squeeze(start_val)
        
        # HUBER LOSS
        critic_loss = optax.huber_loss(start_val, discounted_return, delta=1.0)
        critic_loss = jnp.mean(critic_loss)
        
        total_loss = actor_term + (0.5 * critic_loss) # Weight critic slightly less
        
        # Log Packet
        logs = {
            'rew': jnp.mean(real_rews),
            'ferr': jnp.mean(f_errs),
            'pos': jnp.mean(m_pos),
            # [FIX] Add the split keys here
            'ang_th': jnp.mean(m_ang_th),
            'ang_ab': jnp.mean(m_ang_ab),
            
            # New Keys needed for the print statement
            'vel_lin': jnp.mean(m_vel_lin),
            'vel_ang': jnp.mean(m_vel_ang),
            
            'act_loss': actor_term,
            'crit_loss': critic_loss
        }
        return total_loss, (logs, final_full)

    @jax.jit
    def update(params, opt_state, full_state, key): # <--- Add key argument
        
        # Split key: one for loss_fn (noise), one for next step
        key_loss, key_next = jax.random.split(key)
        
        # Pass key_loss to loss_fn
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, full_state, key_loss)
        
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Return key_next
        return new_params, new_opt, loss, logs, next_state, key_next

    print(f"=== Starting Training: Step {start_step} to {Config.TOTAL_UPDATES} ===")
    rng, key_reset = jax.random.split(rng)
    curr_state = env.reset(key_reset, Config.BATCH_SIZE)
    
    # --- [FIX 1] PRE-COMPILE CHECK NEEDS KEY ---
    print("--> Compiling JAX Update...")
    t0 = time.time()
    # Create a dummy key for compilation
    key_compile = jax.random.PRNGKey(0)
    # capture all outputs to prevent unpacking errors
    _ = update(params, opt_state, curr_state, key_compile) 
    print(f"--> Compilation Finished in {time.time() - t0:.2f}s")
    
    # --- Initialize Global Timer ---
    # Before the loop
    key_explore = jax.random.PRNGKey(999) # Create exploration key
    t_start = time.time()
    for i in range(start_step, Config.TOTAL_UPDATES):
        t0 = time.time() # Start of epoch

        if i == start_step:
            # Print what the Environment thinks the target is
            print(f"DEBUG: Env Target Theta: {env.target[2]:.4f}")
            # Print what the Robot thinks its angle is
            print(f"DEBUG: Spawn Theta: {curr_state[0][0, 2]:.4f}")

        # 1. Update
        params, opt_state, loss, logs, next_state, key_explore = update(params, opt_state, curr_state, key_explore)
    
        # 2. NaN Check
        if jnp.isnan(loss):
            print(f"!!! CRITICAL: NaN detected at step {i} !!! Reseting batch.")
            rng, k_res = jax.random.split(rng)
            curr_state = env.reset(k_res, Config.BATCH_SIZE)
            continue
        
        r_state = next_state[0]
        
        # --- RESET LOGIC START ---
        # A. Detect Crashes & NaNs
        is_nan = jnp.isnan(r_state).any(axis=1)
        is_crashed = (jnp.abs(r_state[:, 0]) > Config.ARENA_W) | (jnp.abs(r_state[:, 1]) > Config.ARENA_W)
        
        # B. Periodic Time-Limit (The "Infinite Horizon" Fix)
        # Every Config.RESET_INTERVAL epochs, force a full reset to prevent getting stuck in "survival loops".
        # This forces the agent to practice takeoff and stabilization repeatedly.
        # We add (i > 0) to prevent an immediate double-reset at step 0.
        is_timeout = (i > 0) & (i % Config.RESET_INTERVAL == 0)
        
        # Combine all reset conditions
        # If is_timeout is True, it broadcasts to reset EVERY environment in the batch.
        reset_mask = is_nan | is_crashed | is_timeout
        # --- RESET LOGIC END ---
    
        rng, k_res = jax.random.split(rng)
        fresh_state = env.reset(k_res, Config.BATCH_SIZE)
    
        curr_state = jax.tree.map(
            lambda x, y: jnp.where(jnp.reshape(reset_mask, (-1,) + (1,)*(x.ndim-1)), y, x),
            next_state, fresh_state
        )

        # --- Time Calculations ---
        dt_epoch = time.time() - t0              # Duration of this specific epoch
        total_elapsed = time.time() - t_start    # Total time since training started

        # 3. Telemetry & Printing
        sample_robot = next_state[0][0] 
        sample_osc = jax.tree.map(lambda x: x[0], next_state[2])

        # 1. WRAP Theta for the Telemetry Brain
        # (This ensures the brain isn't 'dizzy' when we check its actions)
        wrapped_theta = jnp.mod(sample_robot[2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_sample = sample_robot.at[2].set(wrapped_theta)

        # 2. SCALE the observation
        # (The brain now only understands values between -1 and 1)
        scaled_sample = obs_sample / Config.OBS_SCALE

        # 3. Pass SCALED observation to get mods
        sample_mods, _, _ = ac_model.apply(params, scaled_sample)

        angles, _, _, _ = get_wing_kinematics(sample_osc, unpack_action(sample_mods))
        str_angle, dev_angle, pit_angle = angles
    
        mean_x = jnp.mean(curr_state[0][:, 0])
        mean_z = jnp.mean(curr_state[0][:, 1])
    
        # --- NEW: Velocity Breakdown for Debugging ---
        # State Vector Indices: [x, z, theta, phi, vx, vz, w_theta, w_phi]
        # Linear Velocity (vx, vz) are at indices 4, 5
        lin_vels = next_state[0][:, 4:6]
        lin_mag = jnp.mean(jnp.sqrt(jnp.sum(lin_vels**2, axis=1)))

        ang_vels = next_state[0][:, 6:8]
        
        # A. Raw Energy ("The Hum"): Confirms physics is alive (~40,000)
        # (We keep this calculating both to ensure the simulation isn't exploding overall)
        raw_hum_energy = jnp.mean(jnp.sum(ang_vels**2, axis=1))

        # B. Thorax Velocity Only (Isolating Body Rotation)
        # Index 6 is w_theta (Thorax), Index 7 is w_phi (Abdomen)
        # We take the mean absolute value to see how fast it's spinning on average
        thorax_vel = next_state[0][:, 6]
        thorax_mag = jnp.mean(jnp.abs(thorax_vel))
        # ---------------------------------------------

        # Updated Printing with Hum/Thorax Separation
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
    # Convert to absolute path to avoid CWD confusion
    abs_dir = os.path.abspath(args.dir)
    Config.CKPT_DIR = abs_dir
    Config.VIS_DIR = abs_dir
    
    print(f"--> OUTPUT DIRECTORY: {Config.CKPT_DIR}")

    # 2. Handle GPU Config (Optional but helpful for Colab)
    if args.gpu:
        # User wants GPU: Remove the CPU force flag if it was set earlier
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]
        print("--> MODE: GPU Enabled (JAX Default)")
    else:
        # Default to CPU for stability unless flag is passed
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("--> MODE: Force CPU (Use --gpu to enable GPU)")

    train()