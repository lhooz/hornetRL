import os
import time

# Force CPU execution to prevent OOM errors during high-resolution plotting
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import haiku as hk
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# --- USER MODULES ---
from .fluid_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem, PhysParams
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """
    Configuration for the Inference / Evaluation pipeline.
    
    Parameters must match the training configuration to ensure the learned policy 
    behaves correctly. Includes additional settings for visualization and 
    robustness testing (perturbations).
    """
    # --- Model Constants (Must match Training) ---
    BASE_FREQ = 115.0  
    
    # Target State: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    TARGET_STATE = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    
    # Normalization Scale: Maps physics units to Neural Network input range
    OBS_SCALE = jnp.array([
        0.45,   # x
        0.45,   # z
        3.14,   # theta
        1.50,   # phi
        5.00,   # vx
        5.00,   # vz
        50.0,   # w_theta
        50.0    # w_phi
    ])

    # --- Simulation Settings ---
    DT = 3e-5             
    SIM_SUBSTEPS = 20    # Physics integration steps per single Control step
    
    # --- Inference Settings ---
    DURATION = 1.0       # Total simulation time (seconds)
    FPS = 60             # Output GIF frame rate
    DPI = 150            # Output resolution
    
    # Visualization Downsampling:
    # Skips frames to keep memory usage manageable during long simulations.
    # 50 = Render every 50th physics block.
    VIZ_STEP_SKIP = 40
    
    # --- Robustness Testing (Wind Gust) ---
    PERTURBATION = True  
    PERTURB_TIME = 0.02  # Time of impact (s)
    
    # Force Vector: [X, Z] Newtons (Simulates a lateral wind gust)
    PERTURB_FORCE = jnp.array([0.8, -1.4]) 
    # Torque: Positive = Pitch Up (Simulates aerodynamic instability)
    PERTURB_TORQUE = -0.002

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state):
    """
    Reconstructs the Actor-Critic architecture for inference.
    
    Only the Actor (Policy) is used during inference, but the full structure 
    is required to correctly load weights from the checkpoint.
    """
    # 1. Actor (Brain + Muscles)
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=Config.TARGET_STATE,
        obs_scale=Config.OBS_SCALE
    )
    
    # 2. Critic (Exact match to training to preserve parameter structure)
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(robot_state)
    
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. INFERENCE ENVIRONMENT
# ==============================================================================
class InferenceFlyEnv:
    """
    A specialized environment for evaluation.
    
    Differences from Training Env:
    1. Deterministic initialization (starts exactly at target).
    2. Supports external force/torque injection for disturbance rejection testing.
    3. Returns detailed visualization frames including force markers.
    """
    def __init__(self):
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=Config.BASE_FREQ
        )

    def reset(self, key):
        """Initializes the fly in a stable hover state."""
        # 1. Init Robot State (Start exactly at Target)
        batch_size = 1
        
        q_pos = jnp.array([[0.0, 0.0]]) 
        q_ang = jnp.array([[1.08, 0.3]])
        v = jnp.zeros((batch_size, 4))
        
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # 2. Init Oscillator
        osc_state = OscillatorState.init(base_freq=Config.BASE_FREQ)
        osc_state = jax.tree.map(lambda x: jnp.stack([x]*batch_size), osc_state)

       # 3. Define NOMINAL Physics Parameters (No Randomization for Inference)
        # We use scale=1.0 and offset=0.0 to test the "Ideal" Hornet.
        # Can modify this later to stress-test the policy against heavy/light hornets
        
        phys_params = PhysParams(
            thorax_mass_scale=jnp.array([1.0]),
            abd_mass_scale=jnp.array([1.0]),
            thorax_offset_x=jnp.array([0.0]),
            abd_offset_x=jnp.array([0.0]),
            hinge_x_noise=jnp.array([0.0]),
            hinge_z_noise=jnp.array([0.0]),
            stroke_ang_noise=jnp.array([0.0]),
            k_hinge_scale=jnp.array([1.0]),
            b_hinge_scale=jnp.array([1.0]),
            phi_equil_offset=jnp.array([0.0])
        )

        # 4. Calculate Centered Wing Pose for Fluid Surrogate
        self.active_props = jax.tree.map(lambda x: x[0], jax.vmap(self.phys.robot.compute_props)(phys_params))

        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles, k_rates = ret[0], ret[1]
        
        robot_state_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
        # in_axes (0, 0, 0, None) means map the first 3 args, but treat the 4th (props) as a single value
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics, in_axes=(0, 0, 0, None))(robot_state_dummy, k_angles, k_rates, self.active_props)
        
        # --- Center Pose Logic ---
        def get_centered_pose(r_state, w_pose_glob, bias_val, props):
            q, theta = r_state[:4], r_state[2]
            h_x, h_z = props.hinge_offset_x, props.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            total_st_ang = theta + props.stroke_plane_angle
            c_st, s_st = jnp.cos(total_st_ang), jnp.sin(total_st_ang)
            bias_glob_x, bias_glob_z = bias_val * c_st, bias_val * s_st
            
            off_x = hinge_glob_x + bias_glob_x
            off_z = hinge_glob_z + bias_glob_z
            
            p_x = w_pose_glob[0] - (q[0] + off_x)
            p_y = w_pose_glob[1] - (q[1] + off_z)
            return jnp.array([p_x, p_y, w_pose_glob[2]])

        # Again, tell vmap the last argument (props) is not batched
        wing_pose_centered = jax.vmap(get_centered_pose, in_axes=(0, 0, 0, None))(robot_state_v, wing_pose_global, osc_state.bias, self.active_props)
        
        def init_fluid_fn(wp): return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state, phys_params)

    def step(self, full_state, action_mods, external_force=jnp.zeros(2), external_torque=0.0):
        """
        Advances the simulation, applying actions and optional external disturbances.
        """
        robot_st, fluid_st, osc_st, phys_p = full_state
        
        # Unpack Batch (Simulate single agent)
        r = robot_st[0]
        f = jax.tree.map(lambda x: x[0], fluid_st) 
        o = jax.tree.map(lambda x: x[0], osc_st)
        a = action_mods[0]
        p_single = jax.tree.map(lambda x: x[0], phys_p)

        def sub_step_fn(carry, _):
            curr_r, curr_f, curr_o = carry
            
            # 1. Update Oscillator
            o_next = step_oscillator(curr_o, unpack_action(a), Config.DT)
            k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
            action_data = (k_angles, k_rates, tau_abd, bias)

            # 2. Update Physics
            (r_next_v, f_next), _, f_nodal, wing_pose, hinge_marker = self.phys.step(
                self.phys.fluid.params, 
                (curr_r, curr_f), 
                action_data, 
                self.active_props, # <--- Use the stored computed props
                0.0, 
                Config.DT
            )
            
            # --- Apply Perturbations (Wind Gust) ---
            # F = ma  ->  a = F/m
            total_mass = (self.phys.robot.m_thorax * p_single.thorax_mass_scale + 
                          self.phys.robot.m_abdomen * p_single.abd_mass_scale)
            accel_lin = external_force / total_mass
            r_next_v = r_next_v.at[4:6].add(accel_lin * Config.DT)
            
            # Tau = Ia -> alpha = Tau/I
            inertia = self.phys.robot.I_thorax * p_single.thorax_mass_scale
            accel_ang = external_torque / inertia
            r_next_v = r_next_v.at[6].add(accel_ang * Config.DT)
            
            # Pack visualization frame
            viz_frame = (r_next_v, wing_pose, f_nodal, f_next.marker_le, hinge_marker)

            return (r_next_v, f_next, o_next), viz_frame

        init_carry = (r, f, o)
        
        # Scan returns history of all substeps for high-res visualization
        (final_r, final_f, final_o), stacked_viz_frames = jax.lax.scan(
            sub_step_fn, init_carry, None, length=Config.SIM_SUBSTEPS
        )

        # Repack Batch
        r_b = jnp.expand_dims(final_r, 0)
        o_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_o)
        f_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_f)
        
        return (r_b, f_b, o_b, phys_p), stacked_viz_frames

# ==============================================================================
# 4. MAIN SIMULATION LOOP
# ==============================================================================
def run_simulation(params):
    """
    Executes the main inference loop.
    
    Calculates control actions, steps the environment, and collects downsampled
    data for visualization.
    """
    print("--> Initializing Inference Environment...")
    env = InferenceFlyEnv()
    rng = jax.random.PRNGKey(0)
    
    state = env.reset(rng)
    
    total_control_steps = int(Config.DURATION / (Config.DT * Config.SIM_SUBSTEPS))
    
    vis_data = {'r': [], 'w': [], 'f': [], 't': [], 'p_force': [], 'p_torque': [], 'le': [], 'hinge': []}
    
    print(f"--> Simulating {Config.DURATION}s ({total_control_steps} control steps)...")

    # Compile the ENTIRE step (Brain + Physics) into a single optimized function.
    # This prevents JAX from re-compiling or leaking memory traces in the loop.
    @jax.jit
    def single_inference_step(curr_state, curr_params, ext_f, ext_t):
        r_state = curr_state[0]
        
        # 1. Prepare Observation
        wrapped_theta = jnp.mod(r_state[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_input = r_state.at[:, 2].set(wrapped_theta)
        scaled_input = obs_input / Config.OBS_SCALE # Explicitly divide the whole vector
        
        # 2. Run Policy
        mods, _, _ = ac_model.apply(curr_params, scaled_input)
        
        # 3. Run Environment
        next_state, frames = env.step(curr_state, mods, external_force=ext_f, external_torque=ext_t)
        return next_state, frames
    
    t_sim = 0.0
    
    # Warmup compilation (Critical to prevent lag on first frame)
    print("--> Compiling JAX graph (this takes a few seconds)...")
    _ = single_inference_step(state, params, jnp.zeros(2), 0.0)
    print("--> Compilation Complete!")

    for i in range(total_control_steps):
        # --- 1. Determine Perturbation ---
        ext_f = jnp.zeros(2)
        ext_t = 0.0
        
        if Config.PERTURBATION:
            if Config.PERTURB_TIME <= t_sim <= Config.PERTURB_TIME + 0.002:
                ext_f = Config.PERTURB_FORCE
                ext_t = Config.PERTURB_TORQUE
        
        # --- 2. Run JIT-compiled step ---
        state, stacked_frames = single_inference_step(state, params, ext_f, ext_t)
        
        s_r, s_w, s_f, s_le, s_hinge = stacked_frames
        
        # --- 3. Global Skip Logic (Visualization Downsampling) ---
        # Determines which frames to keep based on the global simulation time.
        # This prevents the list from growing too large (OOM protection).
        
        global_start_step = i * Config.SIM_SUBSTEPS
        batch_global_steps = np.arange(global_start_step, global_start_step + Config.SIM_SUBSTEPS)
        
        # Identify indices in this batch that match the skip frequency
        indices_to_keep = np.where(batch_global_steps % Config.VIZ_STEP_SKIP == 0)[0]
        
        if len(indices_to_keep) > 0:
            vis_data['r'].extend([np.array(s_r[j]) for j in indices_to_keep])
            vis_data['w'].extend([np.array(s_w[j]) for j in indices_to_keep])
            vis_data['f'].extend([np.array(s_f[j]) for j in indices_to_keep])
            vis_data['le'].extend([np.array(s_le[j]) for j in indices_to_keep])
            vis_data['hinge'].extend([np.array(s_hinge[j]) for j in indices_to_keep])
            
            step_times = [t_sim + (j+1)*Config.DT for j in indices_to_keep]
            vis_data['t'].extend(step_times)
            
            # Log Perturbation Flags
            has_force = np.linalg.norm(ext_f) > 0
            has_torque = abs(ext_t) > 0
            vis_data['p_force'].extend([has_force] * len(indices_to_keep))
            vis_data['p_torque'].extend([has_torque] * len(indices_to_keep))
        
        t_sim += (Config.DT * Config.SIM_SUBSTEPS)
        
        if i % 100 == 0:
            print(f"    Progress: {i}/{total_control_steps} blocks | T={t_sim:.3f}s")

    return vis_data, env

# ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================
def generate_gif(data, env):
    """
    Renders the collected simulation data into a high-quality GIF.
    Includes trajectory traces, wing motion blur, and perturbation indicators.
    """
    

    print(f"--> Rendering High-Quality GIF ({len(data['r'])} frames collected)...")
    
    r_states = data['r']
    w_poses = data['w']
    times = data['t']
    flag_force = data['p_force']
    flag_torque = data['p_torque']
    le_markers = data['le']
    hinge_markers = data['hinge']
    
    # --- PRE-CALCULATION: Body-Relative Wing History ---
    # Convert global wing coordinates to local body coordinates.
    wing_rel_history = []
    
    for i in range(len(r_states)):
        r, w = r_states[i], w_poses[i]
        
        # 1. Vector from Body Center to Wing Center (Global)
        dx_global = w[0] - r[0]
        dz_global = w[1] - r[1]
        
        # 2. Rotate into Body Frame (Remove Body Rotation)
        c, s = np.cos(-r[2]), np.sin(-r[2])
        x_local = dx_global * c - dz_global * s
        z_local = dx_global * s + dz_global * c
        
        wing_rel_history.append([x_local, z_local])
        
    wing_rel_history = np.array(wing_rel_history)
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 8), dpi=Config.DPI)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f5')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    
    # --- Graphics Objects ---
    traj_line, = ax.plot([], [], color='#555555', linestyle='--', linewidth=1.0, alpha=0.5)

    # Wing Center Trajectory (Purple Trace)
    wing_traj_line, = ax.plot([], [], color='purple', linestyle='-', linewidth=0.8, alpha=0.0, label='Wing Path')
    
    patch_thorax = patches.Ellipse((0,0), width=0.012, height=0.006, facecolor='#2c3e50', edgecolor='k', zorder=10)
    patch_head = patches.Circle((0,0), radius=0.0025, facecolor='#e74c3c', edgecolor='k', zorder=10)
    patch_abd = patches.Ellipse((0,0), width=0.018, height=0.008, facecolor='#f39c12', edgecolor='k', alpha=0.9, zorder=9)
    
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)
    
    # Wing (Motion Blur effect using multiple alpha lines)
    wing_lines = []
    alphas = [0.05, 0.1, 0.2, 1.0] 
    for a in alphas:
        wl, = ax.plot([], [], 'k-', linewidth=1.5, alpha=a, zorder=11)
        wing_lines.append(wl)

    patch_le = patches.Circle((0,0), radius=0.0015, color='red', zorder=15)
    ax.add_patch(patch_le)

    patch_hinge = patches.Circle((0,0), radius=0.0015, color='orange', zorder=15)
    ax.add_patch(patch_hinge)
        
    # --- Perturbation Indicators ---
    arrow_force = patches.FancyArrow(0, 0, 0, 0, width=0.005, color='red', zorder=20, alpha=0.0)
    ax.add_patch(arrow_force)
    
    text_torque = ax.text(0, 0, '', fontsize=30, color='orange', ha='center', va='center', fontweight='bold', zorder=20, alpha=0.0)
    
    txt_time = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold', family='monospace')
    txt_info = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=10, family='monospace', color='#333333')
    
    window_size = 0.25 
    
    def get_wing_coords(r_pos, w_pose):
        wx, wz, wang = w_pose
        wing_len = env.phys.fluid.WING_LEN
        N_pts = 20
        x_local = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        wing_x = wx + x_local * c_w
        wing_z = wz + x_local * s_w
        return wing_x, wing_z

    def update(frame):
        # Use frame index directly (data is already decimated via Skip Logic)
        idx = frame 
        
        if idx >= len(r_states): return
        
        curr_r = r_states[idx]
        rx, rz = curr_r[0], curr_r[1]
        r_th, r_phi = curr_r[2], curr_r[3]
        
        # --- Update Camera ---
        ax.set_xlim(0.0 - window_size, 0.0 + window_size)
        ax.set_ylim(0.0 - window_size, 0.0 + window_size)
        
        # --- Update Body ---
        patch_thorax.set_center((rx, rz))
        patch_thorax.set_angle(np.degrees(r_th))
        
        d1 = env.phys.robot.d1
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        
        d2 = env.phys.robot.d2
        abd_ang = r_th + r_phi
        joint_x = rx - d1 * np.cos(r_th)
        joint_z = rz - d1 * np.sin(r_th)
        patch_abd.set_center((joint_x - d2 * np.cos(abd_ang), joint_z - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))
        
        # --- Update Wings with Trail/Blur ---
        for k in range(4):
            # Scale the offset because data density changed.
            # Previously: offset=3 frames. Now: 1 frame = 20*DT.
            offset = 3-k 
            hist_idx = max(0, idx - offset)
            w_x, w_z = get_wing_coords(r_states[hist_idx], w_poses[hist_idx])
            wing_lines[k].set_data(w_x, w_z)

        # --- Update Markers ---
        le_pos = le_markers[idx]
        patch_le.set_center((le_pos[0], le_pos[1]))

        hinge_pos = hinge_markers[idx]
        patch_hinge.set_center((hinge_pos[0], hinge_pos[1]))
            
        # --- Update Trajectory (Body) ---
        hist_len = 50 
        start_t = max(0, idx - hist_len)
        hist_x = [r[0] for r in r_states[start_t:idx]] 
        hist_z = [r[1] for r in r_states[start_t:idx]]
        traj_line.set_data(hist_x, hist_z)

        # --- Update Trajectory (Wing Center) ---
        wing_hist_len = 22
        start_w = max(0, idx - wing_hist_len)

        # 1. Get the chunk of relative history
        rel_chunk = wing_rel_history[start_w:idx]
        
        if len(rel_chunk) > 0:
            # 2. Transform it to the CURRENT body pose
            # Rot(theta) * local + Pos
            curr_c, curr_s = np.cos(r_th), np.sin(r_th)
            
            # Rotation
            traj_x = rel_chunk[:, 0] * curr_c - rel_chunk[:, 1] * curr_s
            traj_z = rel_chunk[:, 0] * curr_s + rel_chunk[:, 1] * curr_c
            
            # Translation
            traj_x += rx
            traj_z += rz
            
            wing_traj_line.set_data(traj_x, traj_z)
        else:
            wing_traj_line.set_data([], [])
        # ---------------------------------------------------
        
        # --- Visualize Kick (Perturbation) ---
        is_force = flag_force[idx]
        is_torque = flag_torque[idx]
        
        if is_force:
            arrow_force.set_alpha(1.0)

            # 1. Get physics force direction
            fx, fz = Config.PERTURB_FORCE
    
            # 2. Normalize it to a fixed arrow length (e.g., 0.03m)
            mag = np.sqrt(fx**2 + fz**2) + 1e-6
            d_x = (fx / mag) * 0.03
            d_z = (fz / mag) * 0.03
    
            # 3. Position the tail "behind" the force so the tip hits the fly (rx, rz)
            # We subtract the direction vector to find the tail position
            start_x = rx - d_x * 1.5  # 1.5 is a spacing multiplier
            start_z = rz - d_z * 1.5
    
            arrow_force.set_data(x=start_x, y=start_z, dx=d_x, dy=d_z)

        else:
            arrow_force.set_alpha(0.0)
            
        if is_torque:
            text_torque.set_alpha(1.0)
            text_torque.set_position((rx, rz + 0.02)) 

            # Check the torque value from Config
            if Config.PERTURB_TORQUE > 0:
                text_torque.set_text('⟲') # Pitch Up / Counter-Clockwise
            else:
                text_torque.set_text('⟳') # Pitch Down / Clockwise
        else:
            text_torque.set_alpha(0.0)
            
        if is_force or is_torque:
            txt_info.set_text(f"STATUS: !! KICK (F/T) !!")
            txt_info.set_color('red')
        else:
            txt_info.set_text(f"STATUS: STABLE HOVER")
            txt_info.set_color('green')
            
        txt_time.set_text(f"T: {times[idx]:.4f}s")
        
        return [patch_thorax, patch_head, patch_abd, traj_line, wing_traj_line, arrow_force, text_torque, txt_time, txt_info, patch_le, patch_hinge] + wing_lines

    num_frames = int(len(r_states))
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/Config.FPS, blit=True)
    
    out_name = "hornet_flight_inference.gif"
    print(f"--> Saving to {out_name}...")
    ani.save(out_name, writer='pillow', fps=Config.FPS)
    print("--> Done!")
    plt.close(fig)

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import glob
    import re

    # Parse arguments for checkpoint selection
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the .pkl checkpoint file")
    args = parser.parse_args()

    param_file = args.checkpoint

    # Auto-detect latest checkpoint if none provided
    if param_file is None:
        default_dir = "checkpoints_shac"
        if os.path.exists(default_dir):
            files = glob.glob(os.path.join(default_dir, "*.pkl"))
            if files:
                # Sort by iteration number in filename
                files.sort(key=lambda f: int(re.sub(r'\D', '', f)) if re.search(r'\d', f) else 0)
                param_file = files[-1]
                print(f"--> Auto-detected latest checkpoint: {param_file}")
    
    if param_file is None or not os.path.exists(param_file):
        print(f"Error: Checkpoint file not found. Please provide one using --checkpoint")
        print(f"Example: python -m hornet.inference_hornet --checkpoint my_params.pkl")
        exit(1)
        
    print(f"--> Loading parameters from {param_file}")
    with open(param_file, 'rb') as f:
        data = pickle.load(f)

    # 1. Get raw params (Batch Size: 32)
    raw_params = data['params'] 
    
    # 2. Check if PBT state exists to find the best agent
    if 'pbt_state' in data:
        pbt_state = data['pbt_state']
        # Find index of agent with highest running reward
        best_idx = np.argmax(pbt_state.running_reward)
        print(f"--> PBT Detected. Selecting Best Agent: Index {best_idx}")
        print(f"    Score: {pbt_state.running_reward[best_idx]:.2f}")
        print(f"    Weights: {pbt_state.weights[best_idx]}")
        
        # Extract specific index from the batch
        params = jax.tree.map(lambda x: x[best_idx], raw_params)
        
    else:
        # Fallback: Just take the first one if no PBT state (or if it's a legacy checkpoint)
        print("--> No PBT state found. Using Agent 0.")
        # Check if batched by looking at first leaf
        first_leaf = jax.tree_util.tree_leaves(raw_params)[0]
        if len(first_leaf.shape) > 2: # Heuristic: Dense weights usually 2D, Batched is 3D
             params = jax.tree.map(lambda x: x[0], raw_params)
        else:
             params = raw_params # It was already unbatched (e.g., single expert)
        
    sim_data, env = run_simulation(params)
    generate_gif(sim_data, env)