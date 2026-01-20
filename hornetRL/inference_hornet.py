import os
import time

# --- FORCE CPU (CRITICAL FOR VISUALIZATION ON M1/M2) ---
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import haiku as hk
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# --- USER MODULES ---
from .environment_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

# ==============================================================================
# 1. CONFIGURATION (SYNCED WITH TRAINING)
# ==============================================================================
class Config:
    # --- TRAINED CONSTANTS (MUST MATCH TRAINING SCRIPT) ---
    BASE_FREQ = 115.0  
    
    # [x, z, theta, phi, vx, vz, w_theta, w_phi]
    TARGET_STATE = jnp.array([0.0, 0.0, 1.08, 0.3, 0.0, 0.0, 0.0, 0.0])
    
    # Normalization Scale
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

    # --- SIMULATION SETTINGS ---
    DT = 3e-5            
    SIM_SUBSTEPS = 20    # Physics steps per Brain step
    
    # --- INFERENCE SETTINGS ---
    DURATION = 0.5       # Seconds to simulate
    FPS = 60             # GIF Frame rate
    DPI = 150            # Resolution
    
    # Visualization Skip: 1 = All frames, 4 = Slow Mo, 10 = Fast, 20 = Realtime-ish
    # [MEMORY NOTE]: 20 is recommended for long durations to prevent crashes.
    VIZ_STEP_SKIP = 100 
    
    # --- PERTURBATION (WIND GUST) ---
    PERTURBATION = True  
    PERTURB_TIME = 0.20  # Apply force at 0.2s
    
    # Force: [X, Z] Newtons
    PERTURB_FORCE = jnp.array([0.2, -0.1]) 
    # Torque: Positive = Pitch Up
    PERTURB_TORQUE = -0.0005 

# ==============================================================================
# 2. MODEL DEFINITION (EXACT MATCH TO TRAINING)
# ==============================================================================
def actor_critic_fn(robot_state):
    # 1. Actor (Brain + Muscles)
    # We must pass the same args as training to ensure parameter alignment
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=Config.TARGET_STATE,
        obs_scale=Config.OBS_SCALE
    )
    
    # Dummy Value function (Inference doesn't use the critic, 
    # but we need the return structure to match the checkpoint)
    value = jnp.zeros((1, 1))
    
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. INFERENCE ENVIRONMENT
# ==============================================================================
class InferenceFlyEnv:
    def __init__(self):
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=Config.BASE_FREQ
        )

    def reset(self, key):
        # 1. Init Robot State (Start at Hover Target)
        batch_size = 1
        
        # Start exactly at target position
        q_pos = jnp.array([[0.0, 0.0]]) 
        # Start exactly at target angles (Theta=1.08, Phi=0.3)
        q_ang = jnp.array([[1.08, 0.3]])
        v = jnp.zeros((batch_size, 4))
        
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # 2. Init Oscillator
        osc_state = OscillatorState.init(base_freq=Config.BASE_FREQ)
        osc_state = jax.tree.map(lambda x: jnp.stack([x]*batch_size), osc_state)

        # 3. Calculate Wing Pose for Fluid Init (Centered)
        zero_action = jnp.zeros((batch_size, 9)) # Updated to 9 to match training action dim
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles, k_rates = ret[0], ret[1]
        
        robot_state_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_dummy, k_angles, k_rates)
        
        # --- Center Pose Logic (Matches Training) ---
        def get_centered_pose(r_state, w_pose_glob, bias_val):
            q, theta = r_state[:4], r_state[2]
            h_x, h_z = self.phys.robot.hinge_offset_x, self.phys.robot.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            total_st_ang = theta + self.phys.robot.stroke_plane_angle
            c_st, s_st = jnp.cos(total_st_ang), jnp.sin(total_st_ang)
            bias_glob_x, bias_glob_z = bias_val * c_st, bias_val * s_st
            
            off_x = hinge_glob_x + bias_glob_x
            off_z = hinge_glob_z + bias_glob_z
            
            p_x = w_pose_glob[0] - (q[0] + off_x)
            p_y = w_pose_glob[1] - (q[1] + off_z)
            return jnp.array([p_x, p_y, w_pose_glob[2]])

        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias)
        
        def init_fluid_fn(wp): return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state)

    def step(self, full_state, action_mods, external_force=jnp.zeros(2), external_torque=0.0):
        robot_st, fluid_st, osc_st = full_state
        
        # Unpack Batch (Simulate 1 agent)
        r = robot_st[0]
        f = jax.tree.map(lambda x: x[0], fluid_st) 
        o = jax.tree.map(lambda x: x[0], osc_st)
        a = action_mods[0]

        def sub_step_fn(carry, _):
            curr_r, curr_f, curr_o = carry
            
            # Oscillator
            o_next = step_oscillator(curr_o, unpack_action(a), Config.DT)
            k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
            action_data = (k_angles, k_rates, tau_abd, bias)

            # Physics
            (r_next_v, f_next), _, f_nodal, wing_pose, hinge_marker = self.phys.step(
                self.phys.fluid.params, (curr_r, curr_f), action_data, 0.0, Config.DT
            )
            
            # --- PERTURBATION LOGIC ---
            total_mass = self.phys.robot.m_thorax + self.phys.robot.m_abdomen
            accel_lin = external_force / total_mass
            r_next_v = r_next_v.at[4:6].add(accel_lin * Config.DT)
            
            inertia = self.phys.robot.I_thorax
            accel_ang = external_torque / inertia
            r_next_v = r_next_v.at[6].add(accel_ang * Config.DT)
            
            # Accumulate viz frame (Added hinge_marker to match training viz update)
            viz_frame = (r_next_v, wing_pose, f_nodal, f_next.marker_le, hinge_marker)

            return (r_next_v, f_next, o_next), viz_frame

        init_carry = (r, f, o)
        
        # Scan returns history of all substeps
        (final_r, final_f, final_o), stacked_viz_frames = jax.lax.scan(
            sub_step_fn, init_carry, None, length=Config.SIM_SUBSTEPS
        )

        # Repack Batch
        r_b = jnp.expand_dims(final_r, 0)
        o_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_o)
        f_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_f)
        
        return (r_b, f_b, o_b), stacked_viz_frames

# ==============================================================================
# 4. MAIN SIMULATION LOOP (GLOBAL SKIP FIX APPLIED HERE)
# ==============================================================================
def run_simulation(params):
    print("--> Initializing Inference Environment...")
    env = InferenceFlyEnv()
    rng = jax.random.PRNGKey(0)
    
    state = env.reset(rng)
    
    total_control_steps = int(Config.DURATION / (Config.DT * Config.SIM_SUBSTEPS))
    
    vis_data = {'r': [], 'w': [], 'f': [], 't': [], 'p_force': [], 'p_torque': [], 'le': [], 'hinge': []}
    
    print(f"--> Simulating {Config.DURATION}s ({total_control_steps} control steps)...")
    
    t_sim = 0.0
    
    # [REMOVED] Old local keep_indices logic was here. 
    # We now calculate it dynamically inside the loop.

    for i in range(total_control_steps):
        r_state = state[0]

        # --- PRE-PROCESSING ---
        wrapped_theta = jnp.mod(r_state[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_input = r_state.at[:, 2].set(wrapped_theta)
        obs_input = obs_input / Config.OBS_SCALE
        
        mods, _, _ = ac_model.apply(params, obs_input)
        
        # --- DISTURBANCE ---
        ext_f = jnp.zeros(2)
        ext_t = 0.0
        
        if Config.PERTURBATION:
            # (Preserved your update to 0.02s duration)
            if Config.PERTURB_TIME <= t_sim <= Config.PERTURB_TIME + 0.02:
                ext_f = Config.PERTURB_FORCE
                ext_t = Config.PERTURB_TORQUE
        
        # --- STEP PHYSICS ---
        state, stacked_frames = env.step(state, mods, external_force=ext_f, external_torque=ext_t)
        
        s_r, s_w, s_f, s_le, s_hinge = stacked_frames
        
        # --- [NEW] GLOBAL SKIP LOGIC ---
        # 1. Determine the Global Step ID for the start of this batch
        global_start_step = i * Config.SIM_SUBSTEPS
        
        # 2. Create an array of global steps for this specific batch
        # e.g., Batch 0 is [0..19], Batch 1 is [20..39]
        batch_global_steps = np.arange(global_start_step, global_start_step + Config.SIM_SUBSTEPS)
        
        # 3. Find which steps in this batch match the skip criteria
        # If SKIP=100, this returns [0] for the first batch, and [] (empty) for the next 4 batches.
        indices_to_keep = np.where(batch_global_steps % Config.VIZ_STEP_SKIP == 0)[0]
        
        # 4. Only extend lists if we actually found frames to keep in this batch
        if len(indices_to_keep) > 0:
            vis_data['r'].extend([np.array(s_r[j]) for j in indices_to_keep])
            vis_data['w'].extend([np.array(s_w[j]) for j in indices_to_keep])
            vis_data['f'].extend([np.array(s_f[j]) for j in indices_to_keep])
            vis_data['le'].extend([np.array(s_le[j]) for j in indices_to_keep])
            vis_data['hinge'].extend([np.array(s_hinge[j]) for j in indices_to_keep])
            
            step_times = [t_sim + (j+1)*Config.DT for j in indices_to_keep]
            vis_data['t'].extend(step_times)
            
            # Flags
            has_force = np.linalg.norm(ext_f) > 0
            has_torque = abs(ext_t) > 0
            vis_data['p_force'].extend([has_force] * len(indices_to_keep))
            vis_data['p_torque'].extend([has_torque] * len(indices_to_keep))
        
        t_sim += (Config.DT * Config.SIM_SUBSTEPS)
        
        if i % 100 == 0:
            print(f"    Progress: {i}/{total_control_steps} blocks | T={t_sim:.3f}s")

    return vis_data, env

# ==============================================================================
# 5. VISUALIZATION ENGINE (UPDATED WITH WING TRACING)
# ==============================================================================
def generate_gif(data, env):
    # This number will now be much smaller (e.g. ~833 instead of 16660)
    print(f"--> Rendering High-Quality GIF ({len(data['r'])} frames collected)...")
    
    r_states = data['r']
    w_poses = data['w']
    times = data['t']
    flag_force = data['p_force']
    flag_torque = data['p_torque']
    le_markers = data['le']
    hinge_markers = data['hinge']
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=Config.DPI)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f5')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5)
    
    # --- GRAPHICS OBJECTS ---
    traj_line, = ax.plot([], [], color='#555555', linestyle='--', linewidth=1.0, alpha=0.5)

    # [NEW] Wing Center Trajectory (Cyan Dotted Line)
    wing_traj_line, = ax.plot([], [], color='cyan', linestyle=':', linewidth=0.5, alpha=0.6, label='Wing Path')
    
    patch_thorax = patches.Ellipse((0,0), width=0.012, height=0.006, facecolor='#2c3e50', edgecolor='k', zorder=10)
    patch_head = patches.Circle((0,0), radius=0.0025, facecolor='#e74c3c', edgecolor='k', zorder=10)
    patch_abd = patches.Ellipse((0,0), width=0.018, height=0.008, facecolor='#f39c12', edgecolor='k', alpha=0.9, zorder=9)
    
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)
    
    # Wing (Motion Blur)
    wing_lines = []
    alphas = [0.05, 0.1, 0.2, 1.0] 
    for a in alphas:
        wl, = ax.plot([], [], 'k-', linewidth=1.5, alpha=a, zorder=11)
        wing_lines.append(wl)

    # Leading Edge Marker
    patch_le = patches.Circle((0,0), radius=0.0015, color='red', zorder=15)
    ax.add_patch(patch_le)

    # Hinge Marker
    patch_hinge = patches.Circle((0,0), radius=0.0015, color='orange', zorder=15)
    ax.add_patch(patch_hinge)
        
    # --- PERTURBATION VISUALS ---
    arrow_force = patches.FancyArrow(0, 0, 0, 0, width=0.005, color='red', zorder=20, alpha=0.0)
    ax.add_patch(arrow_force)
    
    text_torque = ax.text(0, 0, '⟳', fontsize=30, color='orange', ha='center', va='center', fontweight='bold', zorder=20, alpha=0.0)
    
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
        # [MEMORY FIX] Use frame index directly (data is already skipped)
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
        
        # --- Update Wings ---
        for k in range(4):
            # We must scale the offset because our data density changed
            # Previously: offset=3 frames (3*3e-5s). 
            # Now: 1 frame = 20*3e-5s. So offset should be 1.
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

        # --- [NEW] Update Trajectory (Wing Center) ---
        # 1.5 Wingbeats @ 115Hz = ~0.013s
        # 0.013s / 0.0006s per frame = ~22 frames
        wing_hist_len = 22
        start_w = max(0, idx - wing_hist_len)
        wing_hist_x = [w[0] for w in w_poses[start_w:idx]]
        wing_hist_z = [w[1] for w in w_poses[start_w:idx]]
        wing_traj_line.set_data(wing_hist_x, wing_hist_z)
        
        # --- VISUALIZE KICK ---
        is_force = flag_force[idx]
        is_torque = flag_torque[idx]
        
        # Force Arrow
        if is_force:
            arrow_force.set_alpha(1.0)
            arrow_force.set_data(x=rx-0.05, y=rz, dx=0.03, dy=0) 
        else:
            arrow_force.set_alpha(0.0)
            
        # Torque Marker
        if is_torque:
            text_torque.set_alpha(1.0)
            text_torque.set_position((rx, rz + 0.02)) 
        else:
            text_torque.set_alpha(0.0)
            
        # Status Text
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

    # Allow user to pass a specific checkpoint file
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the .pkl checkpoint file")
    args = parser.parse_args()

    param_file = args.checkpoint

    # If no file provided, try to find the latest in the default folder
    if param_file is None:
        default_dir = "checkpoints_shac"
        if os.path.exists(default_dir):
            files = glob.glob(os.path.join(default_dir, "*.pkl"))
            if files:
                # Sort by number in filename
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
        params = data['params'] if 'params' in data else data
        
    sim_data, env = run_simulation(params)
    generate_gif(sim_data, env)