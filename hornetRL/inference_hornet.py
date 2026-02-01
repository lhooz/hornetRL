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
from matplotlib.lines import Line2D

# --- USER MODULES ---
from .fluid_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem, PhysParams
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # --- Model Constants ---
    BASE_FREQ = 115.0  
    TARGET_STATE = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    # --- Simulation Settings ---
    DT = 3e-5             
    SIM_SUBSTEPS = 72
    
    FPS = 60             
    DPI = 80            
    VIZ_STEP_SKIP = 10

    # --- Mode 1: Nominal GIF Settings ---
    DURATION = 0.1  
    TRACE_HIST_LEN = 40   # ~1.5 wingbeats
    N_SHADOW_WINGS = 14
    
    # --- Mode 2: Chaos Plot Settings ---
    CHAOS_DURATION = 0.2
    CHAOS_BATCH_SIZE = 20
    
    # --- Physics Settings ---
    USE_DOMAIN_RANDOMIZATION = True 
    PERTURBATION = True  
    PERTURB_TIME = 0.02  
    PERTURB_FORCE = jnp.array([1.0, -1.5]) 
    PERTURB_TORQUE = -0.003

def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state):
    target_sym = symlog(Config.TARGET_STATE)
    mods, forces = policy_network_icnn(robot_state, target_state=target_sym)
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(robot_state)
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. ENVIRONMENT (Handles both Single and Batch)
# ==============================================================================
class InferenceFlyEnv:
    def __init__(self):
        self.phys = FlappingFlySystem(model_path='fluid.pkl', target_freq=Config.BASE_FREQ)

    def reset(self, key, batch_size=1, mode='nominal'):
        """
        Initializes the fly/flies. 
        mode='nominal': Starts at target (Stable Hover)
        mode='chaos': Starts at random positions (Recovery Test)
        """
        
        if mode == 'chaos':
            # --- CHAOS INITIALIZATION ---
            k1_c, k2_c = jax.random.split(key)
            k_theta, k_phi = jax.random.split(k2_c)
            
            # Position: Wide window
            q_pos = jax.random.uniform(k1_c, (batch_size, 2), minval=-0.25, maxval=0.25)
            
            # Pitch: Random range
            theta_chaos = jax.random.uniform(k_theta, (batch_size, 1), minval=-2.5, maxval=2.5)
            theta_chaos = theta_chaos + 1.0
            
            # Abdomen: Random range
            phi_chaos = jax.random.uniform(k_phi, (batch_size, 1), minval=-0.3, maxval=0.3)
            phi_chaos = phi_chaos + 0.2
            
            q_ang = jnp.concatenate([theta_chaos, phi_chaos], axis=-1)
            v = jax.random.normal(key, (batch_size, 4)) * 0.1 
            
        else:
            # --- NOMINAL INITIALIZATION (Hover) ---
            q_pos = jnp.zeros((batch_size, 2))
            q_ang = jnp.tile(jnp.array([[1.0, 0.2]]), (batch_size, 1))
            v = jnp.zeros((batch_size, 4))

        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # Init Oscillator
        osc_state = OscillatorState.init(base_freq=Config.BASE_FREQ)
        osc_state = jax.tree.map(lambda x: jnp.stack([x]*batch_size), osc_state)

        # --- PHYSICS PARAMETERS ---
        if Config.USE_DOMAIN_RANDOMIZATION:
            k1, k2, k3, k4, k_shuffle = jax.random.split(key, 5)
            k_mass, k_com, k_hinge, k_st, k_joint = jax.random.split(k3, 5)
            
            mass_scale_th = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)
            mass_scale_ab = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)
            off_x_th = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)
            off_x_ab = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)
            h_x_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)
            h_z_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)
            st_ang_noise = jax.random.uniform(k_st, (batch_size,), minval=-0.08, maxval=0.08)
            k_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.7, maxval=1.3)
            b_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.7, maxval=1.3)
            phi_eq_off = jax.random.uniform(k_joint, (batch_size,), minval=-0.1, maxval=0.1)
        else:
            mass_scale_th = jnp.ones(batch_size)
            mass_scale_ab = jnp.ones(batch_size)
            off_x_th = jnp.zeros(batch_size)
            off_x_ab = jnp.zeros(batch_size)
            h_x_noise = jnp.zeros(batch_size)
            h_z_noise = jnp.zeros(batch_size)
            st_ang_noise = jnp.zeros(batch_size)
            k_hinge_scale = jnp.ones(batch_size)
            b_hinge_scale = jnp.ones(batch_size)
            phi_eq_off = jnp.zeros(batch_size)

        phys_params = PhysParams(
            thorax_mass_scale=mass_scale_th, abd_mass_scale=mass_scale_ab,
            thorax_offset_x=off_x_th, abd_offset_x=off_x_ab,
            hinge_x_noise=h_x_noise, hinge_z_noise=h_z_noise,
            stroke_ang_noise=st_ang_noise,
            k_hinge_scale=k_hinge_scale, b_hinge_scale=b_hinge_scale,
            phi_equil_offset=phi_eq_off
        )

        self.active_props = jax.vmap(self.phys.robot.compute_props)(phys_params)

        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles, k_rates = ret[0], ret[1]
        
        robot_state_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_dummy, k_angles, k_rates, self.active_props)
        
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

        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias, self.active_props)
        def init_fluid_fn(wp): return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state, phys_params)

    def step(self, full_state, action_mods, external_force=jnp.zeros(2), external_torque=0.0):
        """
        Step function supporting batching.
        Returns detailed wing data ONLY for batch_size=1 (Nominal) to save memory.
        """
        robot_st, fluid_st, osc_st, phys_p = full_state
        batch_size = robot_st.shape[0]
        
        # Sub-step function to map over batch
        def sub_step_vmap_fn(curr_r, curr_f, curr_o, curr_a, curr_p, ext_f, ext_t):
            # --- 1. Compute Physical Properties for this agent ---
            curr_props = self.phys.robot.compute_props(curr_p)

            # --- 2. Update Oscillator ---
            o_next = step_oscillator(curr_o, unpack_action(curr_a), Config.DT)
            k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(curr_a))
            action_data = (k_angles, k_rates, tau_abd, bias)

            # --- 3. Update Physics ---
            (r_next_v, f_next), _, f_nodal, wing_pose, hinge_marker = self.phys.step(
                self.phys.fluid.params, (curr_r, curr_f), action_data, curr_props, 0.0, Config.DT
            )
            
            # Apply Perturbation
            total_mass = curr_props.m_thorax + curr_props.m_abdomen
            accel_lin = ext_f / total_mass
            r_next_v = r_next_v.at[4:6].add(accel_lin * Config.DT)
            
            inertia = curr_props.I_thorax
            accel_ang = ext_t / inertia
            r_next_v = r_next_v.at[6].add(accel_ang * Config.DT)
            
            # Return full visualization data
            return (r_next_v, f_next, o_next), (wing_pose, f_nodal, f_next.marker_le, hinge_marker)

        # Scan over time steps
        def scan_fn(carry, _):
            c_r, c_f, c_o = carry
            
            # Map the physics step over all agents in the batch
            (n_r, n_f, n_o), (w_p, f_n, le, h) = jax.vmap(sub_step_vmap_fn, in_axes=(0,0,0,0,0,None,None))(
                c_r, c_f, c_o, action_mods, phys_p, external_force, external_torque
            )
            
            # OPTIMIZATION: If simulating swarm (chaos), don't store wing/fluid data history
            # Only store it for single-agent detailed analysis
            viz_frame = (n_r, w_p, f_n, le, h) if batch_size == 1 else n_r
            
            return (n_r, n_f, n_o), viz_frame

        (final_r, final_f, final_o), stacked_viz_frames = jax.lax.scan(
            scan_fn, (robot_st, fluid_st, osc_st), None, length=Config.SIM_SUBSTEPS
        )

        return (final_r, final_f, final_o, phys_p), stacked_viz_frames

# ==============================================================================
# 4. SIMULATION LOOP (Dual Mode)
# ==============================================================================
def run_simulation(params, mode='nominal'):
    print(f"--> Initializing Environment (Mode: {mode})...")
    env = InferenceFlyEnv()
    rng = jax.random.PRNGKey(int(time.time()))
    
    if mode == 'chaos':
        batch_size = Config.CHAOS_BATCH_SIZE
        duration = Config.CHAOS_DURATION
    else:
        batch_size = 1
        duration = Config.DURATION

    state = env.reset(rng, batch_size=batch_size, mode=mode)
    
    # Capture ALL scales for chaos plotting
    phys_p_batch = state[3]
    all_scales_th = np.array(phys_p_batch.thorax_mass_scale)
    
    # Store single-agent physics scale for GIF
    p_single = jax.tree.map(lambda x: x[0], state[3])
    th_scale = float(p_single.thorax_mass_scale)
    ab_scale = float(p_single.abd_mass_scale)
    
    print(f"--> Simulating {batch_size} Agent(s) for {duration}s...")
    
    total_control_steps = int(duration / (Config.DT * Config.SIM_SUBSTEPS))
    
    # Data containers
    traj_history = []  # For Chaos (Position only)
    detailed_history = {'r': [], 'w': [], 'f': [], 't': [], 'p_force': [], 'p_torque': [], 
                        'meta': {'th_scale': th_scale, 'ab_scale': ab_scale}}

    @jax.jit
    def inference_step(curr_state, curr_params, ext_f, ext_t):
        r_state = curr_state[0]
        wrapped_theta = jnp.mod(r_state[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_input = r_state.at[:, 2].set(wrapped_theta)
        scaled_input = symlog(obs_input)
        mods, _, _ = ac_model.apply(curr_params, scaled_input)
        next_state, frames = env.step(curr_state, mods, external_force=ext_f, external_torque=ext_t)
        return next_state, frames

    # Warmup
    _ = inference_step(state, params, jnp.zeros(2), 0.0)
    print("--> JAX Compilation Complete.")

    t_sim = 0.0
    for i in range(total_control_steps):
        ext_f = jnp.zeros(2)
        ext_t = 0.0
        
        # Apply wind only in Nominal mode
        if mode == 'nominal' and Config.PERTURBATION:
            block_end_time = t_sim + (Config.DT * Config.SIM_SUBSTEPS)
            if (t_sim <= Config.PERTURB_TIME + 0.002) and (block_end_time >= Config.PERTURB_TIME):
                ext_f = Config.PERTURB_FORCE
                ext_t = Config.PERTURB_TORQUE
        
        state, stacked_frames = inference_step(state, params, ext_f, ext_t)
        
        # --- DATA COLLECTION ---
        start_step_idx = i * Config.SIM_SUBSTEPS
        block_indices = np.arange(start_step_idx, start_step_idx + Config.SIM_SUBSTEPS)
        matches = np.where(block_indices % Config.VIZ_STEP_SKIP == 0)[0]
        
        if len(matches) > 0:
            if mode == 'chaos':
                for m in matches:
                    r_snap = np.array(stacked_frames[m]) # [Batch, 8]
                    traj_history.append(r_snap)
            else:
                s_r, s_w, s_f, _, _ = stacked_frames 
                for m in matches:
                    sub_t = t_sim + (m * Config.DT)
                    detailed_history['r'].append(np.array(s_r[m, 0]))
                    detailed_history['w'].append(np.array(s_w[m, 0]))
                    detailed_history['f'].append(np.array(s_f[m, 0]))
                    detailed_history['t'].append(sub_t)
                    is_p = (Config.PERTURB_TIME <= sub_t <= Config.PERTURB_TIME + 0.002)
                    detailed_history['p_force'].append(is_p and np.linalg.norm(ext_f) > 0)
                    detailed_history['p_torque'].append(is_p and abs(ext_t) > 0)

        t_sim += (Config.DT * Config.SIM_SUBSTEPS)
        if i % 100 == 0:
            print(f"    Progress: {i}/{total_control_steps} blocks | T={t_sim:.3f}s")

    if mode == 'chaos':
        return traj_history, all_scales_th, env # Return scales for plotting
    else:
        return detailed_history, env

# ==============================================================================
# 5. VISUALIZATION A: SWARM TRAJECTORY GIF
# ==============================================================================
def generate_chaos_plot(history, scales):
    print("\n--> Rendering Swarm Trajectory GIF...")

    # Data Prep
    data = np.stack(history, axis=0) 
    num_steps, num_agents, _ = data.shape
    time_per_frame = Config.DT * Config.VIZ_STEP_SKIP
    m_min, m_max = np.min(scales), np.max(scales)
    init_pitch = data[0, :, 2]
    p_min, p_max = np.min(init_pitch), np.max(init_pitch)
    
    # Magnified sizes for visual clarity (Cubic scaling)
    base_size = 45
    viz_sizes = base_size * (scales ** 3)

    # --- 1. SETUP: Identical Geometry to Single Agent ---
    dark_bg = '#1a1a1a'
    fig, ax = plt.subplots(figsize=(10, 10), dpi=Config.DPI, facecolor=dark_bg) 
    # This specific adjustment ensures the plot box is the exact same pixel size in both GIFs
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.90) 
    ax.set_aspect('equal')
    ax.set_facecolor(dark_bg)
    ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
    
    # --- 2. STYLING: Dark Theme & Limits ---
    for spine in ax.spines.values(): spine.set_color('#555555')
    ax.tick_params(colors='#888888', labelsize=8)
    
    # Limits (Swarm covers more area, so 0.3 is appropriate vs 0.2 for single)
    limit = 0.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Labels & Title
    ax.set_xlabel("X Position (m)", color='#aaaaaa', fontsize=10, fontweight='bold')
    ax.set_ylabel("Z Position (m)", color='#aaaaaa', fontsize=10, fontweight='bold')
    ax.set_title(f"Swarm Recovery Analysis (N={num_agents})\n"
                 f"Mass: [{m_min:.2f}-{m_max:.2f}M] | Initial Pitch: [{p_min:.2f}-{p_max:.2f} rad]", 
                 color='white', fontsize=11, pad=12, fontweight='bold')

    # --- 3. OBJECTS ---
    # Target Zone (Circle)
    target_zone = patches.Circle((0, 0), radius=0.01, facecolor='#2ecc71', alpha=0.1, 
                                 edgecolor='#27ae60', linestyle='--', linewidth=1.5, zorder=1)
    ax.add_patch(target_zone)
    
    # Target Crosshair (Assign to variable for blitting)
    target_cross = ax.scatter(0, 0, color='#2ecc71', s=100, marker='+', zorder=5, lw=1.5, alpha=0.8)
    
    # Agents (Viridis colormap pops against dark bg)
    colors = plt.cm.viridis(np.linspace(0, 1, num_agents)) 
    scatter = ax.scatter(data[0, :, 0], data[0, :, 1], s=viz_sizes, c=colors, 
                         alpha=0.85, zorder=10, edgecolors='#2c3e50', linewidths=0.5)
    
    # Trails (Low alpha for "flow" effect)
    trails = [ax.plot([], [], color=c, alpha=0.3, lw=0.8)[0] for c in colors] 

    # Time Text
    txt_time = ax.text(0.05, 0.93, '', transform=ax.transAxes, 
                       fontsize=11, family='monospace', fontweight='bold', color='#2ecc71')

    def update(frame):
        # Update Heads
        current_pos = data[frame, :, :2]
        scatter.set_offsets(current_pos)
        
        # Update Trails
        for i, line in enumerate(trails):
            line.set_data(data[:frame+1, i, 0], data[:frame+1, i, 1])
            
        # Time Display
        current_time = frame * time_per_frame
        txt_time.set_text(f"TIME: {current_time:.3f}s")
        
        # Return ALL artists (including static ones) for correct blitting
        return [scatter, txt_time, target_zone, target_cross] + trails

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=30, blit=True)
    
    out_name = "hornet_swarm_traj.gif"
    print(f"--> Saving Swarm GIF to {out_name}...")
    try:
        ani.save(out_name, writer='pillow', fps=30)
        print(f"--> Done! Saved: {out_name}")
    except Exception as e:
        print(f"--> Error: {e}")
    plt.close(fig)

# ==============================================================================
# 5b. VISUALIZATION B: PROFESSIONAL GIF (Single Agent)
# ==============================================================================
def generate_gif(data, env):
    print(f"\n--> Rendering Professional GIF ({len(data['r'])} frames)...")
    r_states, w_poses = data['r'], data['w']
    times = data['t']
    flag_force, flag_torque = data['p_force'], data['p_torque']
    
    # Scales for visualization
    th_viz_scale = np.cbrt(data['meta']['th_scale'])
    ab_viz_scale = np.cbrt(data['meta']['ab_scale'])
    
    # Pre-calculate Wing Coords (Local Frame)
    wing_rel_history = []
    for i in range(len(r_states)):
        r, w = r_states[i], w_poses[i]
        dx, dz = w[0] - r[0], w[1] - r[1]
        c, s = np.cos(-r[2]), np.sin(-r[2])
        wing_rel_history.append([dx * c - dz * s, dx * s + dz * c])
    wing_rel_history = np.array(wing_rel_history)

    # --- 1. SETUP: Identical Geometry to Swarm ---
    dark_bg = '#1a1a1a'
    fig, ax = plt.subplots(figsize=(10, 10), dpi=Config.DPI, facecolor=dark_bg)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.90) # MATCHED EXACTLY
    ax.set_aspect('equal')
    ax.set_facecolor(dark_bg)
    ax.grid(True, color='#333333', linestyle='-', linewidth=0.5)

    for spine in ax.spines.values(): spine.set_color('#555555')
    ax.tick_params(colors='#888888', labelsize=8)

    # --- 2. OBJECTS ---
    # Target 
    target_zone = patches.Circle((0, 0), radius=0.01, facecolor='#2ecc71', alpha=0.08, 
                                 edgecolor='#27ae60', linestyle='--', linewidth=1.2, zorder=1, label='Target')
    ax.add_patch(target_zone)
    
    # Trails (Adjusted alphas for better visibility)
    traj_line, = ax.plot([], [], color='#777777', linestyle='--', linewidth=1.0, alpha=0.6, label='CoM Path')
    wing_traj_line, = ax.plot([], [], color='#bb86fc', linestyle='-', linewidth=1.2, alpha=1.0, label='Wing Trace')

    # --- BODY VISUALIZATION ---
    # "Metallic Silver" with High-Contrast Light Edges
    # Thorax: Metallic Mid-Gray fill, Bright Edge
    patch_thorax = patches.Ellipse((0,0), width=0.012*th_viz_scale, height=0.006*th_viz_scale, 
                                   facecolor='#666666',      # Lighter than background
                                   edgecolor='#d1d1d1',      # Very light gray edge (Pop!)
                                   linewidth=1.0,            # Slightly thicker line
                                   zorder=10)
    
    # Head: Light Silver fill
    patch_head = patches.Circle((0,0), radius=0.0025*th_viz_scale, 
                                facecolor='#aaaaaa',         # Bright silver
                                edgecolor='#d1d1d1',         # Matching edge
                                linewidth=1.0, 
                                zorder=10)
    
    # Abdomen: Darker Steel Gray (to show segmentation contrast)
    patch_abd = patches.Ellipse((0,0), width=0.018*ab_viz_scale, height=0.008*ab_viz_scale, 
                                facecolor='#4a4a4a',         # Visible steel gray (was #3d3d3d)
                                edgecolor='#999999',         # Medium-light edge
                                linewidth=1.0, 
                                alpha=1.0,                   # Removed alpha to prevent background bleed-through
                                zorder=9)
    
    ax.add_patch(patch_thorax); ax.add_patch(patch_head); ax.add_patch(patch_abd)
    
    # --- WINGS ---
    wing_lines = []
    shadow_alphas = np.linspace(0.05, 0.4, Config.N_SHADOW_WINGS) # Slightly higher opacity
    all_alphas = np.concatenate([shadow_alphas, [1.0]])
    
    for a in all_alphas:
        # If it's the main wing (a=1.0), use White/Silver.
        # If it's a shadow, also use White but let the alpha do the work.
        col = '#ffffff' # Pure white looks best against #1a1a1a
        lw = 1.5 if a == 1.0 else 1.0
        wl, = ax.plot([], [], col, linestyle='-', linewidth=lw, alpha=a, zorder=11)
        wing_lines.append(wl)
        
    # Indicators
    arrow_force = patches.FancyArrow(0, 0, 0, 0, width=0.005, color='#cf6679', zorder=20, alpha=0.0) # Muted red
    ax.add_patch(arrow_force)
    text_torque = ax.text(0, 0, '', fontsize=30, color='#ffb74d', ha='center', va='center', fontweight='bold', zorder=20, alpha=0.0) # Muted orange
    
    # Text Telemetry
    txt_time = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold', family='monospace', color='#cccccc')
    txt_info = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=10, family='monospace', color='#888888')
    
    # Helper for wing transform
    def get_wing_coords(r_pos, w_pose):
        wx, wz, wang = w_pose
        wing_len = env.phys.fluid.WING_LEN
        N_pts = 20
        x_local = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        return wx + x_local * c_w, wz + x_local * s_w

    # Legend (Visible on Dark BG)
    leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.1, labelcolor='#aaaaaa', edgecolor='#444444', shadow=False)
    leg.set_zorder(100)

    window_size = 0.2

    def update(frame):
        if frame >= len(r_states): return []
        curr_r = r_states[frame]
        rx, rz, r_th, r_phi = curr_r[0], curr_r[1], curr_r[2], curr_r[3]
        
        # Camera
        ax.set_xlim(0.0 - window_size, 0.0 + window_size)
        ax.set_ylim(0.0 - window_size, 0.0 + window_size)
        
        # Update Body
        patch_thorax.set_center((rx, rz)); patch_thorax.set_angle(np.degrees(r_th))
        d1, d2 = env.phys.robot.d1, env.phys.robot.d2
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        abd_ang = r_th + r_phi
        patch_abd.set_center((rx - d1 * np.cos(r_th) - d2 * np.cos(abd_ang), rz - d1 * np.sin(r_th) - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))
        
        # Update Wings
        num_wings = len(wing_lines)
        for k in range(num_wings):
            hist_idx = max(0, frame - (num_wings - 1 - k))
            wx, wz = get_wing_coords(r_states[hist_idx], w_poses[hist_idx])
            wing_lines[k].set_data(wx, wz)
            
        # Update Trails
        hist_len = 120
        start_t = max(0, frame - hist_len)
        traj_line.set_data([r[0] for r in r_states[start_t:frame]], [r[1] for r in r_states[start_t:frame]])
        
        start_w = max(0, frame - Config.TRACE_HIST_LEN)
        rel_chunk = wing_rel_history[start_w:frame]
        if len(rel_chunk) > 0:
            c, s = np.cos(r_th), np.sin(r_th)
            wing_traj_line.set_data(rel_chunk[:,0]*c - rel_chunk[:,1]*s + rx, rel_chunk[:,0]*s + rel_chunk[:,1]*c + rz)
        else:
            wing_traj_line.set_data([], [])
            
        # Perturbation Flags
        is_force, is_torque = flag_force[frame], flag_torque[frame]
        if is_force:
            arrow_force.set_alpha(1.0)
            fx, fz = Config.PERTURB_FORCE
            mag = np.sqrt(fx**2 + fz**2) + 1e-6
            arrow_force.set_data(x=rx - (fx/mag)*0.045, y=rz - (fz/mag)*0.045, dx=(fx/mag)*0.03, dy=(fz/mag)*0.03)
        else: arrow_force.set_alpha(0.0)
        
        if is_torque:
            text_torque.set_alpha(1.0)
            text_torque.set_position((rx, rz + 0.02))
            text_torque.set_text('⟲' if Config.PERTURB_TORQUE > 0 else '⟳')
        else: text_torque.set_alpha(0.0)
        
        if is_force or is_torque:
            txt_info.set_text("STATUS: !! GUST !!"); txt_info.set_color('#cf6679')
        else:
            txt_info.set_text("STATUS: STABLE HOVER"); txt_info.set_color('#2ecc71')
            
        txt_time.set_text(f"T: {times[frame]:.4f}s")
        
        # Return all for blit
        return [patch_thorax, patch_head, patch_abd, traj_line, wing_traj_line, 
                arrow_force, text_torque, txt_time, txt_info, leg, target_zone] + wing_lines

    ani = animation.FuncAnimation(fig, update, frames=len(r_states), interval=800/Config.FPS, blit=True)
    out_name = "hornet_flight_inference.gif"
    print(f"--> Saving to {out_name}...")
    try:
        ani.save(out_name, writer='pillow', fps=Config.FPS)
        print(f"--> Done! Saved: {out_name}")
    except Exception as e: print(f"--> Error: {e}")
    plt.close(fig)

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse, glob, re
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--chaos", action="store_true")
    args = parser.parse_args()

    param_file = args.checkpoint
    if not param_file:
        files = glob.glob("checkpoints_shac/*.pkl")
        if files: 
            files.sort(key=lambda f: int(re.sub(r'\D', '', f)) if re.search(r'\d', f) else 0)
            param_file = files[-1]
            print(f"--> Auto-detected: {param_file}")
    
    if not param_file: exit("Error: No checkpoint found.")
    
    print(f"--> Loading: {param_file}")
    with open(param_file, 'rb') as f: data = pickle.load(f)
    
    if 'pbt_state' in data:
        idx = np.argmax(data['pbt_state'].running_reward)
        print(f"--> Best Agent: {idx} (Score: {data['pbt_state'].running_reward[idx]:.2f})")
        params = jax.tree.map(lambda x: x[idx], data['params'])
    else:
        params = jax.tree.map(lambda x: x[0], data['params'])

    if args.chaos:
        hist, scales, _ = run_simulation(params, mode='chaos')
        generate_chaos_plot(hist, scales)
    else:
        hist, env = run_simulation(params, mode='nominal')
        generate_gif(hist, env)