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
    
    # --- Mode 1: Nominal GIF Settings ---
    DURATION = 1.0       
    FPS = 60             
    DPI = 60             
    VIZ_STEP_SKIP = 70
    TRACE_HIST_LEN = 6   # ~1.5 wingbeats
    N_SHADOW_WINGS = 7   
    
    # --- Mode 2: Chaos Plot Settings ---
    CHAOS_BATCH_SIZE = 10
    CHAOS_DURATION = 0.5 # Shorter time needed to see recovery
    
    # --- Physics Settings ---
    USE_DOMAIN_RANDOMIZATION = True 
    PERTURBATION = True  
    PERTURB_TIME = 0.02  
    PERTURB_FORCE = jnp.array([0.9, -1.2]) 
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
            
            # Position: Wide window (+/- 15cm)
            q_pos = jax.random.uniform(k1_c, (batch_size, 2), minval=-0.15, maxval=0.15)
            
            # Pitch: Random range (-1.5 to 1.5) + Offset 1.0 -> Range (-0.5 to 2.5)
            theta_chaos = jax.random.uniform(k_theta, (batch_size, 1), minval=-1.5, maxval=1.5)
            theta_chaos = theta_chaos + 1.0
            
            # Abdomen: Random range (-0.3 to 0.3) + Offset 0.2
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
            k_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
            b_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
            phi_eq_off = jax.random.uniform(k_joint, (batch_size,), minval=-0.17, maxval=0.17)
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
                    r_snap = np.array(stacked_frames[m])
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
# 5. VISUALIZATION A: CHAOS PLOT (With Size Scaling)
# ==============================================================================
def generate_chaos_plot(history, scales):
    print("\n--> Generating Chaos Trajectory Plot...")
    
    # history is list of (Batch, 8) arrays
    data = np.stack(history, axis=0) # Result: (Time, Batch, 8)
    num_steps, num_agents, _ = data.shape
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, color='#e0e0e0', linestyle='--', linewidth=0.5)
    
    ax.scatter(0, 0, color='#27ae60', s=150, marker='+', zorder=10, label='Target', linewidth=2)
    
    # Calculate Dot Sizes based on Mass (Cube Root scaling)
    # Norm scale 1.0 -> size 30. Scale 1.2 -> larger.
    base_size = 35
    viz_sizes = base_size * (scales ** (2/3)) # Surface area scaling for 2D dots
    
    for i in range(num_agents):
        traj_x = data[:, i, 0]
        traj_z = data[:, i, 1]
        
        # Use individual size
        s_i = viz_sizes[i]
        
        ax.plot(traj_x, traj_z, color='black', alpha=0.15, linewidth=0.8)
        ax.scatter(traj_x[0], traj_z[0], color='#e74c3c', s=s_i, marker='x', alpha=0.7)
        ax.scatter(traj_x[-1], traj_z[-1], color='#3498db', s=s_i, marker='.', alpha=0.8)

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.set_title(f"Swarm Recovery Analysis (N={num_agents})\nDot Size reflects Fly Mass (Randomized +/- 20%)")
    
    legend_elements = [
        Line2D([0], [0], marker='+', color='w', markeredgecolor='#27ae60', markersize=10, label='Target'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='#e74c3c', markersize=8, label='Start'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='#3498db', markersize=10, label='End'),
        Line2D([0], [0], color='black', lw=1, alpha=0.3, label='Trajectory'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    out_name = "chaos_recovery_plot.png"
    plt.savefig(out_name, bbox_inches='tight')
    print(f"--> Saved Chaos Plot: {os.path.abspath(out_name)}")
    plt.close(fig)

# ==============================================================================
# 5. VISUALIZATION B: PROFESSIONAL GIF (Single Agent)
# ==============================================================================
def generate_gif(data, env):
    print(f"\n--> Rendering Professional GIF ({len(data['r'])} frames)...")
    r_states = data['r']
    w_poses = data['w']
    times = data['t']
    flag_force = data['p_force']
    flag_torque = data['p_torque']
    
    th_viz_scale = np.cbrt(data['meta']['th_scale'])
    ab_viz_scale = np.cbrt(data['meta']['ab_scale'])
    
    wing_rel_history = []
    for i in range(len(r_states)):
        r, w = r_states[i], w_poses[i]
        dx_global = w[0] - r[0]
        dz_global = w[1] - r[1]
        c, s = np.cos(-r[2]), np.sin(-r[2])
        x_local = dx_global * c - dz_global * s
        z_local = dx_global * s + dz_global * c
        wing_rel_history.append([x_local, z_local])
    wing_rel_history = np.array(wing_rel_history)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=Config.DPI)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, color='#e0e0e0', linestyle='-', linewidth=1.0)
    
    traj_line, = ax.plot([], [], color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.8, label='CoM Path')
    wing_traj_line, = ax.plot([], [], color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.9, label='Wing Path')
    
    patch_thorax = patches.Ellipse((0,0), width=0.012*th_viz_scale, height=0.006*th_viz_scale, facecolor='#404040', edgecolor='none', zorder=10)
    patch_head = patches.Circle((0,0), radius=0.0025*th_viz_scale, facecolor='#b0b0b0', edgecolor='none', zorder=10)
    patch_abd = patches.Ellipse((0,0), width=0.018*ab_viz_scale, height=0.008*ab_viz_scale, facecolor='#707070', edgecolor='none', alpha=0.9, zorder=9)
    
    ax.add_patch(patch_thorax); ax.add_patch(patch_head); ax.add_patch(patch_abd)
    
    wing_lines = []
    shadow_alphas = np.linspace(0.02, 0.3, Config.N_SHADOW_WINGS)
    all_alphas = np.concatenate([shadow_alphas, [1.0]])
    for a in all_alphas:
        lw = 1.5 if a == 1.0 else 1.2
        col = 'k' if a == 1.0 else '#202020'
        wl, = ax.plot([], [], col, linestyle='-', linewidth=lw, alpha=a, zorder=11)
        wing_lines.append(wl)
        
    arrow_force = patches.FancyArrow(0, 0, 0, 0, width=0.005, color='#c0392b', zorder=20, alpha=0.0)
    ax.add_patch(arrow_force)
    text_torque = ax.text(0, 0, '', fontsize=30, color='#d35400', ha='center', va='center', fontweight='bold', zorder=20, alpha=0.0)
    txt_time = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold', family='monospace', color='k')
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
        idx = frame
        if idx >= len(r_states): return []
        
        curr_r = r_states[idx]
        rx, rz, r_th, r_phi = curr_r[0], curr_r[1], curr_r[2], curr_r[3]
        
        ax.set_xlim(0.0 - window_size, 0.0 + window_size)
        ax.set_ylim(0.0 - window_size, 0.0 + window_size)
        
        patch_thorax.set_center((rx, rz)); patch_thorax.set_angle(np.degrees(r_th))
        d1 = env.phys.robot.d1; patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        d2 = env.phys.robot.d2; abd_ang = r_th + r_phi
        patch_abd.set_center((rx - d1 * np.cos(r_th) - d2 * np.cos(abd_ang), rz - d1 * np.sin(r_th) - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))
        
        num_total_wings = len(wing_lines)
        for k in range(num_total_wings):
            offset = (num_total_wings - 1) - k
            hist_idx = max(0, idx - offset)
            w_x, w_z = get_wing_coords(r_states[hist_idx], w_poses[hist_idx])
            wing_lines[k].set_data(w_x, w_z)

        hist_len = 50; start_t = max(0, idx - hist_len)
        traj_line.set_data([r[0] for r in r_states[start_t:idx]], [r[1] for r in r_states[start_t:idx]])

        start_w = max(0, idx - Config.TRACE_HIST_LEN); rel_chunk = wing_rel_history[start_w:idx]
        if len(rel_chunk) > 0:
            c, s = np.cos(r_th), np.sin(r_th)
            wing_traj_line.set_data(rel_chunk[:,0]*c - rel_chunk[:,1]*s + rx, rel_chunk[:,0]*s + rel_chunk[:,1]*c + rz)
        else: wing_traj_line.set_data([], [])

        is_force = flag_force[idx]; is_torque = flag_torque[idx]
        if is_force:
            arrow_force.set_alpha(1.0)
            fx, fz = Config.PERTURB_FORCE; mag = np.sqrt(fx**2 + fz**2) + 1e-6
            arrow_force.set_data(x=rx - (fx/mag)*0.045, y=rz - (fz/mag)*0.045, dx=(fx/mag)*0.03, dy=(fz/mag)*0.03)
        else: arrow_force.set_alpha(0.0)
        
        if is_torque:
            text_torque.set_alpha(1.0); text_torque.set_position((rx, rz + 0.02))
            text_torque.set_text('⟲' if Config.PERTURB_TORQUE > 0 else '⟳')
        else: text_torque.set_alpha(0.0)
        
        if is_force or is_torque:
            txt_info.set_text("STATUS: !! PERTURBATION !!"); txt_info.set_color('#c0392b')
        else:
            txt_info.set_text("STATUS: STABLE HOVER"); txt_info.set_color('#27ae60')
            
        txt_time.set_text(f"T: {times[idx]:.4f}s")
        return [patch_thorax, patch_head, patch_abd, traj_line, wing_traj_line, arrow_force, text_torque, txt_time, txt_info] + wing_lines

    ani = animation.FuncAnimation(fig, update, frames=len(r_states), interval=800/Config.FPS, blit=True)
    out_name = "hornet_flight_inference_pro.gif"
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