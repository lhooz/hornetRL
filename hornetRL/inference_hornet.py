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
    """
    # --- Model Constants (Must match Training) ---
    BASE_FREQ = 115.0  
    
    # Target State: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    TARGET_STATE = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    # --- Simulation Settings ---
    DT = 3e-5             
    SIM_SUBSTEPS = 72    
    
    # --- Inference Settings ---
    DURATION = 1.0       
    FPS = 60             
    DPI = 80            
    
    VIZ_STEP_SKIP = 30
    TRACE_HIST_LEN = 15
    N_SHADOW_WINGS = 5
    
    # --- Randomization Settings ---
    # Set to True to see how the agent handles the random physics you defined
    USE_DOMAIN_RANDOMIZATION = True 
    
    # --- Robustness Testing (Wind Gust) ---
    PERTURBATION = True  
    PERTURB_TIME = 0.02  
    PERTURB_FORCE = jnp.array([0.9, -1.2]) 
    PERTURB_TORQUE = -0.0025

def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(robot_state):
    target_sym = symlog(Config.TARGET_STATE)
    mods, forces = policy_network_icnn(
        robot_state, 
        target_state=target_sym,
    )
    # Critic structure needed for loading weights
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(robot_state)
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 3. INFERENCE ENVIRONMENT (With Domain Randomization)
# ==============================================================================
class InferenceFlyEnv:
    def __init__(self):
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=Config.BASE_FREQ
        )

    def reset(self, key):
        """Initializes the fly with Domain Randomization logic."""
        batch_size = 1
        
        # 1. Init Robot State (Start exactly at Target)
        q_pos = jnp.array([[0.0, 0.0]]) 
        q_ang = jnp.array([[1.08, 0.3]])
        v = jnp.zeros((batch_size, 4))
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # 2. Init Oscillator
        osc_state = OscillatorState.init(base_freq=Config.BASE_FREQ)
        osc_state = jax.tree.map(lambda x: jnp.stack([x]*batch_size), osc_state)

        # =========================================================
        # 3. DOMAIN RANDOMIZATION (Physics Parameters)
        # =========================================================
        if Config.USE_DOMAIN_RANDOMIZATION:
            # We use the provided key to trigger the randomization
            k1, k2, k3, k4, k_shuffle = jax.random.split(key, 5)
            k_mass, k_com, k_hinge, k_st, k_joint = jax.random.split(k3, 5)
            
            # A. Mass & Inertia Scaling (+/- 20%)
            mass_scale_th = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)
            mass_scale_ab = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)

            # B. Center of Mass Shifts
            off_x_th = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)
            off_x_ab = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)

            # C. Hinge Location Noise
            h_x_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)
            h_z_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)

            # D. Stroke Plane Angle Noise
            st_ang_noise = jax.random.uniform(k_st, (batch_size,), minval=-0.08, maxval=0.08)

            # E. Joint Stiffness/Damping
            k_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
            b_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
            
            # Equilibrium Angle Noise
            phi_eq_off = jax.random.uniform(k_joint, (batch_size,), minval=-0.17, maxval=0.17)
        else:
            # Nominal parameters (Ideal Hornet)
            mass_scale_th = jnp.array([1.0])
            mass_scale_ab = jnp.array([1.0])
            off_x_th = jnp.array([0.0])
            off_x_ab = jnp.array([0.0])
            h_x_noise = jnp.array([0.0])
            h_z_noise = jnp.array([0.0])
            st_ang_noise = jnp.array([0.0])
            k_hinge_scale = jnp.array([1.0])
            b_hinge_scale = jnp.array([1.0])
            phi_eq_off = jnp.array([0.0])

        # Pack into PhysParams NamedTuple
        phys_params = PhysParams(
            thorax_mass_scale=mass_scale_th,
            abd_mass_scale=mass_scale_ab,
            thorax_offset_x=off_x_th,
            abd_offset_x=off_x_ab,
            hinge_x_noise=h_x_noise,
            hinge_z_noise=h_z_noise,
            stroke_ang_noise=st_ang_noise,
            k_hinge_scale=k_hinge_scale,
            b_hinge_scale=b_hinge_scale,
            phi_equil_offset=phi_eq_off
        )

        # 4. Compute Derived Properties
        self.active_props = jax.tree.map(lambda x: x[0], jax.vmap(self.phys.robot.compute_props)(phys_params))

        # 5. Initialize Wing Pose
        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles, k_rates = ret[0], ret[1]
        
        robot_state_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)
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

        wing_pose_centered = jax.vmap(get_centered_pose, in_axes=(0, 0, 0, None))(robot_state_v, wing_pose_global, osc_state.bias, self.active_props)
        
        def init_fluid_fn(wp): return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state, phys_params)

    def step(self, full_state, action_mods, external_force=jnp.zeros(2), external_torque=0.0):
        # ... (Step logic matches previous, simply passes phys_params through) ...
        robot_st, fluid_st, osc_st, phys_p = full_state
        
        r = robot_st[0]
        f = jax.tree.map(lambda x: x[0], fluid_st) 
        o = jax.tree.map(lambda x: x[0], osc_st)
        a = action_mods[0]
        # phys_p is already unbatched in this context if we map it, 
        # but let's be consistent and take [0] of the batch
        p_single = jax.tree.map(lambda x: x[0], phys_p)

        def sub_step_fn(carry, _):
            curr_r, curr_f, curr_o = carry
            o_next = step_oscillator(curr_o, unpack_action(a), Config.DT)
            k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
            action_data = (k_angles, k_rates, tau_abd, bias)

            (r_next_v, f_next), _, f_nodal, wing_pose, hinge_marker = self.phys.step(
                self.phys.fluid.params, 
                (curr_r, curr_f), 
                action_data, 
                self.active_props, 
                0.0, 
                Config.DT
            )
            
            # --- Apply Perturbations ---
            # Use RANDOMIZED Mass for F=ma calculation
            total_mass = (self.phys.robot.m_thorax * p_single.thorax_mass_scale + 
                          self.phys.robot.m_abdomen * p_single.abd_mass_scale)
            accel_lin = external_force / total_mass
            r_next_v = r_next_v.at[4:6].add(accel_lin * Config.DT)
            
            inertia = self.phys.robot.I_thorax * p_single.thorax_mass_scale
            accel_ang = external_torque / inertia
            r_next_v = r_next_v.at[6].add(accel_ang * Config.DT)
            
            viz_frame = (r_next_v, wing_pose, f_nodal, f_next.marker_le, hinge_marker)
            return (r_next_v, f_next, o_next), viz_frame

        init_carry = (r, f, o)
        (final_r, final_f, final_o), stacked_viz_frames = jax.lax.scan(
            sub_step_fn, init_carry, None, length=Config.SIM_SUBSTEPS
        )

        r_b = jnp.expand_dims(final_r, 0)
        o_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_o)
        f_b = jax.tree.map(lambda x: jnp.expand_dims(x, 0), final_f)
        
        return (r_b, f_b, o_b, phys_p), stacked_viz_frames

# ==============================================================================
# 4. MAIN SIMULATION LOOP
# ==============================================================================
def run_simulation(params):
    print("--> Initializing Inference Environment...")
    env = InferenceFlyEnv()
    
    # Use a fixed seed for reproducibility, or random for variety
    rng = jax.random.PRNGKey(int(time.time())) 
    
    state = env.reset(rng)
    
    # --- EXTRACT SCALING FACTORS ---
    # We grab the generated physics parameters to pass to the visualizer
    phys_p_batch = state[3]
    # Take index 0
    p_single = jax.tree.map(lambda x: x[0], phys_p_batch)
    
    th_scale = float(p_single.thorax_mass_scale)
    ab_scale = float(p_single.abd_mass_scale)
    print(f"--> Randomization Applied:")
    print(f"    Thorax Mass Scale: {th_scale:.3f}")
    print(f"    Abd Mass Scale:    {ab_scale:.3f}")

    total_control_steps = int(Config.DURATION / (Config.DT * Config.SIM_SUBSTEPS))
    
    # Add meta dict to vis_data
    vis_data = {
        'r': [], 'w': [], 'f': [], 't': [], 
        'p_force': [], 'p_torque': [], 
        'meta': {'th_scale': th_scale, 'ab_scale': ab_scale} # <--- STORE SCALES
    }
    
    print(f"--> Simulating {Config.DURATION}s ({total_control_steps} control steps)...")

    @jax.jit
    def single_inference_step(curr_state, curr_params, ext_f, ext_t):
        r_state = curr_state[0]
        wrapped_theta = jnp.mod(r_state[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_input = r_state.at[:, 2].set(wrapped_theta)
        scaled_input = symlog(obs_input)
        mods, _, _ = ac_model.apply(curr_params, scaled_input)
        next_state, frames = env.step(curr_state, mods, external_force=ext_f, external_torque=ext_t)
        return next_state, frames
    
    t_sim = 0.0
    print("--> Compiling JAX graph...")
    _ = single_inference_step(state, params, jnp.zeros(2), 0.0)
    print("--> Compilation Complete!")

    for i in range(total_control_steps):
        ext_f = jnp.zeros(2)
        ext_t = 0.0
        
        if Config.PERTURBATION:
            if Config.PERTURB_TIME <= t_sim <= Config.PERTURB_TIME + 0.002:
                ext_f = Config.PERTURB_FORCE
                ext_t = Config.PERTURB_TORQUE
        
        state, stacked_frames = single_inference_step(state, params, ext_f, ext_t)
        
        s_r, s_w, s_f, s_le, s_hinge = stacked_frames
        
        global_start_step = i * Config.SIM_SUBSTEPS
        batch_global_steps = np.arange(global_start_step, global_start_step + Config.SIM_SUBSTEPS)
        indices_to_keep = np.where(batch_global_steps % Config.VIZ_STEP_SKIP == 0)[0]
        
        if len(indices_to_keep) > 0:
            vis_data['r'].extend([np.array(s_r[j]) for j in indices_to_keep])
            vis_data['w'].extend([np.array(s_w[j]) for j in indices_to_keep])
            vis_data['f'].extend([np.array(s_f[j]) for j in indices_to_keep])
            
            step_times = [t_sim + (j+1)*Config.DT for j in indices_to_keep]
            vis_data['t'].extend(step_times)
            
            has_force = np.linalg.norm(ext_f) > 0
            has_torque = abs(ext_t) > 0
            vis_data['p_force'].extend([has_force] * len(indices_to_keep))
            vis_data['p_torque'].extend([has_torque] * len(indices_to_keep))
        
        t_sim += (Config.DT * Config.SIM_SUBSTEPS)
        
        if i % 100 == 0:
            print(f"    Progress: {i}/{total_control_steps} blocks | T={t_sim:.3f}s")

    return vis_data, env

# ==============================================================================
# 5. VISUALIZATION ENGINE (With Adaptive Scaling)
# ==============================================================================
def generate_gif(data, env):
    print(f"--> Rendering Professional GIF ({len(data['r'])} frames collected)...")
    
    r_states = data['r']
    w_poses = data['w']
    times = data['t']
    flag_force = data['p_force']
    flag_torque = data['p_torque']
    
    # --- EXTRACT SCALES ---
    # We use Cube Root scaling. If mass x2, Size x1.25.
    # This prevents the fly from looking ridiculously huge/small.
    th_scale_mass = data['meta']['th_scale']
    ab_scale_mass = data['meta']['ab_scale']
    
    th_viz_scale = np.cbrt(th_scale_mass)
    ab_viz_scale = np.cbrt(ab_scale_mass)
    
    # --- PRE-CALCULATION ---
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
    
    # --- TRACE 1: Body Trajectory (CoM) ---
    # STYLE: Dashed line, Lighter Grey. Looks like a "Flight Plan".
    traj_line, = ax.plot([], [], color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.8, label='CoM Path')

    # --- TRACE 2: Wing Tip Trace ---
    # STYLE: Solid line, Darker Grey. Looks like "Motion Blur".
    wing_traj_line, = ax.plot([], [], color='#2c3e50', linestyle='-', linewidth=1.2, alpha=0.9, label='Wing Path')
    
    # --- SCALED Body Parts (Flat Design - No Outlines) ---
    # Removed edgecolor='k' and replaced with edgecolor='none'
    
    patch_thorax = patches.Ellipse((0,0), 
                                   width=0.012 * th_viz_scale, 
                                   height=0.006 * th_viz_scale, 
                                   facecolor='#404040', edgecolor='none', zorder=10)
    
    patch_head = patches.Circle((0,0), 
                                radius=0.0025 * th_viz_scale,
                                facecolor='#b0b0b0', edgecolor='none', zorder=10)
    
    patch_abd = patches.Ellipse((0,0), 
                                width=0.018 * ab_viz_scale, 
                                height=0.008 * ab_viz_scale, 
                                facecolor='#707070', edgecolor='none', alpha=0.9, zorder=9)
    
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)
    
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
        if idx >= len(r_states): return
        
        curr_r = r_states[idx]
        rx, rz = curr_r[0], curr_r[1]
        r_th, r_phi = curr_r[2], curr_r[3]
        
        ax.set_xlim(0.0 - window_size, 0.0 + window_size)
        ax.set_ylim(0.0 - window_size, 0.0 + window_size)
        
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
        
        num_total_wings = len(wing_lines)
        for k in range(num_total_wings):
            offset = (num_total_wings - 1) - k
            hist_idx = max(0, idx - offset)
            w_x, w_z = get_wing_coords(r_states[hist_idx], w_poses[hist_idx])
            wing_lines[k].set_data(w_x, w_z)

        hist_len = 50 
        start_t = max(0, idx - hist_len)
        hist_x = [r[0] for r in r_states[start_t:idx]] 
        hist_z = [r[1] for r in r_states[start_t:idx]]
        traj_line.set_data(hist_x, hist_z)

        start_w = max(0, idx - Config.TRACE_HIST_LEN)
        rel_chunk = wing_rel_history[start_w:idx]
        
        if len(rel_chunk) > 0:
            curr_c, curr_s = np.cos(r_th), np.sin(r_th)
            traj_x = rel_chunk[:, 0] * curr_c - rel_chunk[:, 1] * curr_s
            traj_z = rel_chunk[:, 0] * curr_s + rel_chunk[:, 1] * curr_c
            traj_x += rx
            traj_z += rz
            wing_traj_line.set_data(traj_x, traj_z)
        else:
            wing_traj_line.set_data([], [])

        is_force = flag_force[idx]
        is_torque = flag_torque[idx]
        
        if is_force:
            arrow_force.set_alpha(1.0)
            fx, fz = Config.PERTURB_FORCE
            mag = np.sqrt(fx**2 + fz**2) + 1e-6
            d_x = (fx / mag) * 0.03
            d_z = (fz / mag) * 0.03
            start_x = rx - d_x * 1.5
            start_z = rz - d_z * 1.5
            arrow_force.set_data(x=start_x, y=start_z, dx=d_x, dy=d_z)
        else:
            arrow_force.set_alpha(0.0)
            
        if is_torque:
            text_torque.set_alpha(1.0)
            text_torque.set_position((rx, rz + 0.02)) 
            if Config.PERTURB_TORQUE > 0:
                text_torque.set_text('⟲')
            else:
                text_torque.set_text('⟳')
        else:
            text_torque.set_alpha(0.0)
            
        if is_force or is_torque:
            txt_info.set_text(f"STATUS: !! PERTURBATION !!")
            txt_info.set_color('#c0392b')
        else:
            txt_info.set_text(f"STATUS: STABLE HOVER")
            txt_info.set_color('#27ae60')
            
        txt_time.set_text(f"T: {times[idx]:.4f}s")
        return [patch_thorax, patch_head, patch_abd, traj_line, wing_traj_line, arrow_force, text_torque, txt_time, txt_info] + wing_lines

    num_frames = int(len(r_states))
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=800/Config.FPS, blit=True)
    
    out_name = "hornet_flight_inference.gif"
    print(f"--> Saving to {out_name}...")
    try:
        ani.save(out_name, writer='pillow', fps=Config.FPS)
        print(f"--> Done! File saved to {os.path.abspath(out_name)}")
        print(f"--> File Size: {os.path.getsize(out_name) / 1024:.2f} KB")
    except Exception as e:
        print(f"--> Error during saving: {e}")

    plt.close(fig)

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import glob
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the .pkl checkpoint file")
    args = parser.parse_args()

    param_file = args.checkpoint

    if param_file is None:
        default_dir = "checkpoints_shac"
        if os.path.exists(default_dir):
            files = glob.glob(os.path.join(default_dir, "*.pkl"))
            if files:
                files.sort(key=lambda f: int(re.sub(r'\D', '', f)) if re.search(r'\d', f) else 0)
                param_file = files[-1]
                print(f"--> Auto-detected latest checkpoint: {param_file}")
    
    if param_file is None or not os.path.exists(param_file):
        print(f"Error: Checkpoint file not found. Please provide one using --checkpoint")
        exit(1)
        
    print(f"--> Loading parameters from {param_file}")
    with open(param_file, 'rb') as f:
        data = pickle.load(f)

    raw_params = data['params'] 
    
    if 'pbt_state' in data:
        pbt_state = data['pbt_state']
        best_idx = np.argmax(pbt_state.running_reward)
        print(f"--> PBT Detected. Selecting Best Agent: Index {best_idx}")
        print(f"    Score: {pbt_state.running_reward[best_idx]:.2f}")
        params = jax.tree.map(lambda x: x[best_idx], raw_params)
        
    else:
        print("--> No PBT state found. Using Agent 0.")
        first_leaf = jax.tree_util.tree_leaves(raw_params)[0]
        if len(first_leaf.shape) > 2: 
             params = jax.tree.map(lambda x: x[0], raw_params)
        else:
             params = raw_params 
        
    sim_data, env = run_simulation(params)
    generate_gif(sim_data, env)