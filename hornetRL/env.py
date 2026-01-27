import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

# --- USER MODULES ---
# Ensure these are importable from the same directory or package
from .fly_system import FlappingFlySystem, PhysParams
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import unpack_action

# ==============================================================================
# ROBUST ENVIRONMENT
# ==============================================================================
class FlyEnv:
    """
    JAX-based environment handling the coupled Rigid Body and Fluid Surrogate dynamics.
    Implements SHAC-compatible step functions with automatic differentiation support.
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration class or object containing constants like 
                    BASE_FREQ, TARGET_STATE, DT, etc.
        """
        # Store config to access constants like DT, WARMUP_STEPS, etc.
        self.cfg = config
        
        self.phys = FlappingFlySystem(
            model_path='fluid.pkl', 
            target_freq=config.BASE_FREQ 
        )
        self.target = config.TARGET_STATE

    def reset(self, key, batch_size):
        """
        Resets the environment state.
        
        Strategy:
            Uses a mixed initialization curriculum (80% Nominal Hover, 20% Chaotic Perturbation)
            to robustify the policy against disturbances.
        """
        k1, k2, k3, k4, k_shuffle = jax.random.split(key, 5)
        
        # =========================================================
        # 1. INIT ROBOT STATE (Curriculum Strategy)
        # =========================================================
        ratio = getattr(self.cfg, 'CURRICULUM_RATIO', 0.8) # Default to 0.8 if missing
        
        n_nominal = int(batch_size * ratio)
        n_chaos = batch_size - n_nominal
        
        # --- A. Nominal Group (Stable Hover Conditions) ---
        k1_n, k2_n = jax.random.split(k1)
        
        # Position: Tight window (+/- 5cm)
        q_pos_nom = jax.random.uniform(k1_n, (n_nominal, 2), minval=-0.05, maxval=0.05)
        
        # Angle Setup:
        # 1. Pitch: Upright (~1.08 rad)
        theta_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        theta_nom = theta_nom + 1.0
        
        # 2. Abdomen: Equilibrium (~0.3 rad) to minimize initial internal stress
        phi_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        phi_nom = phi_nom + 0.2
        
        q_ang_nom = jnp.concatenate([theta_nom, phi_nom], axis=-1)
        
        # --- B. Chaos Group (Recovery Training) ---
        k1_c, k2_c = jax.random.split(k2)
        k_theta, k_phi = jax.random.split(k2_c)
        
        # Position: Wide window (+/- 30cm)
        q_pos_chaos = jax.random.uniform(k1_c, (n_chaos, 2), minval=-0.15, maxval=0.15)
        
        # Pitch: Increased randomization range
        theta_chaos = jax.random.uniform(k_theta, (n_chaos, 1), minval=-0.5, maxval=0.5)
        theta_chaos = theta_chaos + 1.0
        
        # Abdomen: Increased randomization range; limits (-0.6 to 1.4)
        phi_chaos = jax.random.uniform(k_phi, (n_chaos, 1), minval=-0.3, maxval=0.3)
        phi_chaos = phi_chaos + 0.2
        
        q_ang_chaos = jnp.concatenate([theta_chaos, phi_chaos], axis=-1)
        
        # --- C. Combine & Velocity ---
        # This creates the ordered list [Easy, ..., Easy, Hard, ..., Hard]
        q_pos_ordered = jnp.concatenate([q_pos_nom, q_pos_chaos], axis=0)
        q_ang_ordered = jnp.concatenate([q_ang_nom, q_ang_chaos], axis=0)
        v_ordered     = jnp.zeros((batch_size, 4))
        
        # --- SHUFFLE THE BATCH ---
        # This ensures the "Chaos" condition rotates randomly among agents
        perm = jax.random.permutation(k_shuffle, batch_size)
        
        q_pos = q_pos_ordered[perm]
        q_ang = q_ang_ordered[perm]
        v     = v_ordered[perm] # (Though v is all zeros, good practice to keep aligned)
        # ---------------------------------------
        
        # State Vector: [Batch, 8]
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # =========================================================
        # 2. INIT OSCILLATOR (Random Phase)
        # =========================================================
        osc_state_single = OscillatorState.init(base_freq=self.cfg.BASE_FREQ) 
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

        # =========================================================
        # 4. DOMAIN RANDOMIZATION (Physics Parameters)
        # =========================================================
        # Split keys for different physical properties
        k_mass, k_com, k_hinge, k_st, k_joint = jax.random.split(k3, 5)
        
        # A. Mass & Inertia Scaling (+/- 20%)
        # We vary thorax and abdomen independently to prevent the agent from
        # memorizing a specific mass ratio.
        mass_scale_th = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)
        mass_scale_ab = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)

        # B. Center of Mass Shifts (Body Geometry)
        # Shift CoM forward/back by +/- 2mm. This drastically changes the
        # pitch stability and forces active control.
        off_x_th = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)
        off_x_ab = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)

        # C. Hinge Location Noise (Manufacturing tolerance)
        # Shift hinge point by +/- 1mm
        h_x_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)
        h_z_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)

        # D. Stroke Plane Angle Noise
        # Tilt stroke plane by +/- 5 degrees (~0.08 rad)
        st_ang_noise = jax.random.uniform(k_st, (batch_size,), minval=-0.08, maxval=0.08)

        # E. Joint Stiffness/Damping (Tendon properties)
        # Variance in tissue elasticity (+/- 50%)
        k_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
        b_hinge_scale = jax.random.uniform(k_joint, (batch_size,), minval=0.5, maxval=1.5)
        
        # Equilibrium Angle Noise (+/- 10 degrees)
        # Simulates different "resting" postures for the abdomen
        phi_eq_off = jax.random.uniform(k_joint, (batch_size,), minval=-0.17, maxval=0.17)

        # Pack into PhysParams NamedTuple (Vectorized)
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

        # We need to map compute_props over the batch of params
        active_props = jax.vmap(self.phys.robot.compute_props)(phys_params)

        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_p_dummy, k_angles, k_rates, active_props)
        
        # --- Center Pose Transformation ---
        # The surrogate model inference assumes the wing is at (0,0).
        # We subtract (Body Pos + Hinge Offset + Bias Offset) to move to the local inference frame.
        
        def get_centered_pose(r_state, w_pose_glob, bias_val, props):
            q = r_state[:4]
            theta = q[2]
            
            # A. Global Hinge Offset (Rotated by Body)
            h_x = props.hinge_offset_x
            h_z = props.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            # B. Global Bias Offset (Rotated by Stroke Plane)
            total_st_ang = theta + props.stroke_plane_angle
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

        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias, active_props)

        # =========================================================
        # 5. INIT FLUID STATE
        # =========================================================
        def init_fluid_fn(wp):
            return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
            
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        return (robot_state_v, fluid_state, osc_state, active_props)

    def step_batch(self, full_state, action_mods, step_idx=100):
        """
        Advances the simulation by one control step (Config.SIM_SUBSTEPS physics steps).
        Includes warmup ramping and velocity clamping for numerical stability.
        """
        robot_st, fluid_st, osc_st, active_props = full_state
        
        # Define single agent step function for vmap/scan
        def single_agent_step(r, f, o, props, a):
            
            # --- Sub-stepping Loop (Physics Integration) ---
            def sub_step_fn(carry, _):
                curr_r, curr_f, curr_o = carry
                
                # 1. Oscillator Update (Steps by DT)
                o_next = step_oscillator(curr_o, unpack_action(a), self.cfg.DT)
                k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
                
                action_data = (k_angles, k_rates, tau_abd, bias)

                # 2. Physics Update (Rigid Body + Fluid)
                (r_next_v, f_next), f_wing, f_nodal, wing_pose, hinge_marker = self.phys.step(
                    self.phys.fluid.params, (curr_r, curr_f), action_data, props, 0.0, self.cfg.DT
                )
                
                # --- Warmup Ramp & Stability ---
                # Applies a linear ramp to forces during the first WARMUP_STEPS
                ramp = jnp.clip(step_idx / self.cfg.WARMUP_STEPS, 0.0, 1.0)
                
                # 1. Velocity Reset: Pin fly during warmup
                v_reset = jnp.zeros(4) 
                r_next_v = jnp.where(step_idx < self.cfg.WARMUP_STEPS, r_next_v.at[4:].set(v_reset), r_next_v)
                
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
                sub_step_fn, init_carry, None, length=self.cfg.SIM_SUBSTEPS
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
        r_n, f_n, o_n, f_act, f_nodal_b, w_pose_b, h_marker_b = jax.vmap(single_agent_step_remat)(robot_st, fluid_st, osc_st, active_props, action_mods)
        
        return (r_n, f_n, o_n, active_props), f_act, f_nodal_b, w_pose_b, h_marker_b

    def get_reward_metrics(self, robot_state, u_forces, reward_weights):
        """
        Calculates the scalar reward and detailed cost breakdown.
        Prioritizes position holding while allowing necessary body inclination for movement.
        """
        err = robot_state - self.target
        
        # Wrap Thorax Angle error to [-pi, pi]
        err_theta = jnp.mod(err[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi

        # 1. Squared Errors
        loss_pos = jnp.sqrt(jnp.sum(err[:, :2]**2, axis=1) + 1e-6)   
        loss_ang_thorax = err_theta**2              
        loss_ang_abdomen = err[:, 3]**2
        loss_lin_vel = jnp.sum(err[:, 4:6]**2, axis=1)
        loss_ang_vel = jnp.sum(err[:, 6:8]**2, axis=1)
        loss_eff = jnp.sum(u_forces**2, axis=1)        
        
        # 2. Weighted Cost Function (Agility Tuned)
        # --- DYNAMIC COST CALCULATION ---
        # Extract weights for the batch (Shape: Batch x 6)
        w_pos = reward_weights[:, 0]
        w_th  = reward_weights[:, 1]
        w_ab  = reward_weights[:, 2]
        w_lv  = reward_weights[:, 3]
        w_av  = reward_weights[:, 4]
        w_eff = reward_weights[:, 5]

        cost = (w_pos * loss_pos + 
                w_th  * loss_ang_thorax + 
                w_ab  * loss_ang_abdomen + 
                w_lv  * loss_lin_vel + 
                w_av  * loss_ang_vel + 
                w_eff * loss_eff)
        
        # 3. Soft Fence Constraint
        dist_from_center = loss_pos
        out_of_bounds_cost = jnp.where(dist_from_center > 0.20, 100.0, 0.0)
        cost = cost + out_of_bounds_cost 

        # 4. Proximity Bonuses
        is_close = loss_pos < 0.05 # ~5cm
        bonus = is_close * 5.0
        is_close2 = loss_pos < 0.02 # ~2cm
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
