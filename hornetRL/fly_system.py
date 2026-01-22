import jax
import jax.numpy as jnp
import numpy as np
from .environment_surrogate import JaxSurrogateEngine

class FlyRobotPhysics:
    """
    Defines the physical properties and rigid-body dynamics of the fly robot.
    
    This class handles:
    1. Mass and inertia properties for the thorax and abdomen.
    2. Joint constraints (hinges, limits, and passive stiffness).
    3. Port-Hamiltonian formulation (J-R formalism) for energy-based control.
    """
    def __init__(self):
        # --- Dimensions (meters) ---
        self.L_thorax = 0.012             
        self.L_abdomen = 0.018        
        
        # --- Mass Properties (kg) ---
        # Scaled for a total mass of ~1.5g (Hornet scale)
        self.m_thorax = 0.0011   # 1.1 grams      
        self.m_abdomen = 0.0004  # 0.4 grams        
        
        # --- Inertia Tensor (kg*m^2) ---
        self.I_thorax = 2.0e-8               
        self.I_abdomen = 1.5e-8        
        
        # Center of Mass distances from the joint
        self.d1 = self.L_thorax / 2.0 
        self.d2 = self.L_abdomen / 2.0
        
        # Wing Hinge Geometry relative to Thorax CoM
        self.hinge_offset_x = 0.003    
        self.hinge_offset_z = 0.003    
        self.stroke_plane_angle = -1.5707 

        self.g = 9.81
        
        # --- Damping Coefficients ---
        # Linear: Reduced to ~5% of weight at 1m/s
        self.damping_linear = 8e-4          
        # Angular: Tuned to allow muscle torque authority
        self.damping_angular = 1e-5

        # --- Abdomen Joint Properties ---
        # Passive stiffness (tendon) and equilibrium angle
        self.k_hinge = 5e-5    
        self.b_hinge = 5e-6      
        self.phi_equilibrium = 0.3 

        # --- Joint Limits (Asymmetric) ---
        # Positive Phi = Curling DOWN (Stinging) -> Large Range (~80 deg)
        self.limit_down = 1.4  
        
        # Negative Phi = Curling UP (Towards Wings) -> Small Range (~35 deg)
        self.limit_up = -0.6
        
        # Wall Stiffness: Hard stop (~100x stiffer than tendon)
        self.k_wall = 0.01
        
        # Wall Damping: Dissipates energy on impact to prevent bouncing
        self.b_wall = 5e-4

    def get_mass_matrix(self, q):
        """Computes the mass matrix M(q) for the coupled 2-body system."""
        m1, m2 = self.m_thorax, self.m_abdomen
        I1, I2 = self.I_thorax, self.I_abdomen
        l1, l2 = self.d1, self.d2
        th, ph = q[2], q[3]
        
        c_th, s_th = jnp.cos(th), jnp.sin(th)
        c_thph, s_thph = jnp.cos(th + ph), jnp.sin(th + ph)
        c_ph = jnp.cos(ph)
        
        M_00 = m1 + m2
        M_11 = m1 + m2
        
        h1, h2 = m2 * l1, m2 * l2
        M_02, M_03 = h1 * s_th + h2 * s_thph, h2 * s_thph
        M_12, M_13 = -h1 * c_th - h2 * c_thph, -h2 * c_thph
        
        I_total_th = I1 + I2 + m2*(l1**2 + l2**2 + 2*l1*l2*c_ph)
        I_total_ph = I2 + m2*(l2**2)
        I_coup = I2 + m2*(l2**2 + l1*l2*c_ph)
        
        row0 = jnp.array([M_00, 0.0,  M_02, M_03])
        row1 = jnp.array([0.0,  M_11, M_12, M_13])
        row2 = jnp.array([M_02, M_12, I_total_th, I_coup])
        row3 = jnp.array([M_03, M_13, I_coup,       I_total_ph])
        
        return jnp.stack([row0, row1, row2, row3])

    def get_kinematics(self, state, wing_angles, wing_rates):
        """
        Calculates Global Wing Pose and Velocity based on CPG Inputs.
        
        Args:
            state: [x, z, theta, phi, p_x, p_z, p_theta, p_phi]
            wing_angles: [Stroke, Deviation, Pitch] (Radians/Meters)
            wing_rates:  [dStroke/dt, dDev/dt, dPitch/dt] (Velocities)
            
        Returns:
            wing_pose: [x, z, angle] in global frame.
            wing_vel:  [v_x, v_z, omega] in global frame.
        """
        q = state[:4]  
        p = state[4:]  
        
        x, z, theta, phi = q
        M = self.get_mass_matrix(q)
        v = jnp.linalg.solve(M, p)
        vx, vz, w_theta, w_phi = v
        
        # --- 1. Process CPG Inputs ---
        local_stroke = wing_angles[0] # Phi (Stroke angle)
        local_dev    = wing_angles[1] # Theta (Deviation angle)
        local_pitch  = wing_angles[2] # Alpha (Pitch angle)
        
        d_stroke = wing_rates[0]
        d_dev    = wing_rates[1]
        d_pitch  = wing_rates[2]
        
        # --- 2. Calculate Wing Pose (Global) ---
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        
        # Transform Hinge Offset to Global Frame
        off_x = self.hinge_offset_x * c_th - self.hinge_offset_z * s_th
        off_z = self.hinge_offset_x * s_th + self.hinge_offset_z * c_th
        hinge_x, hinge_z = x + off_x, z + off_z
        
        # Calculate Stroke Plane orientation
        global_st_ang = theta + self.stroke_plane_angle
        c_st, s_st = jnp.cos(global_st_ang), jnp.sin(global_st_ang)
        c_dev, s_dev = -s_st, c_st 
        
        # Wing Slice Position (Center of Lift)
        slice_x = hinge_x + (local_stroke * c_st) + (local_dev * c_dev)
        slice_z = hinge_z + (local_stroke * s_st) + (local_dev * s_dev)
        
        # Wing Pitch Angle (Global)
        wing_ang = global_st_ang + local_pitch + 1.5707
        wing_pose = jnp.array([slice_x, slice_z, wing_ang])
        
        # --- 3. Calculate Wing Velocity (Global) ---
        # Flapping velocity components
        v_flap_x = d_stroke * c_st + d_dev * c_dev
        v_flap_z = d_stroke * s_st + d_dev * s_dev
        
        r_tip_x = slice_x - x
        r_tip_z = slice_z - z
        
        # Rigid Body Rotation Velocity (Tangential = omega x r)
        v_rot_x = -w_theta * r_tip_z  
        v_rot_z =  w_theta * r_tip_x  
        
        v_total_x = vx + v_rot_x + v_flap_x
        v_total_z = vz + v_rot_z + v_flap_z
        
        # Total Rotational Velocity (Body + Pitching)
        w_total = w_theta + d_pitch
        
        wing_vel = jnp.array([v_total_x, v_total_z, w_total], dtype=jnp.float32)
        
        return wing_pose, wing_vel

    def potential_energy(self, q):
        """Calculates Potential Energy (Gravity + Springs + Joint Limits)."""
        x, z, th, ph = q
        pe1 = self.m_thorax * self.g * z
        az = z - self.d1 * jnp.sin(th) - self.d2 * jnp.sin(th + ph)
        pe2 = self.m_abdomen * self.g * az
        
        # Passive Spring Energy: E = 1/2 * k * (phi - rest)^2
        pe_spring = 0.5 * self.k_hinge * (ph - self.phi_equilibrium)**2

        # --- Wall Constraints (Soft Limits) ---
        # 1. Violation Down (Stinging too far)
        viol_down = jnp.maximum(0.0, ph - self.limit_down)
        
        # 2. Violation Up (Hitting wings)
        viol_up = jnp.maximum(0.0, self.limit_up - ph)
        
        # Total Wall Energy
        pe_wall = 0.5 * self.k_wall * (viol_down**2 + viol_up**2)

        return pe1 + pe2 + pe_spring + pe_wall

    def hamiltonian(self, state):
        """Computes the Hamiltonian H = T + V."""
        q, p = state[:4], state[4:]
        M = self.get_mass_matrix(q)
        v = jnp.linalg.solve(M, p)
        T, V = 0.5 * jnp.dot(p, v), self.potential_energy(q)
        return T + V

    def dynamics_step(self, state, u_controls, dt=1e-4):
        """
        Performs a single integration step using Hamiltonian dynamics with damping.
        Calculates x_dot = (J - R) * grad(H) + u.
        """
        state = state.astype(jnp.float32)
        dH = jax.grad(self.hamiltonian)(state)
        u_vec = jnp.concatenate([jnp.zeros(4), u_controls])
        
        # Symplectic matrix J
        J = jnp.block([[jnp.zeros((4,4)), jnp.eye(4)], [-jnp.eye(4), jnp.zeros((4,4))]])
        
        # --- Dynamic Damping Matrix ---
        # Checks if the abdomen joint is hitting limits to apply 'wall damping'
        q = state[:4]
        phi = q[3]
        
        is_past_down = phi > self.limit_down
        is_past_up   = phi < self.limit_up
        is_at_limit  = is_past_down | is_past_up
        
        # Effective hinge friction (Standard + Impact Damping)
        b_hinge_eff = self.b_hinge + jnp.where(is_at_limit, self.b_wall, 0.0)

        # Damping Matrix R
        # Diag: [0...0, D_linear, D_linear, D_angular, D_hinge]
        R = jnp.diag(jnp.array([
            0, 0, 0, 0, 
            self.damping_linear, 
            self.damping_linear, 
            self.damping_angular, 
            b_hinge_eff          
        ], dtype=jnp.float32))
        
        x_dot = (J - R) @ dH + u_vec
        return state + x_dot * dt

class FlappingFlySystem:
    """
    Couples the robot rigid body physics with the fluid surrogate model.
    """
    def __init__(self, model_path='fluid.pkl', target_freq=115.0, sim_dt=3e-5):
        self.fluid = JaxSurrogateEngine(
            model_path=model_path, 
            target_freq=target_freq, 
            sim_dt=sim_dt
        )
        self.robot = FlyRobotPhysics()
        
    def step(self, params, full_state, action_data, t, dt):
        """
        Differentiable Step Function.
        Advances both the robot dynamics and the fluid environment.
        """
        robot_state_v, fluid_state = full_state
        
        # Unpack Action Data (Includes stroke bias for steering)
        wing_angles, wing_rates, abd_torque, stroke_bias = action_data
        
        # --- 1. State Conversion: Velocity -> Momentum ---
        q = robot_state_v[:4] # [x, z, theta, phi]
        v = robot_state_v[4:]
        M = self.robot.get_mass_matrix(q)
        p = M @ v  # Momentum
        
        robot_state_p = jnp.concatenate([q, p])
        
        # --- 2. Kinematics ---
        # Get true global wing pose (Body Motion + Hinge + Bias + Oscillation)
        wing_pose_global, wing_vel_global = self.robot.get_kinematics(robot_state_p, wing_angles, wing_rates)
        
        # ==================================================================
        # 3. Calculate Stroke Center Offset
        # ==================================================================
        # We compute the vector from Body CoM to the "Zero Point" of the stroke.
        theta = q[2]
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        
        # A. Hinge Offset (Rotated to Global Frame)
        h_x = self.robot.hinge_offset_x
        h_z = self.robot.hinge_offset_z
        
        hinge_glob_x = h_x * c_th - h_z * s_th
        hinge_glob_z = h_x * s_th + h_z * c_th
        
        # Visualization: Global Hinge Position
        body_pos_x = q[0]
        body_pos_z = q[1]
        hinge_marker_x = body_pos_x + hinge_glob_x
        hinge_marker_z = body_pos_z + hinge_glob_z
        hinge_marker = jnp.array([hinge_marker_x, hinge_marker_z])
        
        # B. Bias Offset (Translated along Stroke Plane)
        # The bias shifts the center of oscillation relative to the hinge.
        total_stroke_angle = theta + self.robot.stroke_plane_angle
        c_st, s_st = jnp.cos(total_stroke_angle), jnp.sin(total_stroke_angle)
        
        bias_glob_x = stroke_bias * c_st
        bias_glob_z = stroke_bias * s_st
        
        # C. Total Offset Vector (Body CoM -> Instantaneous Stroke Center)
        offset_vec_x = hinge_glob_x + bias_glob_x
        offset_vec_z = hinge_glob_z + bias_glob_z
        
        # ==================================================================
        # 4. Center Wing Pose for Surrogate (Inference Frame)
        # ==================================================================
        # Define a "Virtual Inference Frame" centered at the Stroke Center.
        # This isolates wing aerodynamics from body translation.
        
        # Pose relative to the Stroke Center:
        # (Wing_Global) - (Body_Global + Offset_Vector)
        pose_centered_x = wing_pose_global[0] - (body_pos_x + offset_vec_x)
        pose_centered_z = wing_pose_global[1] - (body_pos_z + offset_vec_z)
        
        wing_pose_centered = jnp.array([pose_centered_x, pose_centered_z, wing_pose_global[2]])

        # ==================================================================
        # 5. Fluid Surrogate Step
        # ==================================================================
        # The surrogate returns positions (s_pos) relative to the Stroke Center.
        
        fluid_next, f_nodal_glob, f_wing_si = self.fluid.step(
            params, 
            fluid_state, 
            None,   # _unused_struct
            wing_pose_centered,  # Use Centered Pose
            wing_vel_global,     # Velocity remains global for drag calculations
            dt, 
            stroke_plane_angle=total_stroke_angle
        )

        # ==================================================================
        # 6. Torque Calculation (Lever Arm Restoration)
        # ==================================================================
        # fluid_next.s_pos is centered at (0,0). We need torques about Body CoM.
        # Lever_Arm = Point_Local + Offset_Vector
        
        r_vecs_x = fluid_next.s_pos[:, 0] + offset_vec_x
        r_vecs_z = fluid_next.s_pos[:, 1] + offset_vec_z
        
        # 2D Cross Product: rx * Fy - ry * Fx
        node_torques = r_vecs_x * f_nodal_glob[:, 1] - r_vecs_z * f_nodal_glob[:, 0]
        tau_wing = jnp.sum(node_torques)
        
        # --- 7. Generalized Forces ---
        u_aero = jnp.array([f_wing_si[0], f_wing_si[1], tau_wing, 0.0]) 
        
        # --- 8. Total Actuation (Aero + CPG Abdomen Torque) ---
        # Internal torque applies only to the relative coordinate (phi)
        u_internal = jnp.array([0.0, 0.0, 0.0, abd_torque]) 
        
        u_total = u_aero + u_internal
        
        # --- 9. Robot Dynamics Step ---
        robot_next_p = self.robot.dynamics_step(robot_state_p, u_total, dt)
        
        # --- 10. State Conversion: Momentum -> Velocity ---
        # Calculate q_next to update marker positions for the next frame
        q_next = robot_next_p[:4]
        p_next = robot_next_p[4:]
        M_next = self.robot.get_mass_matrix(q_next)
        v_next = jnp.linalg.solve(M_next, p_next)
        
        robot_next_v = jnp.concatenate([q_next, v_next])

        # ==================================================================
        # 11. Marker Transformation (Local -> Global)
        # ==================================================================
        # The surrogate simulation occurred in the frame defined at time 't'.
        # We transform the resulting marker back to Global frame using 't' offsets.
        
        global_le_x = fluid_next.marker_le[0] + offset_vec_x + body_pos_x
        global_le_y = fluid_next.marker_le[1] + offset_vec_z + body_pos_z
        
        global_le = jnp.array([global_le_x, global_le_y])
        
        # Update fluid state with the corrected global marker
        fluid_next = fluid_next._replace(marker_le=global_le)
        
        # --- 12. Kinematics Prediction ---
        # Estimate wing pose at t+dt for consistency
        wing_pose_next = wing_pose_global + wing_vel_global * dt
        
        return (robot_next_v, fluid_next), f_wing_si, f_nodal_glob, wing_pose_next, hinge_marker