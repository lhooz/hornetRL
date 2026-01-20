import jax
import jax.numpy as jnp
import numpy as np
from .environment_surrogate import JaxSurrogateEngine

class FlyRobotPhysics:
    def __init__(self):
        # --- SCALED DIMENSIONS (Total Length ~ 3cm) ---
        self.L_thorax = 0.012            
        self.L_abdomen = 0.018        
        
        # --- MASS (Scaled for Hornet: Total ~1.5g) ---
        # WAS: 0.0030 / 0.0010 (Total 4g)
        self.m_thorax = 0.0011   # 1.1 grams     
        self.m_abdomen = 0.0004  # 0.4 grams        
        
        # --- INERTIA (Scaled down with mass) ---
        # WAS: 8e-7 / 4e-7
        self.I_thorax = 2.0e-8              
        self.I_abdomen = 1.5e-8        
        
        self.d1 = self.L_thorax / 2.0 
        self.d2 = self.L_abdomen / 2.0
        
        self.hinge_offset_x = 0.003   
        self.hinge_offset_z = 0.003   
        self.stroke_plane_angle = -1.5707 

        self.g = 9.81
        # 1. Linear: Reduced to ~5% of weight at 1m/s.
        self.damping_linear = 8e-4          
        # 2. Angular: CRITICAL FIX. Reduced by 40x.
        # 5e-5 (Allows the 6e-5 muscle torque to actually turn the body)
        self.damping_angular = 1e-5

        # --- NEW: Passive Abdomen Stiffness ---
        # --- JOINT PROPERTIES ---
        # 1. Natural Stiffness (The Tendon)
        self.k_hinge = 5e-5   
        self.b_hinge = 5e-6     
        self.phi_equilibrium = 0.3 

        # 2. JOINT LIMITS (Asymmetric)
        # ---------------------------------------------------
        # Positive Phi = Curling DOWN (Stinging) -> Large Range (~80 deg)
        self.limit_down = 1.4  
        
        # Negative Phi = Curling UP (Towards Wings) -> Small Range (~35 deg)
        self.limit_up = -0.6
        
        # Wall Stiffness: Needs to be much stiffer than k_hinge
        # 0.01 is ~100x stiffer than the tendon.
        self.k_wall = 0.01
        
        # Wall Damping: Absorbs energy when hitting the limit (Crash)
        # Prevents bouncing.
        self.b_wall = 5e-4
        # ---------------------------------------------------

    def get_mass_matrix(self, q):
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
            state: [x, z, theta, phi, ...]
            wing_angles: [Stroke, Deviation, Pitch] (Radians/Meters)
            wing_rates:  [dStroke/dt, dDev/dt, dPitch/dt] (Velocities)
        """
        q = state[:4]  
        p = state[4:]  
        
        x, z, theta, phi = q
        M = self.get_mass_matrix(q)
        v = jnp.linalg.solve(M, p)
        vx, vz, w_theta, w_phi = v
        
        # --- 1. UNPACK CPG INPUTS ---
        # The CPG has already done the hard work of shaping (Triangles, Flipping).
        # We just apply the geometry here.
        local_stroke = wing_angles[0] # Phi (Stroke angle)
        local_dev    = wing_angles[1] # Theta (Deviation angle)
        local_pitch  = wing_angles[2] # Alpha (Pitch angle)
        
        d_stroke = wing_rates[0]
        d_dev    = wing_rates[1]
        d_pitch  = wing_rates[2]
        
        # --- 2. CALCULATE WING POSE ---
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        
        off_x = self.hinge_offset_x * c_th - self.hinge_offset_z * s_th
        off_z = self.hinge_offset_x * s_th + self.hinge_offset_z * c_th
        hinge_x, hinge_z = x + off_x, z + off_z
        
        global_st_ang = theta + self.stroke_plane_angle
        c_st, s_st = jnp.cos(global_st_ang), jnp.sin(global_st_ang)
        c_dev, s_dev = -s_st, c_st 
        
        # Wing Slice Position (Center of Lift)
        slice_x = hinge_x + (local_stroke * c_st) + (local_dev * c_dev)
        slice_z = hinge_z + (local_stroke * s_st) + (local_dev * s_dev)
        
        # Wing Angle (Global)
        wing_ang = global_st_ang + local_pitch + 1.5707
        wing_pose = jnp.array([slice_x, slice_z, wing_ang])
        
        # --- 3. CALCULATE WING VELOCITY ---
        # Local Velocities (from CPG rates)
        # Note: We assume local_stroke is an arc-length approximation or angle * Radius.
        # If local_stroke is in radians, we multiply by Radius here? 
        # Based on previous code: "local_stroke = -stroke_amp * sin". This implies METERS (Linear).
        # So d_stroke is linear velocity (m/s).
        
        v_flap_x = d_stroke * c_st + d_dev * c_dev
        v_flap_z = d_stroke * s_st + d_dev * s_dev
        
        r_tip_x = slice_x - x
        r_tip_z = slice_z - z
        
        # Rigid Body Rotation Velocity (Tangential)
        v_rot_x = -w_theta * r_tip_z  
        v_rot_z =  w_theta * r_tip_x  
        
        v_total_x = vx + v_rot_x + v_flap_x
        v_total_z = vz + v_rot_z + v_flap_z
        
        # Total Rotational Velocity (Body + Pitching)
        w_total = w_theta + d_pitch
        
        wing_vel = jnp.array([v_total_x, v_total_z, w_total], dtype=jnp.float32)
        
        return wing_pose, wing_vel

    def potential_energy(self, q):
        x, z, th, ph = q
        pe1 = self.m_thorax * self.g * z
        az = z - self.d1 * jnp.sin(th) - self.d2 * jnp.sin(th + ph)
        pe2 = self.m_abdomen * self.g * az
        # --- NEW: Passive Spring Energy ---
        # E = 1/2 * k * (phi - rest_angle)^2
        pe_spring = 0.5 * self.k_hinge * (ph - self.phi_equilibrium)**2

        # 1. Violation Down (Stinging too far)
        viol_down = jnp.maximum(0.0, ph - self.limit_down)
        
        # 2. Violation Up (Hitting wings)
        # Note: ph is negative here, so we check if it's LESS than limit_up
        viol_up = jnp.maximum(0.0, self.limit_up - ph)
        
        # Total Wall Energy
        pe_wall = 0.5 * self.k_wall * (viol_down**2 + viol_up**2)

        return pe1 + pe2 + pe_spring + pe_wall

    def hamiltonian(self, state):
        q, p = state[:4], state[4:]
        M = self.get_mass_matrix(q)
        v = jnp.linalg.solve(M, p)
        T, V = 0.5 * jnp.dot(p, v), self.potential_energy(q)
        return T + V

    def dynamics_step(self, state, u_controls, dt=1e-4):
        state = state.astype(jnp.float32)
        dH = jax.grad(self.hamiltonian)(state)
        u_vec = jnp.concatenate([jnp.zeros(4), u_controls])
        
        J = jnp.block([[jnp.zeros((4,4)), jnp.eye(4)], [-jnp.eye(4), jnp.zeros((4,4))]])
        
        # [NEW] 2. Dynamic Damping Matrix
        # We need extra damping ONLY when we hit the wall to stop the bounce.
        q = state[:4]
        phi = q[3]
        
        # Check limits (Asymmetric)
        is_past_down = phi > self.limit_down
        is_past_up   = phi < self.limit_up
        is_at_limit  = is_past_down | is_past_up
        
        # Increase hinge friction if at limit
        b_hinge_eff = self.b_hinge + jnp.where(is_at_limit, self.b_wall, 0.0)

        # [UPDATED] Damping Matrix R
        # Index 4,5: Linear Damping (vx, vz)
        # Index 6:   Body Angular Damping (w_theta) -> Air Resistance
        # Index 7:   Abdomen Joint Damping (w_phi)  -> Hinge Friction (b_hinge)
        R = jnp.diag(jnp.array([
            0, 0, 0, 0, 
            self.damping_linear, 
            self.damping_linear, 
            self.damping_angular, 
            b_hinge_eff          # <--- Now uses dynamic damping
        ], dtype=jnp.float32))
        
        x_dot = (J - R) @ dH + u_vec
        return state + x_dot * dt

class FlappingFlySystem:
    def __init__(self, model_path='fluid.pkl', target_freq=115.0, sim_dt=3e-5):
        # --- FIX: Pass simulation parameters to the surrogate ---
        self.fluid = JaxSurrogateEngine(
            model_path=model_path, 
            target_freq=target_freq, 
            sim_dt=sim_dt
        )
        self.robot = FlyRobotPhysics()
        
    def step(self, params, full_state, action_data, t, dt):
        """
        Differentiable Step Function.
        """
        robot_state_v, fluid_state = full_state
        
        # Unpack Action Data (Expect stroke_bias now)
        wing_angles, wing_rates, abd_torque, stroke_bias = action_data
        
        # --- 1. BRIDGE: Velocity (External) -> Momentum (Internal) ---
        q = robot_state_v[:4] # [x, z, theta, phi]
        v = robot_state_v[4:]
        M = self.robot.get_mass_matrix(q)
        p = M @ v  # Momentum
        
        robot_state_p = jnp.concatenate([q, p])
        
        # --- 2. Kinematics ---
        # Get TRUE Global Wing Pose (includes Body + Hinge + Bias + Oscillation)
        wing_pose_global, wing_vel_global = self.robot.get_kinematics(robot_state_p, wing_angles, wing_rates)
        
        # ==================================================================
        # 3. CALCULATE STROKE CENTER OFFSET (Body CoM -> Stroke Center)
        # ==================================================================
        # We need the vector from the Body CoM to the "Zero Point" of the stroke.
        theta = q[2]
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        
        # A. Hinge Offset (Rotated by Body)
        h_x = self.robot.hinge_offset_x
        h_z = self.robot.hinge_offset_z
        
        # Vector from CoM to Hinge (Global Frame)
        hinge_glob_x = h_x * c_th - h_z * s_th
        hinge_glob_z = h_x * s_th + h_z * c_th
        
        # --- NEW: Compute Global Hinge Position for Visualization ---
        body_pos_x = q[0]
        body_pos_z = q[1]
        hinge_marker_x = body_pos_x + hinge_glob_x
        hinge_marker_z = body_pos_z + hinge_glob_z
        hinge_marker = jnp.array([hinge_marker_x, hinge_marker_z])
        # ------------------------------------------------------------
        
        # B. Bias Offset (Translated along Stroke Plane)
        # The bias shifts the center of oscillation along the stroke plane.
        total_stroke_angle = theta + self.robot.stroke_plane_angle
        c_st, s_st = jnp.cos(total_stroke_angle), jnp.sin(total_stroke_angle)
        
        # Vector from Hinge to Stroke Center (Global Frame)
        bias_glob_x = stroke_bias * c_st
        bias_glob_z = stroke_bias * s_st
        
        # C. Total Offset Vector (Body CoM -> Instantaneous Stroke Center)
        offset_vec_x = hinge_glob_x + bias_glob_x
        offset_vec_z = hinge_glob_z + bias_glob_z
        
        # ==================================================================
        # 4. CENTER THE WING POSE FOR SURROGATE
        # ==================================================================
        # We define a "Virtual Inference Frame" centered at the Stroke Center.
        # This removes Body Movement and Stroke Center Deviation.
        
        # Pose relative to the Stroke Center
        
        # (Wing_Global) - (Body_Global + Offset_Vector)
        pose_centered_x = wing_pose_global[0] - (body_pos_x + offset_vec_x)
        pose_centered_z = wing_pose_global[1] - (body_pos_z + offset_vec_z)
        
        # Construct Centered Pose (Angle remains global, handled by RM in surrogate)
        wing_pose_centered = jnp.array([pose_centered_x, pose_centered_z, wing_pose_global[2]])

        # ==================================================================
        # 5. FLUID STEP (Inference)
        # ==================================================================
        # Pass the CENTERED pose. 
        # The surrogate will return positions (s_pos) relative to the Stroke Center.
        
        fluid_next, f_nodal_glob, f_wing_si = self.fluid.step(
            params, 
            fluid_state, 
            None,   # _unused_struct
            wing_pose_centered,  # <--- CHANGED: Use Centered Pose
            wing_vel_global,     # Velocity is invariant (keep global for drag calc)
            dt, 
            stroke_plane_angle=total_stroke_angle
        )

        # ==================================================================
        # 6. ACCURATE TORQUE CALCULATION (Restore Lever Arm)
        # ==================================================================
        # fluid_next.s_pos contains points centered at the Stroke Center (0,0).
        # We need the lever arm relative to the Body CoM.
        # Lever_Arm = (Point_Local + Stroke_Center) - Body_CoM
        #           = Point_Local + (Body_CoM + Offset_Vector) - Body_CoM
        #           = Point_Local + Offset_Vector
        
        r_vecs_x = fluid_next.s_pos[:, 0] + offset_vec_x
        r_vecs_z = fluid_next.s_pos[:, 1] + offset_vec_z
        
        # 2D Cross Product: rx * Fy - ry * Fx
        node_torques = r_vecs_x * f_nodal_glob[:, 1] - r_vecs_z * f_nodal_glob[:, 0]
        tau_wing = jnp.sum(node_torques)
        
        # --- 7. Generalized Forces ---
        u_aero = jnp.array([f_wing_si[0], f_wing_si[1], tau_wing, 0.0]) 
        
        # --- 8. Total Actuation (Aero + CPG Abdomen Torque) ---
        # FIX: Internal torque appears only on the relative coordinate (phi)
        u_internal = jnp.array([0.0, 0.0, 0.0, abd_torque]) 
        
        u_total = u_aero + u_internal
        
        # --- 9. Robot Dynamics ---
        robot_next_p = self.robot.dynamics_step(robot_state_p, u_total, dt)
        
        # --- 10. BRIDGE: Momentum -> Velocity (MOVED UP) ---
        # We need q_next to calculate the correct marker position for the new frame
        q_next = robot_next_p[:4]
        p_next = robot_next_p[4:]
        M_next = self.robot.get_mass_matrix(q_next)
        v_next = jnp.linalg.solve(M_next, p_next)
        
        robot_next_v = jnp.concatenate([q_next, v_next])

        # ==================================================================
        # 11. FIX: TRANSFORM MARKER TO GLOBAL FRAME (Corrected)
        # ==================================================================
        # The surrogate simulation occurred in the frame defined at time 't'.
        # Therefore, fluid_next.marker_le is relative to the Stroke Center at time 't'.
        # To get the Global position, we must add the Stroke Center position at time 't'.
        
        # CORRECT: Use (body_pos_x, body_pos_z) and (offset_vec_x, offset_vec_z) from STEP 3/4
        # Do NOT use q_next or new offsets here.
        global_le_x = fluid_next.marker_le[0] + offset_vec_x + body_pos_x
        global_le_y = fluid_next.marker_le[1] + offset_vec_z + body_pos_z
        
        global_le = jnp.array([global_le_x, global_le_y])
        
        # Update fluid state with the corrected global marker
        fluid_next = fluid_next._replace(marker_le=global_le)
        
        # --- 12. Next Kinematics (Estimate) ---
        # FIX: Also predict the wing pose at t+dt so it matches the marker
        wing_pose_next = wing_pose_global + wing_vel_global * dt
        
        return (robot_next_v, fluid_next), f_wing_si, f_nodal_glob, wing_pose_next, hinge_marker