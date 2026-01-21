import os
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pickle
from typing import NamedTuple, Optional

# Force JAX to CPU
# os.environ["JAX_PLATFORMS"] = "cpu"

class SurrogateState(NamedTuple):
    """
    Represents the physical state of the fluid surrogate.
    
    Attributes:
        s_pos: [N, 2] Current positions of the discretization points.
        s_vel: [N, 2] Current velocities of the discretization points.
        s_vel_prev: [N, 2] Velocity at previous step (used for acceleration calculation).
        marker_le: [2] Leading Edge marker coordinates.
    """
    s_pos: jnp.ndarray
    s_vel: jnp.ndarray
    s_vel_prev: jnp.ndarray
    marker_le: jnp.ndarray

# ---------------------------------------------------------------------------
# Haiku Neural Network Architecture
# ---------------------------------------------------------------------------

class ResNetBlock(hk.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

    def __call__(self, x):
        shortcut = x
        x = hk.Conv1D(output_channels=self.channels, kernel_shape=self.kernel_size, padding='SAME')(x)
        x = jax.nn.gelu(x)
        x = hk.Conv1D(output_channels=self.channels, kernel_shape=self.kernel_size, padding='SAME')(x)
        return jax.nn.gelu(x + shortcut)

class FluidSurrogateResNet(hk.Module):
    """
    1D ResNet surrogate model for predicting aerodynamic forces based on 
    kinematic state (position, velocity, acceleration).
    """
    def __init__(self, n_points, hidden_dim=64):
        super().__init__()
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        
    def __call__(self, x):
        batch_size = x.shape[0]
        n2 = self.n_points * 2
        
        # Unpack flattened inputs
        pos_flat = x[:, :n2]
        vel_flat = x[:, n2 : n2*2]
        acc_flat = x[:, n2*2 :] 
        
        pos = jnp.reshape(pos_flat, (batch_size, self.n_points, 2))
        vel = jnp.reshape(vel_flat, (batch_size, self.n_points, 2))
        acc = jnp.reshape(acc_flat, (batch_size, self.n_points, 2))
        
        h = jnp.concatenate([pos, vel, acc], axis=-1)
        h = hk.Conv1D(output_channels=self.hidden_dim, kernel_shape=3, padding='SAME')(h)
        h = jax.nn.gelu(h)
        for _ in range(3):
            h = ResNetBlock(channels=self.hidden_dim, kernel_size=3)(h)
        pred_forces = hk.Conv1D(output_channels=2, kernel_shape=3, padding='SAME')(h)
        return jnp.reshape(pred_forces, (batch_size, -1))

# ---------------------------------------------------------------------------
# Rigid JAX Surrogate Engine
# ---------------------------------------------------------------------------

class JaxSurrogateEngine:
    def __init__(self, model_path='fluid.pkl', target_freq=115.0, sim_dt=3e-5):
        """
        Initializes the surrogate engine with automatic time-step calibration.
        
        Args:
            model_path: Path to the trained model weights.
            target_freq: The expected operating frequency of the robot (e.g. 200 Hz).
            sim_dt: The physics time step of the robot simulation (e.g. 1e-4).
        """
        # Resolve absolute path to the model file for package compatibility
        if model_path == 'fluid.pkl':
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, 'fluid.pkl')

        print(f"--> Initializing ADAPTIVE JAX Surrogate for {target_freq}Hz @ {sim_dt}s dt")
        
        # --- Time Stepping Configuration ---
        self.DT_LBM_ORIG = 3.0e-6
        self.SUBSAMPLE_RATE = 10 
        self.DT_SURR = self.DT_LBM_ORIG * self.SUBSAMPLE_RATE  # 3e-5 seconds (Fixed by training)
        self.TRAIN_FREQ = 20.0 # Base frequency of surrogate training data
        
        # Total physics time per inference step (used for acceleration calculation)
        self.DT_TOTAL = self.DT_SURR 
        
        # --- Automatic Calibration ---
        # 1. Estimate ideal scaling factor (Target Frequency / Training Frequency)
        ideal_scale = target_freq / self.TRAIN_FREQ
        
        # 2. Calculate required substeps (N) to match phase dynamics
        # w_real * dt_real = w_surr * (N * dt_surr)
        ideal_substeps = (ideal_scale * sim_dt) / self.DT_SURR
        
        # 3. Discretize substeps for JAX Scan loop compatibility
        self.SUBSTEPS = int(round(ideal_substeps))
        if self.SUBSTEPS < 1:
            self.SUBSTEPS = 1
            print("WARNING: sim_dt is too small for this surrogate. Forces may be aliased.")
            
        # 4. Recalculate exact time scale based on integer substeps to prevent drift
        self.TIME_SCALE = (self.SUBSTEPS * self.DT_SURR) / sim_dt
        
        print(f"    -> Calibrated Substeps: {self.SUBSTEPS}")
        print(f"    -> Effective Time Scale: {self.TIME_SCALE:.4f}x (Ideal: {ideal_scale:.2f}x)")

        # --- Physics Constants ---
        self.PHYS_SIZE = 0.20
        self.WING_LEN = 0.01
        self.N_PTS = 20
        
        # Normalization constants (Must match training configuration)
        self.NORM_POS = self.PHYS_SIZE 
        self.NORM_VEL = 10.0
        self.NORM_ACC = 1000.0 
        self.NORM_FORCE = 100.0
        
        # --- Model Loading ---
        if not os.path.exists(model_path):
            print("WARNING: Model file not found. Initializing with random params for testing.")
            dummy_params = True
        else:
            dummy_params = False

        def forward_fn(x):
            model = FluidSurrogateResNet(n_points=self.N_PTS, hidden_dim=64)
            return model(x)
        
        self.model_fn = hk.without_apply_rng(hk.transform(forward_fn))

        if dummy_params:
            rng = jax.random.PRNGKey(42)
            dummy_input = jnp.zeros((1, self.N_PTS * 6))
            self.params = self.model_fn.init(rng, dummy_input)
        else:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.params = data['params']
        
        # --- Geometry Setup ---
        indices = jnp.arange(self.N_PTS, dtype=jnp.float32)
        # Local x-coordinates centered at 0. Index 0 = Leading Edge.
        self.x_local_ref = (self.WING_LEN / 2.0) - (indices / (self.N_PTS - 1)) * self.WING_LEN

    def init_state(self, start_bx, start_bz, start_angle):
        """Initializes the surrogate state based on a rigid pose."""
        c, s = jnp.cos(start_angle), jnp.sin(start_angle)
        
        pos_x = start_bx + self.x_local_ref * c
        pos_y = start_bz + self.x_local_ref * s
        s_pos = jnp.stack([pos_x, pos_y], axis=1)
        
        s_vel = jnp.zeros_like(s_pos)
        
        # Identify Leading Edge marker (Index 0)
        marker_le = s_pos[0] 
        
        return SurrogateState(s_pos, s_vel, s_vel, marker_le)

    def predict_aero_forces(self, params, state: SurrogateState):
        """Runs the ResNet inference to calculate aerodynamic forces."""
        # 1. Compute Acceleration (Backward Difference)
        accels = (state.s_vel - state.s_vel_prev) / self.DT_TOTAL
        
        # 2. Normalize inputs
        flat_pts = state.s_pos.flatten() / self.NORM_POS
        flat_vels = state.s_vel.flatten() / self.NORM_VEL
        flat_accs = accels.flatten() / self.NORM_ACC
        
        # 3. Network Inference
        input_vec = jnp.concatenate([flat_pts, flat_vels, flat_accs])
        input_batch = input_vec[None, :] # [1, D]
        
        pred_y = self.model_fn.apply(params, input_batch)
        pred_forces_flat = pred_y[0]
        
        # 4. Denormalize outputs
        forces = pred_forces_flat.reshape((self.N_PTS, 2)) / self.NORM_FORCE
        return forces

    def _step_core(self, params, fluid_state: SurrogateState, wing_pose, wing_vel_surr, stroke_plane_angle):
        """
        Internal Core Step: Calculates forces for a single rigid step in the local frame.
        
        The process involves:
        1. Transforming Global kinematics -> Local frame (horizontal sliding wing).
        2. Computing rigid body dynamics and running model inference.
        3. Transforming Forces -> Global frame.
        """
        # --- 1. Frame Transformation (Global -> Horizontal Local) ---
        rot_angle = -stroke_plane_angle
        c_r, s_r = jnp.cos(rot_angle), jnp.sin(rot_angle)
        RM = jnp.array([[c_r, -s_r], [s_r, c_r]])
        
        # Calculate linear stroke position in the local frame.
        # pivot_vec_global represents Vector(Stroke Center -> Wing Hinge).
        pivot_vec_global = wing_pose[:2]
        pivot_vec_local = jnp.dot(RM, pivot_vec_global)
        
        bx_loc = pivot_vec_local[0] 
        bz_loc = pivot_vec_local[1] 
        
        # Transform previous state velocity to local frame (required for Accel calc)
        s_vel_prev_loc = jnp.dot(fluid_state.s_vel, RM.T)
        
        # Transform kinematics to local frame
        bang_loc = wing_pose[2] + rot_angle
        v_lin_loc = jnp.dot(wing_vel_surr[:2], RM.T)
        vbx_loc, vbz_loc, vang_loc = v_lin_loc[0], v_lin_loc[1], wing_vel_surr[2]

        # --- 2. Compute Rigid State (Local Frame) ---
        c, s = jnp.cos(bang_loc), jnp.sin(bang_loc)
        
        # A. Rigid Positions: Place wing in grid using local hinge coordinates
        gx = bx_loc + self.x_local_ref * c
        gy = bz_loc + self.x_local_ref * s
        current_pos_loc = jnp.stack([gx, gy], axis=1) 
        
        # B. Rigid Velocities: v = v_trans + omega x r
        rx = gx - bx_loc
        ry = gy - bz_loc
        
        vgx = vbx_loc - vang_loc * ry
        vgy = vbz_loc + vang_loc * rx
        current_vel_loc = jnp.stack([vgx, vgy], axis=1) 

        state_loc = SurrogateState(current_pos_loc, current_vel_loc, s_vel_prev_loc, jnp.zeros(2))

        # --- 3. Predict Forces ---
        # Apply scaling factor consistent with training domain
        aero_forces_loc = self.predict_aero_forces(params, state_loc) * 4.0

        # --- 4. Frame Transformation (Local -> Global) ---
        # Transform positions and velocities back to global
        final_pos_glob = jnp.dot(current_pos_loc, RM)
        final_vel_glob = jnp.dot(current_vel_loc, RM)
        
        # Transform forces back to global
        f_nodal_glob = jnp.dot(aero_forces_loc, RM)
        f_wing_si = jnp.sum(f_nodal_glob, axis=0)
        
        # Pack next state
        next_state = SurrogateState(
            s_pos=final_pos_glob,
            s_vel=final_vel_glob,
            s_vel_prev=final_vel_glob,
            marker_le=final_pos_glob[0] 
        )
        
        return next_state, f_nodal_glob, f_wing_si

    def step(self, params, fluid_state: SurrogateState, _unused_struct, wing_pose_start, wing_vel_real, dt=None, stroke_plane_angle=0.0):
        """
        Public Interface: Advances the surrogate model by one simulation step.
        
        Handles super-sampling (substepping) and time dilation to match the robot's
        simulation frequency with the surrogate's training frequency.

        Args:
            wing_pose_start: Pose at current time t (Real World).
            wing_vel_real: Velocity at current time t (Real World, ~200Hz).
            dt: Real world time step (Simulation dt).
        """
        
        # 1. Scale velocity DOWN for Surrogate (Time Dilation)
        # Uses the exact calibrated time scale to prevent phase drift.
        wing_vel_surrogate = wing_vel_real / self.TIME_SCALE
        
        # Calculate real world seconds per substep
        dt_sub = dt / self.SUBSTEPS
        
        # 2. Define the Scan Loop for substepping
        def scan_fn(carry, i):
            curr_state = carry
            
            # Linearly extrapolate Pose for this substep: pose(t) = pose_start + vel * t
            # Note: We use the REAL velocity for position updates to maintain correct world geometry.
            time_offset = i * dt_sub
            current_pose = wing_pose_start + wing_vel_real * time_offset
            
            next_state, f_nodal, f_wing = self._step_core(
                params, 
                curr_state, 
                current_pose, 
                wing_vel_surrogate,  # Pass time-dilated velocity
                stroke_plane_angle
            )
            
            # Scale forces UP to real-world magnitude (Force is proportional to Velocity^2)
            f_wing_real  = f_wing
            f_nodal_real = f_nodal
            
            return next_state, (f_wing_real, f_nodal_real)

        # 3. Execute Substepping Loop
        indices = jnp.arange(self.SUBSTEPS)
        final_state, (batch_forces, batch_nodal) = jax.lax.scan(scan_fn, fluid_state, indices)
        
        # 4. Average Outputs over substeps
        f_wing_avg = jnp.mean(batch_forces, axis=0)
        f_nodal_avg = jnp.mean(batch_nodal, axis=0)
        
        # Fast-forward state to full t + dt for consistent visualization.
        # Velocities are constant in this rigid surrogate, but positions need exact update.
        final_pose_exact = wing_pose_start + wing_vel_real * dt
    
        # Re-run rigid transform logic for marker alignment (no force calculation needed)
        final_state_exact, _, _ = self._step_core(
            params, 
            final_state,  # Use the state from the loop
            final_pose_exact, 
            wing_vel_surrogate, 
            stroke_plane_angle
        )

        return final_state_exact, f_nodal_avg, f_wing_avg