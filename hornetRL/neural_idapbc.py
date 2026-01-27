import jax
import jax.numpy as jnp
import haiku as hk

# Import your new Muscle Map
from .neural_cpg import BiologicalKinematicMap

# ==============================================================================
# 0. SCALING CONFIGURATION (INTERNAL PHYSICS TUNING)
# ==============================================================================
class ScaleConfig:
    """
    Internal scaling parameters for the controller physics.
    Defines the magnitude of control authority and damping relative to physical units.
    
    Attributes:
        CONTROL_SCALE: Gains for the shaping gradients [Fx, Fz, Tau_theta, Tau_phi].
                       Values correspond to ~0.05N linear force and ~5e-5Nm torque.
        DAMPING_BASE:  Minimum damping coefficients to ensure stability.
        DAMPING_SCALE: Dynamic range for the learnable damping injection.
    """
    # Control Authority: [Fx (N), Fz (N), Tau_theta (Nm), Tau_phi (Nm)]
    CONTROL_SCALE = jnp.array([0.05, 0.05, 1.5e-4, 2.0e-4])

    # Damping Baseline: Low drag for efficient flight
    # DAMPING_BASE = jnp.array([0.005, 0.005, 1.0e-4, 5.0e-5])
    DAMPING_BASE = jnp.array([0.005, 0.005, 0.0, 5.0e-5])

    # Damping Range: Allows strong braking (linear) and precise attitude control (angular)
    # DAMPING_SCALE = jnp.array([0.05, 0.05, 1.5e-4, 2.0e-4])
    DAMPING_SCALE = jnp.array([0.05, 0.05, 1.5e-6, 2.0e-4])


# ==============================================================================
# 1. INPUT CONVEX NEURAL NETWORK (ICNN)
# ==============================================================================
class ICNN(hk.Module):
    """
    Input Convex Neural Network (ICNN) implementation.
    
    Used as a learnable potential energy function V(x). The network architecture 
    guarantees convexity with respect to the input by enforcing non-negative 
    weights on the pass-through connections and using convex activation functions.
    """
    def __init__(self, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def __call__(self, x):
        
        
        # Initial transformation (can be negative)
        u = hk.Linear(self.hidden_dim, name="input_layer")(x)
        z = jax.nn.softplus(u) ** 2 
        
        # Convex layers (enforce non-negative weights on z)
        input_injection = hk.Linear(self.hidden_dim, name="hidden_inject")(x)
        
        w_z_init = hk.initializers.VarianceScaling()
        w_z = hk.get_parameter("w_z_1", shape=[self.hidden_dim, self.hidden_dim], init=w_z_init)
        
        # z_{k+1} = sigma(z_k * W_z + x * W_x + b)
        # Note: We take absolute value of W_z to ensure convexity
        z = jax.nn.softplus(jnp.dot(z, jnp.abs(w_z)) + input_injection) ** 2
        
        w_out = hk.get_parameter("w_out", shape=[self.hidden_dim, self.output_dim], init=w_z_init)
        energy = jnp.dot(z, jnp.abs(w_out))
        
        # Add quadratic regularization to ensure strict convexity
        energy = energy + 0.005 * jnp.sum(x**2)
        
        return jnp.squeeze(energy)

# ==============================================================================
# 2. BRAIN: IDA-PBC CONTROLLER
# ==============================================================================
class NeuralIDAPBC_ICNN(hk.Module):
    """
    Neural Interconnection and Damping Assignment - Passivity Based Controller.
    
    This module shapes the closed-loop energy of the system using an ICNN 
    to drive the state error to zero. It computes control forces as:
    u = -grad(V_shaped(error)) - R(x) * velocity
    """
    def __init__(self, target_state, obs_scale):
        super().__init__()
        
        # Normalization Logic:
        # The network input 'x' is normalized by the environment wrapper.
        # To compute a valid error signal, the target state must be normalized 
        # by the SAME scaling factors.
        # Indices 0-3 correspond to positions/angles: [x, z, theta, phi]
        
        raw_target_q = target_state[:4]
        
        # If obs_scale is None, we assume the target is ALREADY scaled (e.g. SymLog).
        if obs_scale is None:
            # No division needed. Trust the input.
            self.target_q = raw_target_q 
        else:
            # Linear scaling mode (Old way)
            scale_q = obs_scale[:4]
            self.target_q = raw_target_q / scale_q

        self.icnn = ICNN()
        
        # Layer Norm for input stability
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
    def __call__(self, x):

        # Split state into Generalized Coordinates (q) and Momentum/Velocity (p)
        # Note: x is already scaled by the environment wrapper
        q = x[..., :4] 
        p = x[..., 4:]
        
        # 1. Energy Shaping (Potential Field)
        # Compute error in the normalized latent space
        error = q - self.target_q
        
        def energy_fn(e): return jnp.sum(self.icnn(e))
        
        # Calculate the gradient of the shaped energy function
        raw_grad = jax.grad(energy_fn)(error)
        
        # Scale the gradient to physical force units (Newtons/Nm)
        grad_Va = raw_grad * ScaleConfig.CONTROL_SCALE
        
        # 2. Damping Injection (Dissipation)
        # Predict dynamic damping gains based on system state
        net_R = hk.Sequential([hk.Linear(32), jax.nn.tanh, hk.Linear(4)])
        raw_damp = net_R(x)
        
        damping_gains = (jax.nn.softplus(raw_damp) * ScaleConfig.DAMPING_SCALE) + ScaleConfig.DAMPING_BASE
        damping_force = -damping_gains * p
        
        # Total Control Law: u = -dV/dq - R * p
        return -grad_Va + damping_force

# ==============================================================================
# 3. FULL POLICY WRAPPER
# ==============================================================================
def policy_network_icnn(x, target_state=None, obs_scale=None):
    """
    Full Policy Pipeline: Brain -> Muscles.
    
    Maps normalized observations to biological modulation parameters.
    
    Args:
        x: Normalized Observation [Batch, 8].
        target_state: Raw Physics Target [8] (passed from Config).
        obs_scale: Scale array used to normalize x [8] (passed from Config).
        
    Returns:
        modulations_vector: The kinematic parameters for the CPG.
        u_forces: The intermediate generalized forces computed by the ICNN.
    """
    # Default hover target if none provided
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    
    # 1. THE BRAIN (Compute Generalized Forces)
    brain = NeuralIDAPBC_ICNN(target_state, obs_scale)
    u_forces = brain(x) 
    
    # 2. THE MUSCLES (Map Forces -> Kinematics)
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces) 
    
    # Stack outputs: [d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_dn, aoa_up, dev_phase]
    modulations_vector = jnp.stack(mod_tuple, axis=-1)
    
    return modulations_vector, u_forces

# ==============================================================================
# 4. HELPER: UNPACK ACTION
# ==============================================================================
def unpack_action(action_vector):
    """
    Utility to decompose the stacked action vector into named components.
    
    Returns:
        Tuple of (d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_dn, aoa_up, dev_phase)
    """
    d_freq = action_vector[..., 0]
    d_amp  = action_vector[..., 1]
    d_bias = action_vector[..., 2]
    pitch_off = action_vector[..., 3]
    dev_amp = action_vector[..., 4]
    abd_tau = action_vector[..., 5]
    aoa_down = action_vector[..., 6]
    aoa_up = action_vector[..., 7]
    dev_phase = action_vector[..., 8] 
    
    return (d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_down, aoa_up, dev_phase)