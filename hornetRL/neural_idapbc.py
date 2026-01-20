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
    (Observation scales are now passed in from the main training config)
    """
    # --- 1. CONTROL FORCE SCALING (THRUST BOOSTED) ---
    # Linear: 0.05 N (~3.3x Gravity). Allows rapid climbing/braking.
    # Angular: 5e-5 Nm. Matches the max muscle torque.
    CONTROL_SCALE = jnp.array([0.05, 0.05, 5e-5, 1e-4])

    # --- 2. DAMPING BASELINE (DRAG REDUCED) ---
    # Linear: 0.005. Lower drag allows faster flight/diving.
    # Angular: 2e-5. Keeps a small stability floor.
    DAMPING_BASE = jnp.array([0.005, 0.005, 5e-5, 2e-5])

    # --- 3. DAMPING DYNAMIC RANGE (BALANCED) ---
    # Linear: 0.05 (Strong Air Brake still available if network wants it)
    # Angular: 5e-5 (Matched to Muscle Torque).
    DAMPING_SCALE = jnp.array([0.05, 0.05, 5e-5, 1e-4])


# ==============================================================================
# 1. ICNN (Unchanged)
# ==============================================================================
class ICNN(hk.Module):
    def __init__(self, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def __call__(self, x):
        u = hk.Linear(self.hidden_dim, name="input_layer")(x)
        z = jax.nn.softplus(u) ** 2 
        
        input_injection = hk.Linear(self.hidden_dim, name="hidden_inject")(x)
        
        w_z_init = hk.initializers.VarianceScaling()
        w_z = hk.get_parameter("w_z_1", shape=[self.hidden_dim, self.hidden_dim], init=w_z_init)
        
        z = jax.nn.softplus(jnp.dot(z, jnp.abs(w_z)) + input_injection) ** 2
        
        w_out = hk.get_parameter("w_out", shape=[self.hidden_dim, self.output_dim], init=w_z_init)
        energy = jnp.dot(z, jnp.abs(w_out))
        
        energy = energy + 0.005 * jnp.sum(x**2)
        
        return jnp.squeeze(energy)

# ==============================================================================
# 2. BRAIN: IDA-PBC CONTROLLER (UPDATED: ACCEPTS OBS_SCALE)
# ==============================================================================
class NeuralIDAPBC_ICNN(hk.Module):
    def __init__(self, target_state, obs_scale):
        super().__init__()
        
        # [CRITICAL LOGIC]
        # The network input 'x' is already normalized by the environment.
        # We must normalize the 'target_state' by the SAME factor so they can be subtracted.
        # Indices 0-3 correspond to [x, z, theta, phi]
        
        raw_target_q = target_state[:4]
        
        # Safety check for missing scale
        if obs_scale is None:
            # Fallback to avoid crash, but this implies unit mismatch
            print("WARNING: No obs_scale passed to NeuralIDAPBC. Assuming 1.0 (Raw Units).")
            scale_q = jnp.ones(4)
        else:
            scale_q = obs_scale[:4]

        # Store the NORMALIZED target
        self.target_q = raw_target_q / scale_q

        self.icnn = ICNN()
        
        # Layer Norm for input stability
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
    def __call__(self, x):
        # [NEW] Apply Normalization to handle unit variance
        x_norm = self.norm(x)

        # x is already scaled by env
        q = x[..., :4] 
        p = x[..., 4:]
        
        # Error calculation in "Normalized Space"
        error = q - self.target_q
        
        def energy_fn(e): return jnp.sum(self.icnn(e))
        
        # 1. Raw Gradient
        raw_grad = jax.grad(energy_fn)(error)
        
        # 2. Control Force (Apply Physics Scaling)
        grad_Va = raw_grad * ScaleConfig.CONTROL_SCALE
        
        # 3. Damping Network (Uses Normalized Input)
        net_R = hk.Sequential([hk.Linear(32), jax.nn.tanh, hk.Linear(4)])
        raw_damp = net_R(x_norm)
        
        damping_gains = (jax.nn.softplus(raw_damp) * ScaleConfig.DAMPING_SCALE) + ScaleConfig.DAMPING_BASE
        damping_force = -damping_gains * p
        
        return -grad_Va + damping_force

# ==============================================================================
# 3. FULL POLICY WRAPPER (UPDATED SIGNATURE)
# ==============================================================================
def policy_network_icnn(x, target_state=None, obs_scale=None):
    """
    Maps State -> Modulations (Action Vector)
    Args:
        x: Normalized Observation [Batch, 8]
        target_state: Raw Physics Target [8] (passed from Config)
        obs_scale: Scale array used to normalize x [8] (passed from Config)
    """
    # Fallback
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.08, 0.3, 0.0, 0.0, 0.0, 0.0])
    
    # 1. THE BRAIN (Pass scale down)
    brain = NeuralIDAPBC_ICNN(target_state, obs_scale)
    u_forces = brain(x) 
    
    # 2. THE MUSCLES
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces) 
    
    # Stack: [d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_dn, aoa_up, dev_phase]
    modulations_vector = jnp.stack(mod_tuple, axis=-1)
    
    return modulations_vector, u_forces

# ==============================================================================
# 4. HELPER: UNPACK ACTION (Unchanged)
# ==============================================================================
def unpack_action(action_vector):
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