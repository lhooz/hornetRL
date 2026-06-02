import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Tuple

# =============================================================================
# 0. Biological Constraints
# =============================================================================
class BioConfig:
    """
    Defines the biological operating limits for the fly's wing kinematics.
    These constraints ensure the controller remains within the physical 
    capabilities of the muscles and aerodynamics.
    """
    # --- Frequency Limits (Hz) ---
    FREQ_MIN = 90.0
    FREQ_MAX = 140.0
    
    # --- Amplitude Limits (rad/m) ---
    # 0.015m corresponds to roughly 1.5cm stroke amplitude.
    AMP_MIN = 0.012
    AMP_MAX = 0.018
    
    # --- Time Constants (s) ---
    # Controls the muscle response time (Low Pass Filter).
    # A tau of 0.005s (5ms) corresponds to ~0.6 wingbeats, ensuring reflex stability.
    TAU_KINEMATICS = 0.005

# =============================================================================
# 1. THE STATE (The "Clock")
# =============================================================================
class OscillatorState(NamedTuple):
    """
    Tracks the internal rhythmic and kinematic state of the fly.
    Includes parameters for temporal shaping to ensure smooth transitions between states.
    
    Attributes:
        phase: Current angle in the cycle [0, 2pi].
        freq: Current flapping frequency (Hz).
        amp: Current stroke amplitude (meters).
        bias: Current stroke center offset (meters).
        pitch_off: Phase shift for pitch rotation relative to stroke reversal.
        dev_amp: Amplitude of the figure-8 deviation path.
        dev_phase: Phase shift for deviation, effectively tilting the figure-8.
        aoa_dn: Target Angle of Attack during downstroke (forward).
        aoa_up: Target Angle of Attack during upstroke (backward).
    """
    phase: jnp.float32      
    freq: jnp.float32       
    amp: jnp.float32        
    bias: jnp.float32       
    
    pitch_off: jnp.float32  
    dev_amp: jnp.float32    
    dev_phase: jnp.float32  
    aoa_dn: jnp.float32     
    aoa_up: jnp.float32     
    
    @classmethod
    def init(cls, base_freq=115.0):
        """Initializes the oscillator in a stable hover configuration."""
        return cls(
            phase=jnp.float32(0.0),
            freq=jnp.float32(base_freq), 
            amp=jnp.float32(0.015),      
            bias=jnp.float32(0.0),
            
            # Initial defaults for stability
            pitch_off=jnp.float32(0.0),
            dev_amp=jnp.float32(0.001),
            dev_phase=jnp.float32(0.0), 
            aoa_dn=jnp.float32(0.75),
            aoa_up=jnp.float32(0.75)
        )

# =============================================================================
# 2. NEURAL KINEMATIC MAP (The "Conductor")
# =============================================================================
class BiologicalKinematicMap(hk.Module):
    """
    Neural network module mapping high-level control inputs to specific 
    kinematic modulation targets (frequency, amplitude, bias, etc.).
    """
    def __init__(self):
        super().__init__()
        self.network = hk.Sequential([
            hk.Linear(32), 
            jax.nn.tanh,
            hk.Linear(32),
            jax.nn.tanh,
            hk.Linear(9)  # Outputs 9 distinct modulation parameters
        ])

    def __call__(self, u_control):
        raw = self.network(u_control)
        
        # Ellipsis (...) handles both single inputs (8,) and batches (N, 8)
        
        # 1. Frequency Modulation (Hz/s)
        # Scaled to allow rapid adjustments within ~5 wingbeats.
        d_freq = raw[..., 0] * 1000.0 
        
        # 2. Amplitude Modulation (m/s)
        # Scaled for agile response within ~2 wingbeats.
        d_amp = raw[..., 1] * 0.4
        
        # 3. Stroke Bias (Target Position in METERS)
        # Shifts the center of the stroke fore/aft.
        bias_target = jnp.clip(raw[..., 2] * 0.0035, -0.0035, 0.0035)
        
        # 4. Pitch Phase Offset
        # Shifts the timing of wing rotation.
        pitch_phase_offset = jnp.clip(0.0 + raw[..., 3] * 0.5, -0.5, 0.5)
        
        # 5. Deviation Amplitude
        # Controls the width of the figure-8 path.
        dev_amp_target = jnp.clip(raw[..., 4] * 0.006, -0.006, 0.006)
        
        # 6. Abdomen Torque
        # Scaled to be biologically realistic (~3.2x Gravity).
        abd_torque = raw[..., 5] * 2e-4
        
        # 7. Angle of Attack (Downstroke)
        aoa_down = jnp.clip(0.75 + raw[..., 6] * 0.75, 0.0, 1.5)
        
        # 8. Angle of Attack (Upstroke)
        aoa_up = jnp.clip(0.75 + raw[..., 7] * 0.75, 0.0, 1.5)
        
        # 9. Deviation Phase Offset
        # Shifts the figure-8 timing, effectively tilting the deviation plane.
        # Range: +/- 0.5 radians (~30 degrees).
        dev_phase_target = jnp.clip(raw[..., 8] * 0.5, -0.5, 0.5)
        
        return (d_freq, d_amp, bias_target, pitch_phase_offset, 
                dev_amp_target, abd_torque, aoa_down, aoa_up, 
                dev_phase_target)

# =============================================================================
# 3. OSCILLATOR DYNAMICS (The "Integrator")
# =============================================================================
def step_oscillator(state: OscillatorState, modulations, dt) -> OscillatorState:
    """
    Advances the internal state of the CPG.
    Integrates phase/frequency and applies Low Pass Filtering to kinematic targets
    to ensure smooth, biologically plausible transitions.
    """
    # 1. Unpack modulations from the Neural Kinematic Map
    d_f, d_a, bias_t, p_target, dev_t, _, dn_t, up_t, dev_phase_t = modulations
    
    # 2. Integrate Frequency and Amplitude
    new_freq = jnp.clip(state.freq + d_f * dt, BioConfig.FREQ_MIN, BioConfig.FREQ_MAX)
    new_amp  = jnp.clip(state.amp  + d_a  * dt, BioConfig.AMP_MIN,  BioConfig.AMP_MAX)
    
    # 3. Apply Smoothing Filters (First-order LPF)
    alpha = dt / (BioConfig.TAU_KINEMATICS + dt) 
    
    new_bias      = state.bias      + alpha * (bias_t       - state.bias)
    new_pitch_off = state.pitch_off + alpha * (p_target     - state.pitch_off)
    new_dev_amp   = state.dev_amp   + alpha * (dev_t        - state.dev_amp)
    new_aoa_dn    = state.aoa_dn    + alpha * (dn_t         - state.aoa_dn)
    new_aoa_up    = state.aoa_up    + alpha * (up_t         - state.aoa_up)
    
    # Smooth the Deviation Phase
    new_dev_phase = state.dev_phase + alpha * (dev_phase_t - state.dev_phase)
    
    # 4. Integrate Phase (The Clock)
    d_phase = 2.0 * jnp.pi * new_freq * dt
    new_phase = jnp.mod(state.phase + d_phase, 2.0 * jnp.pi)
    
    return OscillatorState(
        phase=new_phase, 
        freq=new_freq, 
        amp=new_amp, 
        bias=new_bias,
        pitch_off=new_pitch_off,
        dev_amp=new_dev_amp,
        dev_phase=new_dev_phase,
        aoa_dn=new_aoa_dn,
        aoa_up=new_aoa_up
    )

# =============================================================================
# 4. KINEMATIC SHAPING (The "Biology")
# =============================================================================
def get_wing_kinematics(state: OscillatorState, modulations) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.float32]:
    """
    Constructs the instantaneous wing state (Angles and Velocities).

    Computes wing velocities using closed-form analytical partial derivatives
    instead of jacfwd. This avoids mixed forward-over-reverse AD (jacfwd inside
    value_and_grad) which causes shape leakage in JAX 0.10: the 3-output
    Jacobian dimension propagates into downstream arrays, causing broadcast
    failures such as (3,2,20) vs (20,2) in the fluid surrogate.
    """

    # --- 1. Unpack Modulations ---
    d_freq, d_amp, bias_target, p_target, dev_amp_target, abd_torque, _, _, dev_phase_target = modulations

    # --- 2. Calculate Safe Integration Rates ---
    f_low, f_high = BioConfig.FREQ_MIN, BioConfig.FREQ_MAX
    freq_mask_low  = jax.nn.sigmoid(3.0 * (state.freq - (f_low + 4.0)))
    freq_mask_high = jax.nn.sigmoid(3.0 * ((f_high - 4.0) - state.freq))
    d_freq_safe = d_freq * freq_mask_low * freq_mask_high  # noqa: F841

    a_low, a_high = BioConfig.AMP_MIN, BioConfig.AMP_MAX
    amp_mask_low  = jax.nn.sigmoid(2000.0 * (state.amp - (a_low + 0.0005)))
    amp_mask_high = jax.nn.sigmoid(2000.0 * ((a_high - 0.0005) - state.amp))
    d_amp_safe = d_amp * amp_mask_low * amp_mask_high

    # --- 3. LPF State Derivatives ---
    d_bias_dt      = (bias_target      - state.bias)      / BioConfig.TAU_KINEMATICS
    d_dev_amp_dt   = (dev_amp_target   - state.dev_amp)   / BioConfig.TAU_KINEMATICS
    d_dev_phase_dt = (dev_phase_target - state.dev_phase) / BioConfig.TAU_KINEMATICS
    d_pitch_off_dt = (p_target         - state.pitch_off) / BioConfig.TAU_KINEMATICS
    d_phi_dt       = 2.0 * jnp.pi * state.freq
    d_amp_dt       = d_amp_safe

    # --- 4. Kinematic Angles (same as original kinematic_map) ---
    phi        = state.phase
    amp        = state.amp
    bias       = state.bias
    dev_amp    = state.dev_amp
    dev_phase  = state.dev_phase
    pitch_off  = state.pitch_off

    stroke_val = amp * jnp.sin(phi) + bias

    phi_off    = phi + pitch_off                          # phi + pitch_off
    svs        = jnp.cos(phi_off)                         # stroke_vel_sign
    switch     = 0.5 * (1.0 + jnp.tanh(3.0 * svs))
    mid_aoa    = state.aoa_up + (state.aoa_dn - state.aoa_up) * switch
    pitch_val  = -mid_aoa * jnp.cos(phi_off)

    phi2       = 2.0 * phi + dev_phase
    dev_val    = -dev_amp * jnp.sin(phi2)

    wing_angles = jnp.stack([stroke_val, dev_val, pitch_val])

    # --- 5. Analytical Velocity: d(angles)/dt ---
    # Stroke: d/dt [amp*sin(phi) + bias]
    #       = amp*cos(phi)*dphi + sin(phi)*damp + dbias
    d_stroke_dt = (amp * jnp.cos(phi)) * d_phi_dt \
                + jnp.sin(phi) * d_amp_dt \
                + d_bias_dt

    # Deviation: d/dt [-dev_amp * sin(2*phi + dev_phase)]
    #          = -dev_amp*cos(phi2)*2*dphi  -  sin(phi2)*d_dev_amp  -  dev_amp*cos(phi2)*d_dev_phase
    d_dev_dt = (-dev_amp * jnp.cos(phi2) * 2.0) * d_phi_dt \
             + (-jnp.sin(phi2)) * d_dev_amp_dt \
             + (-dev_amp * jnp.cos(phi2)) * d_dev_phase_dt

    # Pitch: d/dt [-mid_aoa * cos(phi + pitch_off)]
    # Both phi and pitch_off enter through phi_off = phi + pitch_off, so
    #   d_pitch/d_phi = d_pitch/d_pitch_off  (same expression)
    sech2        = 1.0 - jnp.tanh(3.0 * svs) ** 2
    d_switch_darg = -1.5 * jnp.sin(phi_off) * sech2           # d(switch)/d(phi_off)
    d_mid_aoa_darg = (state.aoa_dn - state.aoa_up) * d_switch_darg
    d_pitch_darg   = -d_mid_aoa_darg * jnp.cos(phi_off) \
                   + mid_aoa * jnp.sin(phi_off)               # d(pitch)/d(phi_off)
    d_pitch_dt = d_pitch_darg * (d_phi_dt + d_pitch_off_dt)

    wing_rates = jnp.stack([d_stroke_dt, d_dev_dt, d_pitch_dt])

    return wing_angles, wing_rates, abd_torque, state.bias