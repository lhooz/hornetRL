import jax
import jax.numpy as jnp
import haiku as hk
from typing import NamedTuple, Tuple
# We need jacfwd to automatically calculate the "slope" of the wing kinematics
from jax import jacfwd

# =============================================================================
# 0. Biological Constraints
# =============================================================================
class BioConfig:
    # --- Frequency Limits (Hz) ---
    FREQ_MIN = 90.0
    FREQ_MAX = 140.0
    
    # --- Amplitude Limits (rad/m) ---
    # Note: 0.015m ~ 1.5cm stroke amplitude
    AMP_MIN = 0.012
    AMP_MAX = 0.018
    
    # --- Time Constants (s) ---
    # Controls how fast the muscles can change shape (Low Pass Filter)
    # We use a time constant (tau) to prevent instantaneous jumps.
    # 0.005s (5ms) 0.6 wingbeats perfect for reflex stability.
    TAU_KINEMATICS = 0.005

# =============================================================================
# 1. THE STATE (The "Clock")
# =============================================================================
class OscillatorState(NamedTuple):
    """
    Tracks the internal rhythmic state of the fly.
    Now includes kinematic shaping parameters to ensure temporal smoothness.
    """
    phase: jnp.float32      # Current angle in cycle [0, 2pi]
    freq: jnp.float32       # Current frequency (Hz)
    amp: jnp.float32        # Current stroke amplitude (meters)
    bias: jnp.float32       # Current stroke center offset (meters)
    
    # --- New Shaping Parameters for Smoothness ---
    pitch_off: jnp.float32  # Pitch phase shift (relative to stroke)
    dev_amp: jnp.float32    # Amplitude of the figure-8 deviation
    dev_phase: jnp.float32  # [NEW] Phase shift for deviation (tilts the figure-8)
    aoa_dn: jnp.float32     # Angle of Attack during downstroke (forward)
    aoa_up: jnp.float32     # Angle of Attack during upstroke (backward)
    
    @classmethod
    def init(cls, base_freq=115.0):
        return cls(
            phase=jnp.float32(0.0),
            freq=jnp.float32(base_freq), 
            amp=jnp.float32(0.015),     
            bias=jnp.float32(0.0),
            
            # Initial defaults for a stable hover
            pitch_off=jnp.float32(0.0),
            dev_amp=jnp.float32(0.001),
            dev_phase=jnp.float32(0.0), # [NEW] Default 0 (Standard Figure-8)
            aoa_dn=jnp.float32(0.8),
            aoa_up=jnp.float32(0.8)
        )

# =============================================================================
# 2. NEURAL KINEMATIC MAP (The "Conductor")
# =============================================================================
class BiologicalKinematicMap(hk.Module):
    def __init__(self):
        super().__init__()
        self.network = hk.Sequential([
            hk.Linear(32), 
            jax.nn.tanh,
            hk.Linear(32),
            jax.nn.tanh,
            hk.Linear(9)  # [CHANGE] Increased from 8 to 9
        ])

    def __call__(self, u_control):
        raw = self.network(u_control)
        
        # Use ellipsis (...) to handle both single inputs (8,) and batches (N, 8)
        
        # 1. Frequency (Hz/s)
        # NEW: 1000.0 (Takes ~5 wingbeats. Snappier.)
        d_freq = raw[..., 0] * 1000.0 
        
        # 2. Amplitude (m/s)
        # NEW: 0.4 (Takes ~2 wingbeats. Agile/Responsive.)
        d_amp = raw[..., 1] * 0.4
        
        # 3. Bias (Target Position in METERS)
        bias_target = jnp.clip(raw[..., 2] * 0.0035, -0.0035, 0.0035)
        
        # 4. Pitch Phase Offset
        pitch_phase_offset = jnp.clip(0.0 + raw[..., 3] * 0.5, -0.5, 0.5)
        
        # 5. Deviation Amplitude
        dev_amp_target = jnp.clip(raw[..., 4] * 0.006, -0.006, 0.006)
        
        # 6. Abdomen Torque
        # OLD: 1e-4 (Superhuman)
        # NEW: 5e-5 (Bio-Faithful ~1.4x Gravity)
        abd_torque = raw[..., 5] * 5e-5
        
        # 7. AoA Down (Forward Stroke Pitch Limit)
        aoa_down = jnp.clip(0.8 + raw[..., 6] * 0.8, 0.0, 1.6)
        
        # 8. AoA Up (Backward Stroke Pitch Limit)
        aoa_up = jnp.clip(0.8 + raw[..., 7] * 0.8, 0.0, 1.6)
        
        # [NEW] 9. Deviation Phase Offset
        # Range: +/- 0.5 radians (~30 degrees)
        dev_phase_target = jnp.clip(raw[..., 8] * 0.5, -0.5, 0.5)
        
        return (d_freq, d_amp, bias_target, pitch_phase_offset, 
                dev_amp_target, abd_torque, aoa_down, aoa_up, 
                dev_phase_target) # [UPDATE] Return 9 items

# =============================================================================
# 3. OSCILLATOR DYNAMICS (The "Integrator")
# =============================================================================
def step_oscillator(state: OscillatorState, modulations, dt) -> OscillatorState:
    # 1. Unpack modulations from the Neural Kinematic Map
    # [UPDATE] Unpack 9 items
    d_f, d_a, bias_t, p_target, dev_t, _, dn_t, up_t, dev_phase_t = modulations
    
    # 2. INTEGRATORS (Frequency, Amplitude)
    new_freq = jnp.clip(state.freq + d_f * dt, BioConfig.FREQ_MIN, BioConfig.FREQ_MAX)
    new_amp  = jnp.clip(state.amp  + d_a  * dt, BioConfig.AMP_MIN,  BioConfig.AMP_MAX)
    
    # 3. SMOOTHING FILTERS (Bias, Shape)
    alpha = dt / (BioConfig.TAU_KINEMATICS + dt) 
    
    new_bias      = state.bias      + alpha * (bias_t      - state.bias)
    new_pitch_off = state.pitch_off + alpha * (p_target    - state.pitch_off)
    new_dev_amp   = state.dev_amp   + alpha * (dev_t       - state.dev_amp)
    new_aoa_dn    = state.aoa_dn    + alpha * (dn_t        - state.aoa_dn)
    new_aoa_up    = state.aoa_up    + alpha * (up_t        - state.aoa_up)
    
    # [NEW] Smooth the Deviation Phase
    new_dev_phase = state.dev_phase + alpha * (dev_phase_t - state.dev_phase)
    
    # 4. Integrate Phase
    d_phase = 2.0 * jnp.pi * new_freq * dt
    new_phase = jnp.mod(state.phase + d_phase, 2.0 * jnp.pi)
    
    return OscillatorState(
        phase=new_phase, 
        freq=new_freq, 
        amp=new_amp, 
        bias=new_bias,
        pitch_off=new_pitch_off,
        dev_amp=new_dev_amp,
        dev_phase=new_dev_phase, # [NEW]
        aoa_dn=new_aoa_dn,
        aoa_up=new_aoa_up
    )

# =============================================================================
# 4. KINEMATIC SHAPING (The "Biology")
# =============================================================================
def get_wing_kinematics(state: OscillatorState, modulations) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.float32]:
    """
    Converts Phase -> Angles AND Rates (Velocities).
    NOW includes:
      1. Boundary-Aware Rate Masking (for Amp/Freq)
      2. Deviation Phase Velocity Contribution [NEW]
      3. Pitch Phase Velocity Contribution [FIXED]
    """
    
    # --- 1. UNPACK MODULATIONS ---
    # [UPDATE] Unpack 9 items (ignore targets here, use state for current pos)
    d_freq, d_amp, bias_target, p_target, dev_amp_target, abd_torque, _, _, dev_phase_target = modulations
    
    # --- 2. CALCULATE RATES FOR INTEGRATORS (Amp, Freq) ---
    f_low, f_high = BioConfig.FREQ_MIN, BioConfig.FREQ_MAX
    freq_mask_low  = jax.nn.sigmoid(3.0 * (state.freq - (f_low + 4.0))) 
    freq_mask_high = jax.nn.sigmoid(3.0 * ((f_high - 4.0) - state.freq))
    d_freq_safe = d_freq * freq_mask_low * freq_mask_high

    a_low, a_high = BioConfig.AMP_MIN, BioConfig.AMP_MAX
    amp_mask_low  = jax.nn.sigmoid(2000.0 * (state.amp - (a_low + 0.0005))) 
    amp_mask_high = jax.nn.sigmoid(2000.0 * ((a_high - 0.0005) - state.amp))
    d_amp_safe = d_amp * amp_mask_low * amp_mask_high
    
    # --- 3. CALCULATE RATES FOR LPF VARIABLES ---
    # Velocity = Distance to Target / Time Constant
    d_bias_dt      = (bias_target      - state.bias)      / BioConfig.TAU_KINEMATICS
    d_dev_amp_dt   = (dev_amp_target   - state.dev_amp)   / BioConfig.TAU_KINEMATICS
    
    # [NEW] Rate for Deviation Phase
    d_dev_phase_dt = (dev_phase_target - state.dev_phase) / BioConfig.TAU_KINEMATICS
    
    # [FIX] Rate for Pitch Phase (Treating it equally now)
    d_pitch_off_dt = (p_target         - state.pitch_off) / BioConfig.TAU_KINEMATICS

    d_phi_dt = 2.0 * jnp.pi * state.freq 
    d_amp_dt = d_amp_safe 
    
    # --- 4. DEFINE EXPLICIT KINEMATIC MAP ---
    # [UPDATE] Function now accepts dev_phase AND pitch_off as arguments
    def kinematic_map(phi, amp, bias, dev_amp, dev_phase, pitch_off):
        # 1. Stroke Angle 
        stroke_val = (amp * jnp.sin(phi)) + bias
        
        # 2. Pitch Angle (Use argument pitch_off, NOT state.pitch_off)
        stroke_vel_sign = jnp.cos(phi + pitch_off)
        switch = 0.5 * (1.0 + jnp.tanh(3.0 * stroke_vel_sign))
        current_mid_aoa = state.aoa_up + (state.aoa_dn - state.aoa_up) * switch
        pitch_val = -current_mid_aoa * jnp.cos(phi + pitch_off)
        
        # 3. Deviation Angle (Function of dev_amp AND dev_phase)
        # [NEW MATH] Add phase shift inside the sine wave
        dev_val = -dev_amp * jnp.sin(2.0 * phi + dev_phase)
        
        return jnp.array([stroke_val, dev_val, pitch_val])

    # --- 5. CALCULATE POSITIONS ---
    # Pass state.dev_phase and state.pitch_off here
    wing_angles = kinematic_map(state.phase, state.amp, state.bias, state.dev_amp, state.dev_phase, state.pitch_off)
    
    # --- 6. CALCULATE VELOCITIES ---
    # [UPDATE] Differentiate w.r.t arg 4 (dev_phase) and arg 5 (pitch_off)
    J = jacfwd(kinematic_map, argnums=(0, 1, 2, 3, 4, 5))(
        state.phase, state.amp, state.bias, state.dev_amp, state.dev_phase, state.pitch_off
    )
    dPos_dPhi, dPos_dAmp, dPos_dBias, dPos_dDev, dPos_dDevPhase, dPos_dPitchOff = J
    
    # [UPDATE] Add deviation phase AND pitch phase velocity terms
    wing_rates = (dPos_dPhi * d_phi_dt) + \
                 (dPos_dAmp * d_amp_dt) + \
                 (dPos_dBias * d_bias_dt) + \
                 (dPos_dDev * d_dev_amp_dt) + \
                 (dPos_dDevPhase * d_dev_phase_dt) + \
                 (dPos_dPitchOff * d_pitch_off_dt)

    return wing_angles, wing_rates, abd_torque, state.bias