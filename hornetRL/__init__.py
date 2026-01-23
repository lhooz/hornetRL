# hornetRL/__init__.py

# Core Physics & Surrogate
from .fluid_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem, PhysParams

# Neural Control & CPG
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

# Environment (New)
from .env import FlyEnv

# Population Based Training (New)
from .pbt_manager import init_pbt_state, pbt_evolve, PBTHyperparams

__version__ = "0.1.1"