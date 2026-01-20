# hornetRL/__init__.py

from .environment_surrogate import JaxSurrogateEngine
from .fly_system import FlappingFlySystem
from .neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from .neural_idapbc import policy_network_icnn, unpack_action

# Expose the inference and training modules so they can be accessed easily
# (Optional, but helpful)
from . import inference_hornet
from . import train

__version__ = "0.1.0"