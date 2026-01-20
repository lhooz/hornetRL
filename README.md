# HornetRL: Bio-Inspired Flapping Flight Control

**HornetRL** is a JAX-based simulation and control framework for **robophysical hornet models**. It combines differentiable physics with modern reinforcement learning to achieve stable flight for biomimetic robotic insects.

Key features include:
* **Short-Horizon Actor-Critic (SHAC):** Efficient reinforcement learning utilizing differentiable simulation gradients.
* **High-Fidelity Physics:** High-frequency unsteady aerodynamics via a differentiable surrogate model.
* **Bio-Inspired Actuation:** A Central Pattern Generator (CPG) based muscle model.
* **Structured Control:** A Neural IDA-PBC policy architecture that guarantees physical consistency.

### 🎓 Try it now
Run the full training demo in your browser with zero setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhooz/hornetRL/blob/main/notebooks/demo_train.ipynb)

## 📂 Project Structure

```text
hornetRL_repo/                <-- Repository Root
├── hornetRL/                 <-- Main Package
│   ├── environment_surrogate.py  # JAX surrogate for unsteady aerodynamics
│   ├── fly_system.py             # Rigid body dynamics & kinematics
│   ├── neural_cpg.py             # Oscillator & Muscle mapping
│   ├── neural_idapbc.py          # Neural IDA-PBC Policy
│   ├── train.py                  # Training loop (PPO/SHAC)
│   ├── inference_hornet.py       # Visualization & Inference
│   └── fluid.pkl                 # Pre-trained fluid dynamics data
├── notebooks/                <-- Demo Notebooks
│   └── demo_train.ipynb          # Colab-ready training script
├── pyproject.toml
└── README.md

```

---

## 🚀 Installation

### 1. Local Installation

Clone the repository and install it in editable mode:

```bash
git clone [https://github.com/lhooz/hornetRL.git](https://github.com/lhooz/hornetRL.git)
cd hornetRL
pip install -e .

```

### 2. Google Colab Installation

You can install directly from GitHub inside a Colab notebook:

```python
!pip install git+[https://github.com/lhooz/hornetRL.git](https://github.com/lhooz/hornetRL.git)

```

---

## 🏋️‍♂️ Training

The training script uses a differentiable physics pipeline to train the neural controller.

**Run locally (CPU):**

```bash
python -m hornetRL.train

```

**Run on GPU (e.g., Google Colab):**

```bash
python -m hornetRL.train --gpu

```

**Save checkpoints to a specific folder (e.g., Google Drive):**

```bash
python -m hornetRL.train --gpu --dir "/content/drive/MyDrive/Hornet_Experiments"

```

> **Note:** Checkpoints are saved as `shac_params_{STEP}.pkl`.

---

## 🎬 Inference & Visualization

To visualize a trained policy, run the inference module. This will generate a `.gif` of the flight.

```bash
python -m hornetRL.inference_hornet --checkpoint checkpoints_shac/shac_params_1000.pkl

```

If you do not provide a checkpoint, it will attempt to find the latest one in the default `checkpoints_shac/` folder.

---

## ⚙️ Configuration

Key simulation parameters can be found in `hornetRL/train.py` (for training hyperparameters) and `hornetRL/fly_system.py` (for physical properties like mass and inertia).

* **Base Frequency:** 115 Hz
* **Simulation DT:** 3e-5 s
* **Control Rate:** ~1666 Hz (Every 20 physics steps)

## 📦 Dependencies

* [JAX](https://github.com/google/jax)
* [DM-Haiku](https://github.com/deepmind/dm-haiku)
* [Optax](https://github.com/deepmind/optax)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)