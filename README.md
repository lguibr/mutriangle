
[![CI/CD Status](https://github.com/lguibr/mutriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/mutriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/mutriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/mutriangle) - [![PyPI version](https://badge.fury.io/py/mutriangle.svg)](https://badge.fury.io/py/mutriangle) - [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# ▲ MuTriangle ▲

<p align="center">
  <img src="bitmap.png" alt="MuTriangle Logo" width="300"/>
</p>

Welcome to MuTriangle! This project implements an artificial intelligence agent based on the principles of **MuZero** to learn and master a custom triangle puzzle game. The agent learns entirely through **headless self-play reinforcement learning**, guided by a powerful Monte Carlo Tree Search (MCTS) algorithm and a deep neural network built with PyTorch.

**Key Features:**

*   🧠 **MuZero Implementation:** Advanced model-based RL with learned dynamics model
*   🎮 **Custom Game Engine:** Uses the high-performance [`trianglengin>=2.0.7`](https://github.com/lguibr/trianglengin) C++/Python library for game rules and state
*   🌲 **High-Performance MCTS:** Integrates the optimized C++ [`mutrimcts>=1.0.0`](https://github.com/lguibr/mutrimcts) library for efficient MuZero tree search
*   🤖 **Three-Network Architecture:** Representation (h), Dynamics (g), and Prediction (f) networks
*   🔮 **Latent Dynamics:** Learns to predict game outcomes without knowing the game rules
*   🎯 **Distributional RL:** C51 distributional heads for both value and reward prediction
*   🚀 **Parallel Self-Play:** Leverages **Ray** for distributed data generation, automatically scaling workers based on available CPU cores
*   📊 **Asynchronous Monitoring & Persistence:** Uses the [`trieye>=0.1.5`](https://github.com/lguibr/trieye) library via a dedicated Ray actor for robust logging to MLflow/TensorBoard and checkpointing
*   ✨ **Guided & Interactive CLI:** User-friendly command-line interface built with Typer and Rich. Running `mutriangle` without arguments provides a guided experience
*   🔧 **Interactive Configuration Management:** Create, list, show, edit, delete, validate, and set a **default** configuration set easily via the CLI
*   ⚙️ **Headless Operation:** Designed for efficient training on servers or local machines without requiring a graphical display
*   📈 **Optional Profiling:** Supports `cProfile` for performance analysis of self-play workers
*   🎁 **Configuration Presets:** Includes predefined configurations (`toy`, `simple`, `medium`, `large`)
*   📁 **Unified Data Directory:** All generated data stored within `.mutriangle_data/MuTriangle/`, managed by `Trieye`

---

## 🧩 The Triangle Puzzle Game Guide 🎮

This project trains an agent for the game defined by the `trianglengin` library. The objective is to strategically place randomly generated triangular shapes onto a grid, clearing rows to score points and survive as long as possible.

➡️ **Learn the detailed rules and mechanics:** [`trianglengin` Game Guide](https://github.com/lguibr/trianglengin/blob/main/README.md#the-ultimate-triangle-puzzle-guide-)

---

## 💡 Core Concepts: MuZero & MCTS

MuTriangle learns using the MuZero algorithm:

1.  **Self-Play:** The neural network plays games against itself, collecting complete trajectory data
2.  **Search (MCTS):** For each move during self-play, **Monte Carlo Tree Search (MCTS)** explores possible future game states
    *   Uses the **representation network** h(o) to convert observations to hidden states
    *   Uses the **dynamics network** g(s,a) to predict next states and rewards in the latent space
    *   Uses the **prediction network** f(s) to predict policy and value from hidden states
    *   The search operates entirely in the learned latent space after the initial observation
    *   We use the highly optimized C++ `mutrimcts` library for MuZero tree search
3.  **Training:** Complete game trajectories are stored in a replay buffer
    *   The trainer samples positions and unrolls K steps using the learned dynamics
    *   **Initial inference**: h(observation₀) → hidden_state₀, then f(hidden_state₀) → policy₀, value₀
    *   **Recurrent inference**: For k=0..K-1: g(hidden_stateₖ, actionₖ) → hidden_stateₖ₊₁, rewardₖ; then f(hidden_stateₖ₊₁) → policyₖ₊₁, valueₖ₊₁
    *   All three networks train simultaneously with policy, value, and reward losses
    *   Gradient scaling applied to dynamics network for training stability
4.  **Iteration:** The improved network leads to stronger self-play and better MCTS guidance

**Monitoring & Persistence (`trieye`):** A separate asynchronous process (the `TrieyeActor`) collects statistics from training and workers, aggregates them, and logs to MLflow/TensorBoard. It also handles saving/loading training progress within `.mutriangle_data/MuTriangle/`.

---

## 🛠️ Core Technologies

*   **Python 3.10+**
*   **Game:** `trianglengin>=2.0.7`
*   **MCTS:** `mutrimcts>=1.0.0`
*   **Stats/Persistence:** `trieye>=0.1.5`
*   **Deep Learning:** `pytorch>=2.0.0`
*   **Parallelism:** `ray[default]`
*   **Numerical:** `numpy`
*   **CLI:** `typer[all]`, `rich`, `questionary`
*   **Serialization:** `cloudpickle`
*   **Monitoring:** `mlflow`, `tensorboard`
*   **Configuration:** `pydantic`
*   **Optimization:** `numba` (Optional)
*   **Testing:** `pytest`

---

## 📁 Project Structure

```markdown
.
├── .github/workflows/      # GitHub Actions CI/CD
├── .mutriangle_data/       # Unified data directory (GITIGNORED)
│   └── MuTriangle/         # App-specific root directory
│       ├── configs/        # Saved configuration sets (JSON)
│       │   ├── toy.config.json
│       │   ├── simple.config.json
│       │   ├── medium.config.json
│       │   ├── large.config.json
│       │   └── .default_config_marker
│       ├── mlruns/         # MLflow tracking data (Managed by Trieye)
│       └── runs/           # Run-specific data (Managed by Trieye)
│           └── <run_name>/
│               ├── checkpoints/
│               ├── buffers/
│               ├── logs/
│               ├── tensorboard/
│               └── profile_data/
├── mutriangle/             # Source code
│   ├── __init__.py
│   ├── cli.py              # Main CLI
│   ├── config/             # Pydantic configuration models
│   ├── config_mgmt/        # Config management logic
│   ├── features/           # Feature extraction
│   ├── logging_config.py
│   ├── nn/                 # MuZero three-network architecture
│   ├── presets/            # Configuration presets
│   ├── rl/                 # RL components (Trainer, Buffer, Worker)
│   ├── training/           # Training orchestration
│   └── utils/              # Shared utilities and types
├── tests/                  # Unit and integration tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 🧩 Key Modules (`mutriangle`)

*   **`cli`:** Main Typer app with commands (`train`, `create`, `list`, `stats`, `artifacts`, etc.)
*   **`config`:** Pydantic models (ModelConfig, TrainConfig, MuTriangleMCTSConfig, RunContext)
*   **`config_mgmt`:** Interactive configuration creation and storage
*   **`features`:** Converts `GameState` to numerical features for NN input
*   **`nn`:** MuZero three-network architecture (RepresentationNetwork, DynamicsNetwork, PredictionNetwork)
*   **`rl`:** Core RL components (Trainer with unrolled loss, GameHistoryBuffer, SelfPlayWorker)
*   **`training`:** Orchestrates the headless training loop with Ray workers
*   **`utils`:** Helper functions and type definitions

---

## 🚀 Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lguibr/mutriangle.git
    cd mutriangle
    ```
    
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
3.  **Install the package:**
    ```bash
    # For users:
    pip install mutriangle
    
    # For developers (editable install):
    pip install -e .[dev]
    ```
    *Note: Ensure C++ build tools (CMake, compiler) are installed for trianglengin/mutrimcts*

4.  **Add data directory to `.gitignore`:**
    ```
    .mutriangle_data/
    ```

---

## ▶️ Running the Code (CLI)

**1. First Run & Guided Start:**

```bash
mutriangle
```

*   Creates `.mutriangle_data/MuTriangle/configs/` directory
*   Copies preset configurations (`toy`, `simple`, `medium`, `large`)
*   Sets `simple` as default configuration
*   Prompts to run training

**2. Configuration Management:**

```bash
# Create configuration interactively
mutriangle create

# List configurations
mutriangle list

# Show configuration details
mutriangle show simple

# Edit configuration
mutriangle edit simple

# Delete configuration
mutriangle delete toy

# Validate configuration
mutriangle validate medium

# Manage default
mutriangle default           # Show current default
mutriangle default large     # Set 'large' as default
```

**3. Run Training:**

```bash
# Using default configuration
mutriangle train

# Using specific configuration
mutriangle train large --seed 42 --run-name my_muzero_run

# CLI options override loaded config values
mutriangle train simple --seed 123 --profile
```

**4. Monitoring & Analysis:**

```bash
# Launch TensorBoard
mutriangle stats [--host <ip>] [--port <num>]

# Launch MLflow UI
mutriangle artifacts [--host <ip>] [--port <num>]

# Show Ray Dashboard instructions
mutriangle monitor

# Analyze profiling data (if --profile was used)
python scripts/analyze_profiles.py .mutriangle_data/MuTriangle/runs/<run_name>/profile_data/worker_0_ep_*.prof
```

**5. Testing:**

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=mutriangle --cov-report=html

# Run specific test
pytest tests/nn/test_model.py -v
```

---

## ⚙️ Configuration Explained

### Model Configuration (`ModelConfig`)

**MuZero-Specific Parameters:**
*   `HIDDEN_STATE_DIM`: Dimension of latent hidden state (default: 128)
*   `REPRESENTATION_HIDDEN_DIMS`: MLP layers in representation network
*   `DYNAMICS_HIDDEN_DIMS`: MLP layers in dynamics network
*   `PREDICTION_HIDDEN_DIMS`: MLP layers in prediction network
*   `NUM_REWARD_ATOMS`: Atoms for distributional reward prediction (default: 51)
*   `REWARD_MIN/MAX`: Support range for reward distribution

**Architecture Parameters:**
*   `CONV_FILTERS`, `NUM_RESIDUAL_BLOCKS`: CNN/ResNet architecture
*   `USE_TRANSFORMER`, `TRANSFORMER_LAYERS`: Optional Transformer Encoder
*   `NUM_VALUE_ATOMS`, `VALUE_MIN/MAX`: Distributional value head parameters

### Training Configuration (`TrainConfig`)

**MuZero Training:**
*   `UNROLL_STEPS`: Number of dynamics unroll steps (default: 5)
*   `REWARD_LOSS_WEIGHT`: Weight for reward prediction loss
*   `DYNAMICS_GRADIENT_SCALE`: Gradient scaling for dynamics network (default: 0.5)

**General Training:**
*   `MAX_TRAINING_STEPS`: Total optimizer steps (default: 100,000)
*   `NUM_SELF_PLAY_WORKERS`: Parallel workers (default: -1 for auto-detect)
*   `BATCH_SIZE`: Training batch size (default: 256)
*   `BUFFER_CAPACITY`: GameHistory buffer size (default: 250,000)
*   `N_STEP_RETURNS`: N-step bootstrapping (default: 5)
*   `LEARNING_RATE`: Base learning rate (default: 2e-4)
*   `USE_PER`: Prioritized Experience Replay

### MCTS Configuration (`MuTriangleMCTSConfig`)

*   `max_simulations`: MCTS simulations per move (default: 64) - **Critical parameter**
*   `cpuct`: Exploration constant (default: 1.5)
*   `dirichlet_alpha/epsilon`: Root exploration noise
*   `mcts_batch_size`: Batch size for network evaluations in MCTS

---

## 🏗️ MuZero Architecture

### Three Networks

1.  **Representation Network** `h(o) → s`
    *   Input: Raw observation (grid + features)
    *   Output: Hidden state (latent representation)
    *   Architecture: CNN/ResNet + optional Transformer + MLP

2.  **Dynamics Network** `g(s, a) → (s', r)`
    *   Input: Hidden state + action
    *   Output: Next hidden state + reward prediction
    *   Architecture: MLP with two heads

3.  **Prediction Network** `f(s) → (p, v)`
    *   Input: Hidden state
    *   Output: Policy logits + value prediction
    *   Architecture: MLP with two heads

### Training Process

```
Initial Step (k=0):
  s₀ = h(observation₀)
  p₀, v₀ = f(s₀)
  
Unrolled Steps (k=1..K):
  sₖ, rₖ₋₁ = g(sₖ₋₁, aₖ₋₁)
  pₖ, vₖ = f(sₖ)

Total Loss:
  L = Σ[L_policy(pₖ, πₖ) + L_value(vₖ, zₖ) + L_reward(rₖ, uₖ)]
  
Gradient Scaling:
  ∇θ_dynamics ← 0.5 × ∇θ_dynamics
```

---

## 💾 Data Storage

All data stored in `.mutriangle_data/MuTriangle/`:

*   **`configs/`:** Configuration sets (JSON) and default marker
*   **`mlruns/`:** MLflow tracking data (managed by Trieye)
*   **`runs/<run_name>/`:** Run-specific data (managed by Trieye)
    *   `checkpoints/`: Network weights and optimizer state
    *   `buffers/`: Saved GameHistory buffers
    *   `logs/`: Plain text logs
    *   `tensorboard/`: TensorBoard event files
    *   `profile_data/`: cProfile output (if enabled)

---

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=mutriangle --cov-report=term-missing

# Specific module
pytest tests/nn/ -v

# Integration test
pytest tests/integration/test_full_run.py -v
```

---

## 📦 Publishing to PyPI

The CI/CD workflow automatically publishes to PyPI when you push a version tag:

```bash
git tag v2.1.0
git push origin v2.1.0
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Run linters: `ruff check .` and `ruff format .`
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

*   **MuZero Algorithm:** DeepMind (Schrittwieser et al., 2020)
*   **Game Engine:** [`trianglengin`](https://github.com/lguibr/trianglengin)
*   **MCTS Library:** [`mutrimcts`](https://github.com/lguibr/mutrimcts)
*   **Stats Library:** [`trieye`](https://github.com/lguibr/trieye)

---

## 📚 References

*   [MuZero Paper](https://arxiv.org/abs/1911.08265) - Schrittwieser et al., 2020
*   [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Silver et al., 2017
*   [Distributional RL](https://arxiv.org/abs/1707.06887) - C51 Algorithm

---

**Ready to train a MuZero agent? Run `mutriangle` to get started! 🚀**
