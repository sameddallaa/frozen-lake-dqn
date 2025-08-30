# FrozenLake DQN Project

This project implements a Deep Q-Network (DQN) agent to solve the FrozenLake environment from Gymnasium using PyTorch. The agent learns to navigate a grid world to reach a goal while avoiding holes, with options for slippery or non-slippery surfaces.

## Project Structure

The project is organized as follows:

- **demo/**: Contains video recordings of the trained model's performance.
  - Two demo videos showcasing the DQN agent's behavior in the FrozenLake environment.
- **notebooks/**: Includes Jupyter notebooks for training and analysis.
  - A training notebook demonstrating the DQN training process.
- **src/**: Source code for the DQN implementation.
  - `dqn.py`: Defines the DQN neural network architecture.
  - `frozen_lake_dqn.py`: Implements the FrozenLakeDQN class for training and demo.
  - `replay_memory.py`: Implements the replay memory buffer for experience replay.
- **figures/**: Stores training statistics visualizations.
  - Plots of rewards and epsilon values generated during training.
- **models/**: Stores the trained model weights.
  - Contains the saved model file.
- **main.py**: The main script to train the DQN agent and optionally save demo videos, configurable via command-line arguments.
- **requirements.txt**: Lists the Python dependencies required to run the project.

## Prerequisites

- Python 3.8 or higher
- A virtual environment (recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone sameddallaa/frozen-lake-dqn
   cd frozen-lake-dqn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project Locally

The `main.py` script allows you to train the DQN agent and optionally generate demo videos. You can configure training parameters via command-line arguments.

### Training the Model

To train the DQN agent with default parameters:
```bash
python main.py
```

This will train the agent for 1000 episodes and save the model to `models/frozen_lake_dqn.pth` and training plots to `figures/frozen_lake_dqn.png`.

To customize training, use the following command-line arguments:
```bash
python main.py --learning_rate 0.001 --discount_factor 0.99 --sync_rate 100 --replay_memory_size 10000 --batch_size 32 --episodes 1000 --render --is_slippery --save_demo --demo_path demo/
```

- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--discount_factor`: Discount factor for future rewards (default: 0.99)
- `--sync_rate`: Target network sync rate in steps (default: 100)
- `--replay_memory_size`: Size of replay memory (default: 10000)
- `--batch_size`: Batch size for training (default: 32)
- `--episodes`: Number of training episodes (default: 1000)
- `--render`: Enable rendering of the environment during training
- `--is_slippery`: Enable slippery mode for FrozenLake
- `--save_demo`: Save demo videos after training
- `--demo_path`: Path to save demo videos (default: demo/)
- `--device`: Device to run the model on (default: cuda if available, else cpu)

### Generating Demo Videos

To train and save demo videos:
```bash
python main.py --save_demo --demo_path demo/
```

This will train the model and save demo videos to the `demo/` directory.

### Example Command

Train with rendering, slippery mode, and save a demo:
```bash
python main.py --episodes 500 --render --is_slippery --save_demo
```