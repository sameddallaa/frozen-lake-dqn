import argparse
import torch
from src.frozen_lake_dqn import FrozenLakeDQN

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on FrozenLake environment")
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                       help='Discount factor for future rewards (default: 0.99)')
    parser.add_argument('--sync_rate', type=int, default=100,
                       help='Target network sync rate in steps (default: 100)')
    parser.add_argument('--replay_memory_size', type=int, default=10000,
                       help='Size of replay memory (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--is_slippery', action='store_true',
                       help='Use slippery mode for FrozenLake')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run the model on (default: cuda if available, else cpu)')
    parser.add_argument('--save_demo', action='store_true',
                       help='Save a demo video after training')
    parser.add_argument('--demo_path', type=str, default='demo/',
                       help='Path to save demo videos (default: demo/)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    agent = FrozenLakeDQN(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        sync_rate=args.sync_rate,
        replay_memory_size=args.replay_memory_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"Starting training with {args.episodes} episodes...")
    print(f"Parameters: learning_rate={args.learning_rate}, "
          f"discount_factor={args.discount_factor}, "
          f"sync_rate={args.sync_rate}, "
          f"replay_memory_size={args.replay_memory_size}, "
          f"batch_size={args.batch_size}, "
          f"render={args.render}, "
          f"is_slippery={args.is_slippery}, "
          f"device={args.device}, "
          f"save_demo={args.save_demo}, "
          f"demo_path={args.demo_path}")
    
    agent.train(episodes=args.episodes, 
                render=args.render, 
                is_slippery=args.is_slippery)
    
    print("Training completed.")
    
    if args.save_demo:
        print("Running and saving demo...")
        agent.demo(path=args.demo_path, is_slippery=args.is_slippery)
        print("Demo saved.")

if __name__ == "__main__":
    main()