import argparse
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as plot_import_error:  # pragma: no cover
    plt = None

try:
    from train import train_dqn
except ImportError as e:
    print("Failed to import train_dqn from train.py. Ensure you're running from the project root.")
    raise

def plot_training(episode_rewards: list[float], window: int = 20) -> None:
    if plt is None:
        print("matplotlib is not available; cannot plot. Install matplotlib or run in an environment with GUI support.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Episode Reward")
    
    if len(episode_rewards) >= max(1, window):
        kernel = np.ones(window) / window
        moving_avg = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"Moving Avg ({window})")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DQN training progress")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--target_update", type=int, default=10, help="Target network update frequency")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--horizon", type=int, default=100, help="Planning horizon in months")
    parser.add_argument("--income", type=float, default=10000.0, help="Monthly income for environment reset")
    parser.add_argument("--periods_per_month", type=int, default=2, help="Number of periods per month")

    args = parser.parse_args()

    episode_rewards = train_dqn(
        episodes=args.episodes,
        target_update=args.target_update,
        batch_size=args.batch_size,
        horizon=args.horizon,
        income=args.income,
        periods_per_month=args.periods_per_month,
    )

    if not isinstance(episode_rewards, (list, tuple)) or not len(episode_rewards):
        print("No rewards returned from training; nothing to plot.")
        sys.exit(0)

    plot_training(list(episode_rewards))
