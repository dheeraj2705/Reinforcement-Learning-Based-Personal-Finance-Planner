import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def load_training_data(filepath):
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")

def smooth_curve(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')

def plot_training_curves(df, title, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(12, 6))

    if "Episode" in df.columns and "Reward" in df.columns:
        episodes = df["Episode"]
        rewards = df["Reward"]
        plt.plot(episodes, rewards, color="blue", alpha=0.4, label="Raw Reward")
        plt.plot(episodes[:len(smooth_curve(rewards))], smooth_curve(rewards), color="red", label="Smoothed Reward", linewidth=2)
        plt.title(f"{title} - Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{title}_reward_curve.png"))
        plt.close()

    if "Loss" in df.columns:
        plt.figure(figsize=(12, 6))
        losses = df["Loss"]
        plt.plot(df["Episode"], losses, color="orange", alpha=0.6, label="Raw Loss")
        plt.plot(df["Episode"][:len(smooth_curve(losses))], smooth_curve(losses), color="red", label="Smoothed Loss", linewidth=2)
        plt.title(f"{title} - Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{title}_loss_curve.png"))
        plt.close()

    print(f"Plots saved in '{output_folder}' folder for {title}.")

def visualize_training_logs():
    print("Training Visualization Module")
    print("Choose the planner type to visualize:")
    print("1. Goal Planner")
    print("2. Schedule Planner")
    choice = input("Enter choice (1/2): ")

    planner_type = "Goal Planner" if choice == "1" else "Schedule Planner"

    file_path = input(f"Enter {planner_type} training log path (CSV/JSON): ").strip().replace('"', '').replace("'", "")
    if not os.path.exists(file_path):
        print("File not found. Check your path and try again.")
        return

    df = load_training_data(file_path)

    # Check required columns
    required_cols = {"Episode", "Reward"}
    if not required_cols.issubset(df.columns):
        print("Training log missing 'Episode' or 'Reward' columns. Please check your log format.")
        return

    plot_training_curves(df, planner_type)

def main():
    visualize_training_logs()

if __name__ == "__main__":
    main()
