import os
import torch
import numpy as np
from env_personal_finance import PersonalFinanceEnv, FinanceConfig
from dqn_agent import DQNAgent

def train_dqn(episodes=200, target_update=10, batch_size=64, horizon=120,
              income=10000.0, periods_per_month=1, save_path="checkpoints/dqn.pth"):

    cfg = FinanceConfig(monthly_income=income, horizon_months=horizon,
                        periods_per_month=periods_per_month)
    env = PersonalFinanceEnv(cfg)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    all_rewards = []

    for ep in range(episodes):
        state = env.reset(monthly_income=income)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.update(batch_size)

        all_rewards.append(total_reward)

        if ep % target_update == 0:
            agent.update_target()

        if (ep + 1) % 20 == 0:
            avg = np.mean(all_rewards[-20:])
            print(f"Episode {ep+1}/{episodes}, Avg Reward (last 20): {avg:.2f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"✅ Training complete. Model saved to {save_path}")
    return all_rewards


if __name__ == "__main__":
    rewards = train_dqn(episodes=200, target_update=10, batch_size=64,
                        horizon=120, income=10000.0, periods_per_month=1)
    print("Final average reward over last 10 episodes:", np.mean(rewards[-10:]))
