import numpy as np
import pandas as pd
import torch
from dqn_agent import DQNAgent
from env_personal_finance import PersonalFinanceEnv
import os

def generate_goal_plan(monthly_income: float, monthly_expense: float, months: int, episodes: int = 150):
    state_dim = 3
    action_dim = 3
    env = PersonalFinanceEnv(monthly_income, monthly_expense, months)
    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes} Reward: {total_reward:.2f}")

    final_action = agent.select_action(env.state)
    proportions = final_action / np.sum(final_action)
    spend_pct, save_pct, invest_pct = proportions

    print("\nOptimal Allocation Learned:")
    print(f"Spend: {spend_pct*100:.2f}%  |  Save: {save_pct*100:.2f}%  |  Invest: {invest_pct*100:.2f}%")

    records = []
    balance = 0
    for m in range(1, months + 1):
        expense = np.random.uniform(monthly_expense * 0.9, monthly_expense * 1.1)
        spend = spend_pct * monthly_income
        save = save_pct * (monthly_income - expense)
        invest = invest_pct * (monthly_income - expense)
        balance += save + invest
        records.append({
            "Month": m,
            "Income": monthly_income,
            "ExpenseNeed": expense,
            "SpendAmt": round(spend, 2),
            "SaveAmt": round(save, 2),
            "InvestAmt": round(invest, 2),
            "CumulativeWealth": round(balance, 2)
        })

    df = pd.DataFrame(records)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/goal_plan.csv", index=False)
    print("\nGoal plan saved to outputs/goal_plan.csv")
    return df

if __name__ == "__main__":
    income = float(input("Enter your monthly income: "))
    expense = float(input("Enter your average monthly expense: "))
    months = int(input("Enter number of months for goal planning: "))
    generate_goal_plan(income, expense, months)
