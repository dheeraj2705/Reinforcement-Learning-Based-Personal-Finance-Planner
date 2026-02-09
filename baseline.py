import numpy as np 
from typing import Tuple
from env_personal_finance import PersonalFinanceEnv, FinanceConfig

def simulate_baseline(spend: float=0.5, save: float=0.3, invest: float=0.2,
                      episodes: int=50, steps: int=120, seed: int=123) -> float:
    cfg = FinanceConfig(seed=seed, horizon_months=steps)
    env = PersonalFinanceEnv(cfg)
    # overwrite env actions to single fixed allocation
    env.actions = np.array([[spend, save, invest]], dtype=np.float32)
    env.action_dim = 1

    total_rewards = []
    for ep in range(episodes):
        s = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for t in range(steps):
            s2, r, d, info = env.step(0)
            ep_reward += r
            s = s2
            if d:
                break
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))

if __name__ == "__main__":
    for alloc in [(0.5,0.3,0.2),(0.6,0.2,0.2),(0.4,0.3,0.3)]:
        avg_r = simulate_baseline(*alloc)
        print(f"Baseline {alloc}: avg episode reward = {avg_r:.2f}")