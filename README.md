
# RL for Personal Finance — Paper Replica

This project replicates the core setup described in *"Reinforcement Learning for Personal Finance Management"* (state = [income, savings, investments, expenses]; discrete allocation actions; stochastic investment returns; Q-learning via DQN).

## Environment

- State: `[income, savings, investments, required_expense]`
- Actions (discrete): allocation triplets `(spend%, save%, invest%)`
- Transition: apply allocation, realize expenses, grow investments with random return in `[-5%, +10%]`
- Reward: `wealth = savings + investments` minus penalties for unmet expenses and negative cash
- Horizon: 120 months (10 years)

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

For baselines:
```bash
python baseline.py
```

## Files
- `env_personal_finance.py` — custom Gym-like environment
- `dqn_agent.py` — PyTorch DQN (Q-network, replay buffer, epsilon-greedy)
- `train.py` — training loop and logging
- `baseline.py` — fixed-allocation baselines (e.g., 50-30-20)
- `config.yaml` — hyperparameters
- `requirements.txt` — Python deps

## Notes
This replica follows the paper's simplified simulation assumptions and discrete actions. You can extend it to variable income, richer reward shaping, or alternative RL algorithms (PPO, A2C).
