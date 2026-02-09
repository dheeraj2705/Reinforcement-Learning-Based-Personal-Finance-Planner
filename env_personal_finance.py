
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class FinanceConfig:
    monthly_income: float = 10000.0
    expense_min: float = 4000.0
    expense_max: float = 6500.0
    periods_per_month: int = 2
    horizon_months: int = 12
    seed: int = 42


class PersonalFinanceEnv:
    

    def __init__(self, config: FinanceConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.month = 0
        self.period = 0
        self.done = False

        self.state_dim = 3
        self.action_dim = 9

        ratios = [0.35,0.45,0.5, 0.55]
        self.actions = np.array(
            [[s, sv, 1 - s - sv] for s in ratios for sv in [0.2, 0.3,0.35, 0.4]]
        )
        self.actions = np.clip(self.actions, 0, 1)

        self.savings = 0.0
        self.invest = 0.0
        self.wealth = 0.0
        self.expense_need = 0.0
        self.income = config.monthly_income / config.periods_per_month

    
    def reset(self, seed: int = None, monthly_income: float = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if monthly_income is not None:
            self.cfg.monthly_iancome = monthly_income
            self.income = monthly_income / self.cfg.periods_per_month

        self.month = 0
        self.period = 0
        self.done = False
        self.savings = 0.0
        self.invest = 0.0
        self.wealth = 0.0
        self.expense_need = self._sample_expense()

        return self._get_state()

    def _sample_expense(self):
        return float(
            self.rng.uniform(self.cfg.expense_min, self.cfg.expense_max)
        )

    def _get_state(self):
        return np.array(
            [
                self.income / self.cfg.monthly_income,
                self.expense_need / self.cfg.expense_max,
                self.wealth / (self.cfg.monthly_income * self.cfg.horizon_months),
            ],
            dtype=np.float32,
        )

    def step(self, action_idx: int):
        
        alloc = self.actions[action_idx]
        spend_ratio, save_ratio, invest_ratio = alloc
        income = self.income

        spend_amt = income * spend_ratio
        save_amt = income * save_ratio
        invest_amt = income * invest_ratio
        total_alloc = spend_amt + save_amt + invest_amt

        overspend = max(0.0, spend_amt - self.expense_need)
        underspend = max(0.0, self.expense_need - spend_amt)

   
        self.savings += save_amt
        self.invest += invest_amt
        self.wealth = self.savings + self.invest

        
        reward = (
            (save_amt + invest_amt) * 0.002
            - (overspend * 0.0025)
            + (underspend * 0.0005)
        )

        balance_penalty = abs(spend_ratio - 0.5) + abs(save_ratio - 0.25) + abs(invest_ratio - 0.25)
        reward -= 0.05 * balance_penalty

        self.period += 1
        if self.period >= self.cfg.periods_per_month:
            self.month += 1
            self.period = 0

        if self.month >= self.cfg.horizon_months:
            self.done = True

        self.expense_need = self._sample_expense()

        info = {
            "month": self.month,
            "period": self.period,
            "wealth": self.wealth,
            "savings": self.savings,
            "invest": self.invest,
        }

        return self._get_state(), reward, self.done, info

 
    def render(self):
        print(
            f"Month {self.month + 1}, Period {self.period + 1} | "
            f"Expense Need: {self.expense_need:.2f} | "
            f"Wealth: {self.wealth:.2f}"
        )

    def sample_action(self):
        return int(self.rng.integers(0, self.action_dim))
