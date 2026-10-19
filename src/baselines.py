"""Classical inventory baselines adapted to multi-supplier allocation."""
import numpy as np
import pandas as pd
from evaluation import evaluate


class InventoryPolicy:
    """kind='sS'  : if IP <= s, order up to S.
       kind='fixed': if IP <= s, order a fixed quantity Q.
       Orders are split across suppliers by a cost + lead-time ranking."""

    def __init__(self, kind, s, S=None, Q=None, lead_weight=0.0):
        assert kind in ("sS", "fixed")
        self.kind, self.s, self.S, self.Q = kind, s, S, Q
        self.lead_weight = lead_weight

    def _ranking(self, env):
        scores = [sp["unit_cost"] + self.lead_weight * np.mean(sp["lead"]) for sp in env.sup]
        return np.argsort(scores, kind="stable")

    def __call__(self, env, obs=None):
        pos = env.inventory + sum(q for p in env.pipeline for q, _ in p)
        action = np.zeros(env.n, dtype=np.int64)
        if pos > self.s:
            return action
        remaining = (self.S - pos) if self.kind == "sS" else self.Q
        levels = env.levels
        for i in self._ranking(env):
            if remaining <= 0:
                break
            if env.down[i]:
                continue
            ok = levels >= remaining
            idx = int(np.argmax(ok)) if ok.any() else len(levels) - 1
            action[i] = idx
            remaining -= levels[idx]
        return action

    def __repr__(self):
        return (f"{self.kind}(s={self.s}, S={self.S}, Q={self.Q}, w={self.lead_weight})")


def grid_search(env, kind, grid, seeds):
    rows = []
    for params in grid:
        df = evaluate(env, InventoryPolicy(kind, **params), seeds)
        rows.append({**params, "mean_cost": df.cost.mean(), "service": df.service.mean()})
    return pd.DataFrame(rows).sort_values("mean_cost").reset_index(drop=True)


class EmergencyPolicy(InventoryPolicy):
    """(s,S) with an emergency rule: when an order is triggered AND on-hand stock
    is <= s_emerg, rank suppliers by speed (fastest first) instead of cost."""

    def __init__(self, s, S, s_emerg, lead_weight=0.0):
        super().__init__("sS", s=s, S=S, lead_weight=lead_weight)
        self.s_emerg = s_emerg
        self._emerg = False

    def _ranking(self, env):
        if self._emerg:
            return np.argsort([np.mean(sp["lead"]) for sp in env.sup], kind="stable")
        return super()._ranking(env)

    def __call__(self, env, obs=None):
        self._emerg = env.inventory <= self.s_emerg
        return super().__call__(env, obs)

    def __repr__(self):
        return f"emergency(s={self.s}, S={self.S}, s_emerg={self.s_emerg})"
