"""Shared evaluation harness: any policy is a callable policy(env, obs) -> action."""
import numpy as np
import pandas as pd


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    cost = demand = lost = fulfilled = stockout_days = steps = 0
    inv_sum = 0.0
    parts = np.zeros(3)
    ordered = np.zeros(env.n)
    rejected = np.zeros(env.n)
    done = False
    while not done:
        action = policy(env, obs)
        obs, r, term, trunc, info = env.step(action)
        cost += info["cost"]
        parts += [info["order_cost"], info["holding_cost"], info["stockout_cost"]]
        demand += info["demand"]
        lost += info["lost"]
        fulfilled += info["fulfilled"]
        stockout_days += int(info["lost"] > 0)
        inv_sum += info["inventory"]
        ordered += info["ordered"]
        rejected += info["rejected"]
        steps += 1
        done = term or trunc
    avg_inv = inv_sum / steps
    row = dict(
        seed=seed,
        cost=cost,
        service=1 - lost / demand if demand else 1.0,
        stockout_freq=stockout_days / steps,
        turnover=fulfilled / avg_inv if avg_inv > 0 else np.nan,
        avg_inventory=avg_inv,
        order_cost=parts[0],
        holding_cost=parts[1],
        stockout_cost=parts[2],
        controllable_cost=parts[1] + parts[2],
    )
    tot = ordered.sum()
    for i in range(env.n):
        row[f"share_S{i}"] = ordered[i] / tot if tot > 0 else 0.0
        row[f"rejected_S{i}"] = rejected[i]
    return row


def evaluate(env, policy, seeds):
    return pd.DataFrame([run_episode(env, policy, s) for s in seeds])


def summarize(df):
    """mean, std and 95% CI half-width for each metric."""
    cols = [c for c in df.columns if c != "seed"]
    out = pd.DataFrame({
        "mean": df[cols].mean(),
        "std": df[cols].std(),
    })
    out["ci95"] = 1.96 * out["std"] / np.sqrt(len(df))
    return out
