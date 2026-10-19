import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from supply_env import SupplyChainEnv
from evaluation import evaluate


def ppo_policy(model):
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def train(seed, total_steps=1_000_000, chunk=100_000, out_dir=".",
          val_seeds=range(20000, 20020), env_kwargs=None, **ppo_kwargs):
    env_kwargs = env_kwargs or {}
    params = dict(n_steps=1024, batch_size=256, ent_coef=0.01, learning_rate=3e-4,
                  gamma=0.99, device="cpu", seed=seed, verbose=0)
    params.update(ppo_kwargs)

    venv = DummyVecEnv([lambda: SupplyChainEnv(**env_kwargs) for _ in range(4)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=params["gamma"])
    model = PPO("MlpPolicy", venv, **params)
    eval_env = SupplyChainEnv(**env_kwargs)

    os.makedirs(out_dir, exist_ok=True)
    log, best, steps = [], np.inf, 0
    while steps < total_steps:
        model.learn(chunk, reset_num_timesteps=False)
        steps += chunk
        df = evaluate(eval_env, ppo_policy(model), val_seeds)
        row = dict(steps=steps, val_cost=df.cost.mean(), val_service=df.service.mean(),
                   val_controllable=df.controllable_cost.mean())
        log.append(row)
        print({k: round(v, 3) for k, v in row.items()})
        if row["val_cost"] < best:
            best = row["val_cost"]
            model.save(f"{out_dir}/best")
    model.save(f"{out_dir}/final")
    pd.DataFrame(log).to_csv(f"{out_dir}/val_log.csv", index=False)
    return pd.DataFrame(log)
