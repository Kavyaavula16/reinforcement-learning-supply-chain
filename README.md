# reinforcement-learning-supply-chain
# Reinforcement Learning for Autonomous Supply Chain Optimization

MS course project (CSCI 6970) by Kavya Avula.

A custom Gymnasium environment for single-product inventory control with **multiple heterogeneous suppliers**, where the agent jointly decides *how much to order* and *from whom*. A PPO agent is trained in this environment and compared with tuned classical baselines ((s,S) and fixed reorder-point) under identical random conditions.

## Key results

Test set: 100 held-out episodes (seeds 10000-10099), 365 days each. PPO is the mean over 5 training seeds (final checkpoints). Differences are paired over identical test seeds, with 95% CIs.

| Policy | Mean cost | Service level | PPO cost minus policy cost (95% CI) |
|---|---|---|---|
| **PPO** (5 seeds) | **44,561** (std 51 across seeds) | 93.7% | n/a |
| (s,S), best cost | 46,582 | 88% | -2,021 ± 231 |
| Fixed reorder, best cost | 46,717 | 87% | -2,156 ± 242 |
| (s,S), ≈95% service | 47,388 | 94% | -2,828 ± 157 |
| Fixed reorder, ≈95% service | 47,184 | 96% | -2,623 ± 112 |

- PPO is 4-6% cheaper than every tuned baseline, including baselines tuned to a similar service level.
- At matched service against (s,S) ≈95%, stockout cost is about the same (4.6k vs 4.3k) but holding cost is less than half (4.2k vs 9.0k).
- Results are consistent across 5 PPO seeds (per-seed mean cost 44.5k-44.6k).
- A classical emergency rule (switch to the fastest supplier when stock is critical) did not improve on plain (s,S) in this environment.

### Supplier allocation

PPO splits volume between the cheap/slow supplier and the mid supplier; the baselines commit to one primary supplier.

| Policy | S0 (cheap, slow, risky) | S1 (mid) | S2 (fast, reliable) |
|---|---|---|---|
| PPO (range over 5 seeds) | 43-57% | 40-54% | 2-3% |
| (s,S) best cost | 78% | 21% | 1% |

When S0 is available, PPO still sends 24-44% of volume to S1 (baseline: about 2%), so this is proactive mixing, not only fallback during outages. The exact split varies by seed with no cost difference. *Why* mixing helps (for example, S1's shorter lead time allowing less safety stock) is a hypothesis that has not been tested.

### Robustness (no retraining; 50 seeds; PPO minus baseline, negative = PPO cheaper)

| Scenario | vs (s,S) best | vs (s,S) ≈95% | vs fixed ≈95% | PPO service |
|---|---|---|---|---|
| Nominal | -2,009 ± 298 | -2,851 ± 193 | -2,620 ± 149 | 93.8% |
| 2x demand spike (20 days) | -2,291 ± 294 | -2,755 ± 198 | -2,423 ± 192 | 90.8% |
| Demand +30% | -2,440 ± 365 | -575 ± 314 | -556 ± 282 | 85.4% |
| S0 long outages (~50% downtime) | -3,740 ± 773 | -3,718 ± 488 | -1,994 ± 203 | 91.8% |
| Slower lead times | -2,839 ± 333 | -1,190 ± 339 | -10,022 ± 153* | 79.2% |

\*The fixed ≈95% baseline reaches 99.9% service in this scenario by over-ordering, while PPO runs at 79%. The -10k figure is not a like-for-like comparison. **PPO's service level drops under slower lead times**, which is a real weakness. The advantage is also smallest under sustained demand growth.

Figures are in [`final/figures/`](final/figures/).

## Environment (`src/supply_env.py`)

- **Horizon:** 365 daily steps. **Demand:** Poisson(20 x (1 + 0.3 sin(2*pi*t/365))). Unmet demand is lost.
- **Costs:** holding 0.5 per unit-day, stockout 10 per unit, fixed order cost 10 per order, plus unit cost.
- **Inventory:** start 60, cap 300.
- **Suppliers** (each is up/down as a Markov chain; down suppliers reject new orders and hold shipments):

| Supplier | Unit cost | Lead time (days) | P(down)/day | P(recover)/day |
|---|---|---|---|---|
| S0 cheap, slow, risky | 4.5 | 3-6 | 0.03 | 0.10 |
| S1 mid | 5.0 | 2-4 | 0.01 | 0.30 |
| S2 fast, reliable | 6.0 | 1-2 | 0.002 | 0.50 |

- **Action:** `MultiDiscrete([4,4,4])`, an order level from {0, 10, 20, 40} for each supplier (order size and supplier choice decided jointly).
- **Observation (19 values):** inventory, pipeline quantity per supplier, days to next arrival per supplier, supplier up/down flags, last 7 demands, sin/cos of day.
- **Reward:** minus the day's total cost.
- **Fair comparison:** demand, disruptions, and lead times use separate random streams that are independent of the action, so every policy sees identical conditions for a given seed.

## Methods

- **Baselines** (`src/baselines.py`): fixed reorder-point and (s,S) on inventory position, with suppliers ranked by `unit_cost + w * mean_lead_time`, skipping suppliers that are down. Parameters were grid-searched on training seeds 0-9 only (`src/baseline_params.json`). Two variants each: lowest cost, and ≈95% service on training seeds.
- **PPO** (`src/train_ppo.py`, Stable-Baselines3): MLP policy, 1M steps, 4 parallel envs, n_steps 1024, batch 256, lr 3e-4, gamma 0.99, entropy coef 0.01, reward normalization. Five training seeds (0-4), about 16.5 min each on Colab CPU. Checkpoints are evaluated on validation seeds 20000-20019.
- **Seed separation:** baseline tuning seeds 0-9, validation 20000-20019, test 10000-10099. PPO trains on random episodes and never sees the test seeds.
- **Evaluation** (`src/evaluation.py`): total cost, cost decomposition, service level, stockout frequency, inventory turnover, supplier shares.

## Design decisions and honesty notes

- **Environment recalibration.** With the first supplier parameters (S0 at cost 4.0), S0 dominated the baselines (91% of volume), so supplier choice carried little decision value. Supplier parameters were therefore recalibrated *after seeing baseline results but before any PPO was trained*, and the baselines were re-tuned on the final environment. The environment was not changed after PPO results existed.
- **Baselines are tuned only on the nominal environment**, as is PPO, so the robustness tests measure transfer for both. Baselines re-tuned per scenario were not evaluated.
- The `final` PPO checkpoints are reported (no selection on validation seeds); `best` checkpoints give nearly the same result (44,604).

## Limitations

- Single product, single echelon, single demand segment. The proposal's **per-segment service fairness analysis is not covered**; multiple demand segments are future work.
- **Supplier-side fairness:** no supplier is starved among S0/S1 (each keeps roughly half the volume). S2 receives only 2-3% (baselines: about 1%), consistent with it being the most expensive supplier. The reward is pure cost, so contractual or relationship value of supplier volume is not modeled.
- PPO's service level falls to 79% under much slower lead times, and its advantage shrinks under sustained +30% demand.
- The cause of PPO's supplier mixing is untested.
- All parameters are synthetic; no real data was used.

## Repository layout

```
src/        supply_env.py, evaluation.py, baselines.py, train_ppo.py, baseline_params.json
final/      validation logs, test and robustness CSVs, mechanism analysis, figures/
notebooks/  Colab notebook used to run all experiments
```
Trained models (`models/`) are not in the repo; they are produced by `src/train_ppo.py`.

## Reproduce

1. Open the notebook in Colab, mount Drive, and `pip install gymnasium stable-baselines3 tensorboard` (versions in `requirements.txt`; PyTorch is the Colab default, CPU).
2. Tune baselines on seeds 0-9, then train PPO: `train(seed=s, total_steps=1_000_000, chunk=50_000, out_dir=...)` for s in 0..4.
3. Evaluate all policies on test seeds 10000-10099 with `evaluation.evaluate`, then run the robustness scenarios.

## References

Hubbs et al. (2020) OR-Gym; Stranieri & Stella (2022); Oroojlooyjadid et al. (2022); Rolf et al. (2023); Kim, Kim & Lee (2024); Sultana et al. (2020); Preil & Krapp (2022).
