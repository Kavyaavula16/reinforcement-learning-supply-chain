"""Single-product inventory env with multiple heterogeneous suppliers."""
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEFAULT_SUPPLIERS = [
    dict(name="cheap_slow_risky", unit_cost=4.5, lead=(3, 6), p_down=0.03,  p_up=0.10),
    dict(name="mid",              unit_cost=5.0, lead=(2, 4), p_down=0.01,  p_up=0.30),
    dict(name="fast_reliable",    unit_cost=6.0, lead=(1, 2), p_down=0.002, p_up=0.50),
]


class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, suppliers=None, horizon=365, base_demand=20.0, season_amp=0.3,
                 order_levels=(0, 10, 20, 40), hold_cost=0.5, stockout_cost=10.0,
                 fixed_order_cost=10.0, init_inventory=60, max_inventory=300, hist_len=7):
        super().__init__()
        self.sup = suppliers if suppliers is not None else DEFAULT_SUPPLIERS
        self.n = len(self.sup)
        self.horizon, self.base_demand, self.season_amp = horizon, base_demand, season_amp
        self.levels = np.array(order_levels, dtype=np.int64)
        self.hold_cost, self.stockout_cost = hold_cost, stockout_cost
        self.fixed_order_cost = fixed_order_cost
        self.init_inventory, self.max_inventory = init_inventory, max_inventory
        self.hist_len = hist_len
        # Stress-test hooks (used later for robustness experiments)
        self.demand_multiplier = lambda t: 1.0

        self.action_space = spaces.MultiDiscrete([len(self.levels)] * self.n)
        obs_dim = 1 + self.n + self.n + self.n + hist_len + 2
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(obs_dim,), dtype=np.float32)

    # ---------- helpers ----------
    def _expected_demand(self, t):
        return self.base_demand * (1 + self.season_amp * np.sin(2 * np.pi * t / 365))

    def _obs(self):
        pipe = np.array([sum(q for q, _ in s) for s in self.pipeline], dtype=np.float32)
        eta = np.array([min((d for _, d in s), default=0) for s in self.pipeline], dtype=np.float32)
        ang = 2 * np.pi * self.t / 365
        obs = np.concatenate([
            [self.inventory / self.max_inventory],
            pipe / 100.0,
            eta / 10.0,
            self.down.astype(np.float32),
            np.array(self.hist, dtype=np.float32) / 40.0,
            [np.sin(ang), np.cos(ang)],
        ]).astype(np.float32)
        return obs

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        d_ss, x_ss, l_ss = np.random.SeedSequence(s).spawn(3)
        self.demand_rng = np.random.default_rng(d_ss)
        self.disrupt_rng = np.random.default_rng(x_ss)
        self.lead_rng = np.random.default_rng(l_ss)

        self.t = 0
        self.inventory = float(self.init_inventory)
        self.pipeline = [[] for _ in range(self.n)]   # per supplier: [qty, days_left]
        self.down = np.zeros(self.n, dtype=bool)
        self.hist = deque([self.base_demand] * self.hist_len, maxlen=self.hist_len)
        return self._obs(), {}

    def step(self, action):
        qty = self.levels[np.asarray(action, dtype=np.int64)]

        # Draw randomness every step for every supplier (keeps streams policy-independent)
        u_dis = self.disrupt_rng.random(self.n)
        leads = np.array([self.lead_rng.integers(s["lead"][0], s["lead"][1] + 1) for s in self.sup])
        demand = int(self.demand_rng.poisson(self._expected_demand(self.t) * self.demand_multiplier(self.t)))

        # 1) disruption state transitions
        for i, s in enumerate(self.sup):
            if self.down[i]:
                if u_dis[i] < s["p_up"]:
                    self.down[i] = False
            elif u_dis[i] < s["p_down"]:
                self.down[i] = True

        # 2) place orders (rejected if supplier is down)
        order_cost, ordered, rejected = 0.0, np.zeros(self.n), np.zeros(self.n)
        for i, s in enumerate(self.sup):
            q = int(qty[i])
            if q <= 0:
                continue
            if self.down[i]:
                rejected[i] = q
                continue
            self.pipeline[i].append([q, int(leads[i])])
            ordered[i] = q
            order_cost += self.fixed_order_cost + s["unit_cost"] * q

        # 3) advance shipments; deliver those due from suppliers that are up
        received = 0
        for i in range(self.n):
            remaining = []
            for q, d in self.pipeline[i]:
                d -= 1
                if d <= 0 and not self.down[i]:
                    received += q
                else:
                    remaining.append([q, max(d, 0)])
            self.pipeline[i] = remaining
        self.inventory += received
        overflow = max(0.0, self.inventory - self.max_inventory)
        self.inventory = min(self.inventory, self.max_inventory)

        # 4) demand
        fulfilled = min(self.inventory, demand)
        lost = demand - fulfilled
        self.inventory -= fulfilled
        self.hist.append(demand)

        # 5) costs
        holding = self.hold_cost * self.inventory
        stockout = self.stockout_cost * lost
        cost = order_cost + holding + stockout

        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon
        info = dict(demand=demand, fulfilled=fulfilled, lost=lost, ordered=ordered,
                    rejected=rejected, received=received, overflow=overflow,
                    order_cost=order_cost, holding_cost=holding, stockout_cost=stockout,
                    cost=cost, inventory=self.inventory)
        return self._obs(), -float(cost), terminated, truncated, info
