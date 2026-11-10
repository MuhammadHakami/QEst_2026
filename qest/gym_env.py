"""Gymnasium (1.x) environment over the KratTrade market, and a PPO agent (Stable-Baselines3) trained in it.

One episode = one Tallinn delivery day, one step (a contextual bandit, the gate structure makes it so):
  observation (gate A, D-1 09:00): 16 day-level numbers from the gate-A forecasts only (DA level / spread / peak-trough,
               the four capacity prices, forecast aFRR shares, hybrid PV, weekday, weekend);
  action:      the MILP's risk knobs for that day, in [0, 1]^3: kappa_up, kappa_dn (haircut on the activation energy the
               day-ahead plan leans on) and tol (SoC drift before gate C prices a direction out, mapped to 1-30 %);
  reward:      the day's official settlement (slide 12), battery + hybrid, in EUR 100k.
The MILP (qest.optimize) keeps every physical and market constraint, so the agent can never send an infeasible plan:
it learns when to trust the forecast activations (an RL-in-the-loop optimiser, not an RL dispatcher).
Training uses validation (Q3) days only; Q4 is scored once with the deterministic policy.
usage: .venv/bin/python -m qest.gym_env [--steps 3000]   -> results/rl_ppo_test.csv
"""
import argparse

import gymnasium as gym
import numpy as np

from qest import market
from qest.data import afrr_margin_table, days, frame
from qest.evaluate import forecasts, unit_gen
from qest.power import generation
from qest.strategies import DayContext, calibration, milp


def day_features(ctx):
    fc = ctx.fc; c = lambda k: fc[f'{k}_q50'].fillna(0).to_numpy()
    da = c('da'); wd = ctx.idx.tz_convert('Europe/Tallinn')[0].dayofweek
    return np.array([da.mean() / 100, da.std() / 100, (da.max() - da.min()) / 100,
                     *(c(f'cap_{k}').mean() / 50 for k in market.PRODUCTS),
                     c('afrr_up_s').mean(), c('afrr_dn_s').mean(), c('mfrr').mean() / 100, c('imb').mean() / 100,
                     (fc['da_q90'] - fc['da_q10']).mean() / 100 if 'da_q90' in fc else 0.0,
                     float(np.mean(ctx.gen_fc.get('hybrid', [0]))) / 20, wd / 6, float(wd >= 5), 1.0], np.float32)


class KratTradeEnv(gym.Env):
    metadata = {}

    def __init__(self, split='val', units=('battery', 'hybrid'), sequential=False, kind='lgbm', margins=None):
        self.f = frame(); self.cfg = market.Config(); self.units = market.portfolio(units, official=self.cfg.official)
        fit = days(self.f, 'train') if split == 'val' else days(self.f, ('train', 'val'))
        self.calib = calibration(self.f, fit) if margins is None else calibration(self.f, fit, margins)
        self.days = days(self.f, split); self.fc = forecasts(split, kind)
        self.act, self.gfc = unit_gen(generation(), self.units); self.atab = afrr_margin_table().reindex(self.f.index).fillna(0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (16,), np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, (3,), np.float32)
        self.sequential = sequential; self.i = -1

    def _ctx(self, d):
        idx = self.f.index[(self.f.day == d).to_numpy()]
        return DayContext(idx, self.fc.reindex(idx), {k: v.reindex(idx).fillna(0).to_numpy() for k, v in self.gfc.items()}, dict(self.calib))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = (self.i + 1) % len(self.days) if self.sequential else int(self.np_random.integers(len(self.days)))
        self.d = self.days[self.i]; self.ctx = self._ctx(self.d)
        return day_features(self.ctx), {'day': str(self.d)}

    def step(self, action):
        a = np.clip(np.asarray(action, float), 0, 1)
        self.ctx.calib.update(kappa_up=a[0], kappa_dn=a[1], tol=.01 * 30 ** a[2])
        plan = milp(self.units, self.ctx, self.cfg); idx = self.ctx.idx
        res = market.settle_day(self.units, plan, self.f.loc[idx], {k: v.reindex(idx).fillna(0).to_numpy() for k, v in self.act.items()},
                                self.cfg, afrr_tab=self.atab.loc[idx])
        eur = float(sum(r['total'] for r in res.values()))
        return day_features(self.ctx), eur / 1e5, True, False, {'eur': eur, 'day': str(self.d)}


def main():
    from stable_baselines3 import PPO
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=3000); a = ap.parse_args()
    env = KratTradeEnv('val')
    model = PPO('MlpPolicy', env, n_steps=128, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.0, seed=0, verbose=0,
                policy_kwargs=dict(net_arch=[32, 32]))
    model.learn(total_timesteps=a.steps)
    rows = []
    for split in ('val', 'test'):
        e = KratTradeEnv(split, sequential=True)
        for _ in e.days:
            obs, info = e.reset(); act, _ = model.predict(obs, deterministic=True); _, r, _, _, inf = e.step(act)
            rows.append(dict(split=split, day=inf['day'], eur=inf['eur'], kappa_up=act[0], kappa_dn=act[1], tol=.01 * 30 ** act[2]))
    import pandas as pd
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'rl_ppo.csv', index=False); model.save(RESULTS / 'rl_ppo_model')
    print(R.groupby('split')[['eur']].sum().round(0).to_string()); print(R.groupby('split')[['kappa_up', 'kappa_dn', 'tol']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
