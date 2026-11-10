"""A day-by-day environment for new algorithms (RL or anything else): the same gates, information and settlement as
qest.evaluate, behind reset() / step().

    from qest.env import QestEnv
    env = QestEnv('val')                 # 'train' | 'val' | 'test'; forecasts walk forward as in evaluate
    obs = env.reset()
    while obs is not None:
        plan = my_policy(obs)            # {unit: dict(da=..., cap={...}, margins={...}, soc_ref=..., policy='soc_aware')}
        obs, reward, info = env.step(plan)   # reward = the day's EUR (all units); info = per-unit breakdown

obs (gate A, D-1 09:00): day, idx (MTU starts, UTC), fc (quantile forecasts per MTU), gen_fc (D-1 MW per unit),
calib (activation tables), units. Nothing from the delivery day itself is in it.
"""
import numpy as np

from qest import market
from qest.data import afrr_margin_table, days, frame
from qest.evaluate import forecasts, unit_gen
from qest.power import generation
from qest.strategies import DayContext, calibration


class QestEnv:
    def __init__(self, split='val', units=('battery', 'hybrid', 'solar', 'wind'), cfg=None, forecast='lgbm'):
        self.f = frame(); self.cfg = cfg or market.Config(); self.units = market.portfolio(units, official=self.cfg.official)
        fit = days(self.f, 'train') if split in ('train', 'val') else days(self.f, ('train', 'val'))
        self.calib = calibration(self.f, fit); self.days = days(self.f, split)
        self.fc = forecasts(split, forecast) if split != 'train' else None  # train: in-sample, bring your own forecasts
        self.act, self.gfc = unit_gen(generation(), self.units); self.atab = afrr_margin_table().reindex(self.f.index).fillna(0)
        self.i = 0

    def _obs(self):
        if self.i >= len(self.days):
            return None
        d = self.days[self.i]; idx = self.f.index[(self.f.day == d).to_numpy()]
        return DayContext(idx, self.fc.reindex(idx) if self.fc is not None else None,
                          {k: v.reindex(idx).fillna(0).to_numpy() for k, v in self.gfc.items()}, self.calib)

    def reset(self):
        self.i = 0; self.obs = self._obs(); return self.obs

    def step(self, plan):
        idx = self.obs.idx; day = self.f.loc[idx]
        res = market.settle_day(self.units, plan, day, {k: v.reindex(idx).fillna(0).to_numpy() for k, v in self.act.items()}, self.cfg,
                                afrr_tab=self.atab.loc[idx])
        self.i += 1; self.obs = self._obs()
        return self.obs, float(sum(r['total'] for r in res.values())), res


if __name__ == '__main__':
    from qest.strategies import naive_da
    env = QestEnv('test'); obs = env.reset(); total = 0.0
    while obs is not None:
        obs, r, info = env.step(naive_da(env.units, obs, env.cfg)); total += r
    print('naive_da on Q4 via QestEnv: EUR %.0f (matches qest.evaluate before the monthly tariff)' % total)
