"""The competition's data as one 15-min frame (UTC index = MTU start), plus the time-series split.

Sources (dataset/, Estonia, delivery days 1 Jan - 25 Oct 2026, Tallinn time):
- prices_15min: day-ahead (Nord Pool EE), mFRR energy, aFRR energy (Baltic transparency + PICASSO 15-min), imbalance,
  capacity prices (aFRR/mFRR, up/down, EUR/MW/h).
- afrr_picasso_4s: PICASSO aFRR prices every 4 s. Aggregated here per MTU and direction: the mean price over the ticks
  where the direction was active and the active share of the MTU (an empty price means no activation that tick: not zero).
- solarpark_forecast_15min: the 42.3 MW park, latest forecast (proxy for what was produced) and the D-1 forecast.

Split (whole Tallinn delivery days): train Q1-Q2 (1 Jan - 30 Jun), validation Q3 (1 Jul - 30 Sep), test Q4 (1 - 25 Oct).
Models are fitted on train, tuned on validation, then refitted on train + validation and scored once on test.
usage: .venv/bin/python -m qest.data   (builds data/derived/frame.parquet and prints a summary)
"""
import numpy as np
import pandas as pd

from qest.paths import DATASET, DERIVED, RAW

TZ = 'Europe/Tallinn'
SPLITS = dict(train=('2026-01-01', '2026-06-30'), val=('2026-07-01', '2026-09-30'), test=('2026-10-01', '2026-10-25'))


def afrr_15min():
    """Per MTU and direction: mean PICASSO price over the active 4-s ticks, active share (0-1), tick count."""
    a = pd.read_parquet(DATASET / 'afrr_picasso_4s.parquet', columns=['mtu_utc', 'picasso_afrr_up_eur_mwh', 'picasso_afrr_down_eur_mwh'])
    g = a.groupby('mtu_utc')
    out = pd.DataFrame({'afrr4s_up_price': g.picasso_afrr_up_eur_mwh.mean(), 'afrr4s_down_price': g.picasso_afrr_down_eur_mwh.mean(),
                        'afrr4s_up_share': g.picasso_afrr_up_eur_mwh.count() / 225, 'afrr4s_down_share': g.picasso_afrr_down_eur_mwh.count() / 225,
                        'afrr4s_ticks': g.size()})
    out.index.name = 'datetime_utc'
    return out


MARGINS = (0, 10, 25, 50, 100, 200, 500)  # EUR/MWh beyond the MTU's DA price at which our aFRR energy bid sits


def afrr_margin_table():
    """PICASSO merit order, per MTU: if our aFRR UP energy bid is DA + m, it is activated on the 4-s ticks whose UP price
    is >= the bid (every cheaper bid runs fully at the cross-border marginal price) and earns that tick's price; DOWN
    mirrors at DA - m. Columns up_s_<m>, up_p_<m> (share of the MTU, mean price over the activated ticks), dn_s_<m>, dn_p_<m>."""
    p = DERIVED / 'afrr_margins.parquet'
    if p.exists():
        return pd.read_parquet(p)
    a = pd.read_parquet(DATASET / 'afrr_picasso_4s.parquet', columns=['mtu_utc', 'picasso_afrr_up_eur_mwh', 'picasso_afrr_down_eur_mwh'])
    da = pd.read_parquet(DATASET / 'prices_15min.parquet', columns=['datetime_utc', 'nps_dayahead_eur_mwh']).set_index('datetime_utc').nps_dayahead_eur_mwh
    a['da'] = a.mtu_utc.map(da)
    cols = {}
    g = a.groupby('mtu_utc')
    for m in MARGINS:
        up = a.picasso_afrr_up_eur_mwh.where(a.picasso_afrr_up_eur_mwh >= a.da + m)
        dn = a.picasso_afrr_down_eur_mwh.where(a.picasso_afrr_down_eur_mwh <= a.da - m)
        cols[f'up_s_{m}'] = up.groupby(a.mtu_utc).count() / 225; cols[f'up_p_{m}'] = up.groupby(a.mtu_utc).mean()
        cols[f'dn_s_{m}'] = dn.groupby(a.mtu_utc).count() / 225; cols[f'dn_p_{m}'] = dn.groupby(a.mtu_utc).mean()
    out = pd.DataFrame(cols).fillna(0.0); out.index.name = 'datetime_utc'
    DERIVED.mkdir(parents=True, exist_ok=True); out.to_parquet(p)
    return out


def elering():
    import json
    d = json.load(open(RAW / 'elering_system.json'))
    real = pd.DataFrame(d['real']).assign(t=lambda x: pd.to_datetime(x.timestamp, unit='s', utc=True)).set_index('t').drop(columns='timestamp')
    plan = pd.DataFrame(d['plan']).assign(t=lambda x: pd.to_datetime(x.timestamp, unit='s', utc=True)).set_index('t').drop(columns='timestamp')
    real = real[['production', 'consumption', 'system_balance', 'production_renewable', 'solar_energy_production']].resample('15min').mean().add_prefix('ee_real_')
    plan = plan[['production', 'consumption', 'production_renewable', 'solar_energy_forecast']].resample('15min').mean().add_prefix('ee_plan_')
    return real.join(plan, how='outer')


def build():
    p = pd.read_parquet(DATASET / 'prices_15min.parquet').set_index('datetime_utc')
    s = pd.read_parquet(DATASET / 'solarpark_forecast_15min.parquet').set_index('datetime_utc')
    f = p.drop(columns=['datetime_local']).join(s[['solarpark_forecast_mw', 'solarpark_forecast_d1_mw', 'solarpark_forecast_devices']])
    f = f.join(afrr_15min())
    if (RAW / 'elering_system.json').exists():
        f = f.join(elering())
    f['local'] = f.index.tz_convert(TZ)
    f['day'] = f.local.dt.normalize().dt.tz_localize(None)  # Tallinn delivery day
    f['mtu'] = f.groupby('day').cumcount()  # 0..95 (92/100 on DST days)
    f['split'] = 'none'
    for k, (a, b) in SPLITS.items():
        f.loc[(f.day >= a) & (f.day <= b), 'split'] = k
    DERIVED.mkdir(parents=True, exist_ok=True); f.to_parquet(DERIVED / 'frame.parquet')
    return f


def frame():
    p = DERIVED / 'frame.parquet'
    return pd.read_parquet(p) if p.exists() else build()


def days(f, split):
    return sorted(f.loc[f.split == split, 'day'].unique()) if isinstance(split, str) else sorted(set().union(*(days(f, s) for s in split)))


def check(f):
    """Our 4-s aggregation must match the dataset's own 15-min PICASSO columns (same prices, same active minutes)."""
    m = f.picasso_afrr_up_eur_mwh.notna() & f.afrr4s_up_price.notna()
    d_price = (f.picasso_afrr_up_eur_mwh - f.afrr4s_up_price)[m].abs()
    d_share = (f.picasso_afrr_up_active_min / 15 - f.afrr4s_up_share).abs()
    return dict(up_price_mae=round(float(d_price.mean()), 4), up_price_p99=round(float(d_price.quantile(.99)), 4), share_mae=round(float(d_share.mean()), 4))


if __name__ == '__main__':
    f = build()
    print(f.shape, f.index.min(), f.index.max())
    print({k: (len(days(f, k)), int((f.split == k).sum())) for k in SPLITS})
    print('aFRR aggregation vs dataset 15-min:', check(f))
    print(f.groupby('split')[['nps_dayahead_eur_mwh', 'imbalance_eur_mwh', 'cap_mfrr_up_eur_mw_h', 'cap_afrr_up_eur_mw_h', 'solarpark_forecast_mw']].mean().round(2))
