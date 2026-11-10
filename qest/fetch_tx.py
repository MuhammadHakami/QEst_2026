"""Elering open API: cross-border flows, transfer capacities and planned trade, the Baltic-block drivers of balancing activations.

Why: the TSO activates aFRR / mFRR against the LFC block's imbalance, which is shaped by what crosses the borders: whether the
Estlink import is at its limit (no room to balance from Finland), the Latvian flow, and the hourly steps of the scheduled
trade. None of this is in the organisers' dataset. Endpoints (dashboard.elering.ee/v3/api-docs, open, no key):
  transmission/cross-border            realised flows, 15 min: finland, latvia, estlink_1, estlink_2 (MW, + = import)  -> group D-2 (delivery-day data)
  transmission/cross-border-capacity   in/out ATC and NTC per border (FI, LV), 15 min                                 -> group D-1 (published before the DA auction)
  transmission/cross-border-planned-trade  scheduled trade per border after the DA / ID auctions                        -> group D-2 for the day plan (D's schedule is fixed after 13:00)
Written to data/raw/elering_tx.parquet (UTC 15-min index). Information timing is enforced where the columns are consumed
(qest.actfc), not here.

usage: .venv/bin/python -m qest.fetch_tx [START END]   (defaults 2025-10-01 .. 2026-10-27)
"""
import json
import sys
import time
import urllib.request

import pandas as pd

from qest.paths import RAW

API = 'https://dashboard.elering.ee/api/transmission/{ep}?start={a:%Y-%m-%dT%H:%M:%S}.000Z&end={b:%Y-%m-%dT%H:%M:%S}.000Z'
OUT = RAW / 'elering_tx.parquet'
H = {'User-Agent': 'krattrade-qest'}


def _get(ep, a, b):
    return json.loads(urllib.request.urlopen(urllib.request.Request(API.format(ep=ep, a=a, b=b), headers=H), timeout=90).read())['data']


def _frame(rows, prefix):
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows); d.index = pd.to_datetime(d.pop('timestamp'), unit='s', utc=True); d = d[~d.index.duplicated()].sort_index(); return d.add_prefix(prefix)


def fetch(start='2025-10-01', end='2026-10-27'):
    a, end = pd.Timestamp(start), pd.Timestamp(end); parts = []
    while a < end:
        b = min(a + pd.DateOffset(days=14), end)
        flow = _frame(_get('cross-border', a, b), 'flow_')
        cap = _get('cross-border-capacity', a, b); caps = pd.concat([_frame(v, f'cap_{k.lower()}_') for k, v in cap.items()], axis=1) if isinstance(cap, dict) else pd.DataFrame()
        plan = _get('cross-border-planned-trade', a, b); plan = _frame(plan, 'plan_') if isinstance(plan, list) else pd.concat([_frame(v, f'plan_{k.lower()}_') for k, v in plan.items()], axis=1)
        d = pd.concat([flow, caps, plan], axis=1); parts.append(d); print(a.date(), b.date(), len(d), 'rows', list(d.columns)[:4], '...', flush=True); a = b; time.sleep(.3)
    D = pd.concat(parts); D = D[~D.index.duplicated()].sort_index(); D = D.loc[:, D.notna().any()]
    D15 = D.resample('15min').mean().interpolate(limit=3)  # capacities come hourly or on change; flows and plans 15-min
    OUT.parent.mkdir(parents=True, exist_ok=True); D15.to_parquet(OUT); return D15


def tx():
    return pd.read_parquet(OUT) if OUT.exists() else fetch()


if __name__ == '__main__':
    D = fetch(*sys.argv[1:3]) if len(sys.argv) > 2 else fetch()
    pd.set_option('display.width', 220); print('rows', len(D), D.index.min(), D.index.max()); print(D.notna().mean().round(2).to_string()); print(D.describe().T[['mean', 'min', 'max']].round(0).to_string())
