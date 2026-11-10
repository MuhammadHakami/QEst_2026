"""Nord Pool day-ahead prices of the four coupled zones (EE, FI, LV, LT) from Elering's open dashboard API, 15-min MTUs.

Why: Estonia's price is set by the coupling (Estlink to FI, the EE-LV border), so the neighbours' D-1 prices are the
missing cross-zonal context for the gate-A price forecast (qest.chronos, kind 'chronosx'). Same information set as the EE
price: known through D-1 at the gate. Written to data/raw/nps_prices.parquet (UTC index, columns ee, fi, lv, lt in EUR/MWh).

usage: .venv/bin/python -m qest.fetch_nps [START END]   (defaults 2025-10-01 .. 2026-10-27)
"""
import json
import sys
import time
import urllib.request

import pandas as pd

from qest.paths import RAW

URL = 'https://dashboard.elering.ee/api/nps/price?start={a:%Y-%m-%dT%H:%M:%S}.000Z&end={b:%Y-%m-%dT%H:%M:%S}.000Z'
OUT = RAW / 'nps_prices.parquet'


def fetch(start='2025-10-01', end='2026-10-27'):
    a, end = pd.Timestamp(start), pd.Timestamp(end); frames = []
    while a < end:
        b = min(a + pd.Timedelta(days=31), end)
        d = json.loads(urllib.request.urlopen(urllib.request.Request(URL.format(a=a, b=b), headers={'User-Agent': 'krattrade-qest'}), timeout=60).read())['data']
        cols = {z: pd.Series({pd.Timestamp(r['timestamp'], unit='s', tz='UTC'): r['price'] for r in d.get(z, [])}) for z in ('ee', 'fi', 'lv', 'lt')}
        frames.append(pd.DataFrame(cols)); print(a.date(), b.date(), {z: len(v) for z, v in cols.items()}, flush=True); a = b; time.sleep(.5)
    P = pd.concat(frames); P = P[~P.index.duplicated()].sort_index(); OUT.parent.mkdir(parents=True, exist_ok=True); P.to_parquet(OUT); return P


def prices():
    return pd.read_parquet(OUT) if OUT.exists() else fetch()


if __name__ == '__main__':
    P = fetch(*sys.argv[1:3]) if len(sys.argv) > 2 else fetch()
    from qest.data import frame
    f = frame(); j = f[['nps_dayahead_eur_mwh']].join(P.ee, how='inner')
    print('rows', len(P), P.index.min(), P.index.max(), '| EE vs dataset: max |diff|', float((j.nps_dayahead_eur_mwh - j.ee).abs().max()).__round__(2), 'on', len(j), 'MTUs')
