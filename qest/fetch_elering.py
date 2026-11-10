"""Estonia's power system in 2026 from Elering's open dashboard API (keyless), into data/raw/elering_system.json.

/api/system/with-plan gives, every 15 min, the 'real' measured values (production, consumption, renewable production,
solar production, system balance, AC balance) and the 'plan' (Elering's planned production/consumption). Real values
are known only after the fact; the plan is the part a trader may use before a gate (qest.features takes care of that).
usage: .venv/bin/python -m qest.fetch_elering
"""
import json
import time
import urllib.request

import pandas as pd

from qest.paths import RAW


def get(u):
    for k in range(5):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'qest-2026-hackathon/0.1'}), timeout=300).read())
        except Exception as e:
            print('retry', k, e, flush=True); time.sleep(20 * (k + 1))
    raise RuntimeError(u)


def main():
    p = RAW / 'elering_system.json'
    if p.exists():
        return
    out = dict(real=[], plan=[])
    for a in pd.date_range('2025-12-31', '2026-10-27', freq='14D', tz='UTC'):  # two weeks a request
        b = min(a + pd.Timedelta('14D'), pd.Timestamp('2026-10-27', tz='UTC'))
        d = get(f'https://dashboard.elering.ee/api/system/with-plan?start={a:%Y-%m-%dT%H:%M:%S.000Z}&end={b:%Y-%m-%dT%H:%M:%S.000Z}')['data']
        for k in out: out[k] += d.get(k) or []
        print(a.date(), len(d.get('real') or []), len(d.get('plan') or []), flush=True); time.sleep(1)
    p.write_text(json.dumps(out))


if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True); main()
