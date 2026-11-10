"""Baltic Transparency Dashboard (Elering, AST, Litgrid): the balancing data the dataset lacks, for a faithful settlement.

What we need and why (research/gap_to_hindsight.md):
  normal_activations_mfrr   activated mFRR MW per MTU, direction, area: caps how much of our 100 MW can be activated at all
  activations_afrr          activated aFRR MW: the same cap for aFRR (Baltic demand is ~90-120 MW an hour in total)
  procured_reserves, reserves_demand, price_procured_reserves   the capacity market's volumes and prices per product / direction:
                            our won MW cannot exceed the demand, and 100 MW is a large share of it
  mfrr_bid_volumes, volumes_afrr   market depth behind the merit order
  imbalance_volumes_v2, direction_of_balancing_v2   the system state the T-25 gate can see

Two ways in, same output (data/raw/btd/<report>.parquet, long format: start, end, area, direction, product, value):
  1. the open API through the `baltic` package (pip name baltic-py; https://api-baltic.transparency-dashboard.eu/api/v1/export).
     From our network Cloudflare answers 403 to every client (9 Nov 2026), so this path needs a network the dashboard admits.
  2. files from the dashboard's Download Center (CSV or XLSX, any period, UTC or local) dropped into data/raw/btd/inbox/
     named <report>*.csv|xlsx; `--inbox` parses them. This is the browser / VPN route.

usage: .venv/bin/python -m qest.fetch_btd [--api START END] [--inbox] [--check]
"""
import re
import sys
from pathlib import Path

import pandas as pd

from qest.paths import RAW

BTD = RAW / 'btd'; INBOX = BTD / 'inbox'
REPORTS = ('normal_activations_mfrr', 'activations_afrr', 'procured_reserves', 'reserves_demand', 'price_procured_reserves',
           'mfrr_bid_volumes', 'volumes_afrr', 'imbalance_volumes_v2', 'direction_of_balancing_v2', 'local_marginal_price_mfrr')
COLS = ['start', 'end', 'area', 'direction', 'product', 'value']


def _tidy(df, report):
    """Any of the dashboard's layouts -> the long format. Column names are matched loosely (the exports vary by report)."""
    d = df.copy(); d.columns = [re.sub(r'\W+', '_', str(c)).strip('_').lower() for c in d.columns]
    t0 = next((c for c in d.columns if c in ('start', 'start_time', 'from', 'timestamp', 'datetime', 'time')), None)
    if t0 is None:
        raise ValueError(f'{report}: no start column in {list(d.columns)[:10]}')
    t1 = next((c for c in d.columns if c in ('end', 'end_time', 'to')), None)
    out = pd.DataFrame({'start': pd.to_datetime(d[t0], utc=True), 'end': pd.to_datetime(d[t1], utc=True) if t1 else pd.NaT})
    for k, names in (('area', ('area', 'country', 'zone', 'bidding_zone', 'lfc_area')), ('direction', ('direction', 'dir')), ('product', ('product', 'reserve_type', 'type'))):
        c = next((c for c in d.columns if c in names), None); out[k] = d[c].astype(str) if c else ''
    vcol = next((c for c in d.columns if c in ('value', 'volume', 'price', 'mw', 'quantity')), None)
    if vcol is None:  # wide layout: one column per series, e.g. 'ee_up', 'lv_down'
        num = [c for c in d.columns if c not in (t0, t1) and pd.api.types.is_numeric_dtype(d[c])]
        rows = []
        for c in num:
            m = re.match(r'(?P<area>ee|lv|lt|baltic)?_?(?P<dir>up|down|upward|downward)?', c)
            rows.append(pd.DataFrame({'start': out.start, 'end': out.end, 'area': (m and m.group('area') or '').upper(),
                                      'direction': (m and (m.group('dir') or '')).replace('ward', ''), 'product': c, 'value': d[c].astype(float)}))
        return pd.concat(rows, ignore_index=True)[COLS]
    out['value'] = pd.to_numeric(d[vcol], errors='coerce'); return out[COLS]


def from_api(start, end, reports=REPORTS):
    import baltic
    BTD.mkdir(parents=True, exist_ok=True)
    for r in reports:
        try:
            e = baltic.export(r, start=start, end=end); df = pd.DataFrame(e.records())
        except Exception as ex:
            print(r, 'failed:', repr(ex)[:120]); continue
        if len(df):
            _tidy(df, r).to_parquet(BTD / f'{r}.parquet'); print(r, len(df), 'records')


def from_inbox():
    BTD.mkdir(parents=True, exist_ok=True); INBOX.mkdir(exist_ok=True); n = 0
    for f in sorted(INBOX.iterdir()):
        r = next((r for r in REPORTS if f.name.startswith(r)), None)
        if r is None or f.suffix.lower() not in ('.csv', '.xlsx'):
            continue
        df = pd.read_csv(f, sep=None, engine='python') if f.suffix.lower() == '.csv' else pd.read_excel(f)
        t = _tidy(df, r); p = BTD / f'{r}.parquet'
        if p.exists():
            t = pd.concat([pd.read_parquet(p), t]).drop_duplicates(['start', 'area', 'direction', 'product'], keep='last')
        t.sort_values('start').to_parquet(p); print(f.name, '->', p.name, len(t), 'rows'); n += 1
    print(n, 'files parsed') if n else print('inbox empty: drop the Download Center files into', INBOX)


def activated_mw(report='normal_activations_mfrr', area='EE'):
    """Activated MW per MTU and direction for one area (UTC index), or None when the data is not there yet."""
    p = BTD / f'{report}.parquet'
    if not p.exists():
        return None
    d = pd.read_parquet(p); d = d[d.area.str.upper().isin([area, '']) & d.direction.isin(['up', 'down'])]
    return d.pivot_table(index='start', columns='direction', values='value', aggfunc='sum')


def _check():
    wide = pd.DataFrame({'Start time': ['2026-08-14T00:00Z', '2026-08-14T00:15Z'], 'EE up': [12.5, 0.0], 'EE down': [0.0, 40.0], 'LV up': [5.0, 5.0]})
    t = _tidy(wide, 'normal_activations_mfrr'); assert set(t.direction) == {'up', 'down'} and t.value.sum() == 62.5, t
    long = pd.DataFrame({'start': ['2026-08-14T00:00Z'], 'end': ['2026-08-14T00:15Z'], 'area': ['EE'], 'direction': ['up'], 'value': [7]})
    t = _tidy(long, 'activations_afrr'); assert t.value.iloc[0] == 7 and t.area.iloc[0] == 'EE'
    print('fetch_btd check ok: wide and long exports tidy; data present:', [r for r in REPORTS if (BTD / f'{r}.parquet').exists()] or 'none yet')


if __name__ == '__main__':
    if '--check' in sys.argv:
        _check()
    elif '--api' in sys.argv:
        i = sys.argv.index('--api'); from_api(sys.argv[i + 1], sys.argv[i + 2])
    elif '--inbox' in sys.argv:
        from_inbox()
    else:
        print(__doc__)
