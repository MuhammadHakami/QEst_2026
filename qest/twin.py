"""KratTrade 2026 digital twin: a year-long scenario simulation of Estonia's generation and storage assets, on one page.

Simulation (map): every wind turbine, solar plant and the Hertz 1 battery at its real position (qest.assets register,
  corrected by public records), on the Land Board orthophoto of Estonia with the 110-330 kV grid. A clock steps through
  2026 hour by hour: each wind plant runs its turbines' model on its ERA5 wind, each solar plant its rating on its grid
  cell's irradiance, and the battery follows the trading simulation's dispatch (qest.evaluate settlement traces) for the
  baseline and the two strategies with the most Q4 revenue. The national totals run beside Elering's measurements.
  Zoomed in, each asset's own close-up photo (0.3-0.6 m / px) appears in place with its automatic check.
Parks (three.js): the site orthophotos with the assets at their mapped / detected positions, turbines drawn per maker at
  the record's hub and rotor; rotors follow the ERA5 hub wind and direction of the clock's hour.
Results: 2026 day by day (8 Jan to 25 Oct): euros of the Chronos-2 + MILP baseline and the two strategies (qest.qdecide,
  official settlement, battery + hybrid). Q1-Q2 are a replay (activation tables fitted on them, knobs tuned on Q3);
  Q3 is validation, Q4 the test, scored once.
usage: .venv/bin/python -m qest.twin [out_dir]   -> <out_dir>/index.html, data.json, images (default results/figs/twin)
       .venv/bin/python -m qest.twin --traces    (re-runs the strategies keeping the battery traces; CUDA-Q)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from qest.fetch_ortho import WIND_3D
from qest.paths import RAW, RESULTS
from qest.assets import CHIP, PER, VERIFIED, imagery, in_estonia, register
from qest.paths import DERIVED
from qest.sites import BATTERY, PORTFOLIO_WIND, SOLAR, weather_sites, wind_key, wind_plants

STRATS = {'milp@chronos': 'Baseline · Chronos-2 + MILP', 'samples@chronos': 'Best sampled pattern (classical)',
          'qaoa@chronos': 'LR-QAOA, 24 qubits (quantum)', 'p_nqs@chronos': 'Distilled QPU, 12-qubit windows, no quantum calls (quantum-trained)'}
NQS = ('p_nqs@chronos',)  # rows live in days_*_qdecide_chronos_nqs.parquet (qest.qdecide keeps partitioned runs apart)


def day_rows(sp):
    fs = [RESULTS / f'days_{sp}_qdecide_chronos.parquet', RESULTS / f'days_{sp}_qdecide_chronos_nqs.parquet']
    return pd.concat([pd.read_parquet(f) for f in fs if f.exists()]).drop_duplicates(['strategy', 'day', 'unit'])  # both files carry the MILP rows
TEX = 2048
FAMILY = (('ge', 'ge'), ('enercon', 'enercon'), ('e-', 'enercon'), ('e7', 'enercon'), ('e8', 'enercon'), ('vestas', 'vestas'), ('v1', 'vestas'),
          ('v8', 'vestas'), ('nordex', 'nordex'), ('delta', 'nordex'), ('n1', 'nordex'), ('n-9', 'nordex'), ('n9', 'nordex'), ('wwd', 'winwind'),
          ('eleon', 'eleon'), ('3m116', 'eleon'))


def family(model):
    m = (model or '').lower()
    return next((f for k, f in FAMILY if m.startswith(k)), 'generic')


def r(x, n=1):
    return [None if not np.isfinite(v) else round(float(v), n) for v in np.asarray(x, float)]


# ------------------------------------------------------------------ map
def ortho_meta(k):
    return json.loads((RAW / f'ortho_{k}.json').read_text())


def b64(a):
    import base64
    return base64.b64encode(np.asarray(a, np.uint8).tobytes()).decode()


def q8(x, scale):
    """Values in [0, scale] -> uint8 0..250; missing -> 255."""
    x = np.asarray(x, float); out = np.clip(np.round(x / scale * 250), 0, 250); out[~np.isfinite(x)] = 255; return out


def img(kind, i):
    c = imagery()[kind].get(i)
    return None if c is None else dict(a=c['atlas'], c=c['cell'], b=[round(c['north'], 6), round(c['south'], 6), round(c['west'], 6), round(c['east'], 6)],
                                       ok=c['ok'], mpp=round(c['mpp'], 2), **{k: c[k] for k in ('offset_m', 'module_m2', 'share', 'mwp', 'note', 'compound_m2') if k in c})


LAT0, LON0 = 58.65, 24.9  # the 3D view's origin (twin.html L3)
KX, KZ = 111_320 * math.cos(math.radians(LAT0)), 110_574


def tables3d(Rg, F):
    """Every solar plant's tables for the 3D view, as local metres (int32) and the plant each belongs to (uint16).
    Where the plant's photo shows modules: one 4 m table per module cell found there. Otherwise its OSM outline filled with
    south-facing rows (6 m pitch, 4 m tables): built after the flight, a rooftop the colour rule misses, or unconfirmed."""
    from matplotlib.path import Path as MPath
    xs, zs, own, src = [], [], [], []
    for i, s in enumerate(Rg['solar']):
        c = imagery()['solar'].get(i) or {}; cx, cy = c.get('cells_px') or ([], [])
        if F['solar'][i]['seen'] and len(cx):
            lat = c['north'] + (c['south'] - c['north']) * np.asarray(cy); lon = c['west'] + (c['east'] - c['west']) * np.asarray(cx); src.append('photo')
        elif s['poly'] and len(s['poly']) > 2:
            P = np.asarray(s['poly']); la0, la1, lo0, lo1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
            gy, gx = np.meshgrid(np.arange(la0, la1, 6 / KZ), np.arange(lo0, lo1, 4 / (111_320 * math.cos(math.radians(la0)))), indexing='ij')
            pts = np.c_[gx.ravel(), gy.ravel()]; ok = MPath(P[:, ::-1]).contains_points(pts) if len(pts) else np.zeros(0, bool)
            lon, lat = (pts[ok, 0], pts[ok, 1]) if ok.any() else (np.array([P[:, 1].mean()]), np.array([P[:, 0].mean()]))
            src.append('outline')
        else:
            lat, lon = np.array([s['lat']]), np.array([s['lon']]); src.append('point')
        xs.append((lon - LON0) * KX); zs.append(-(lat - LAT0) * KZ); own.append(np.full(len(lat), i))
    X, Zz, O = np.concatenate(xs), np.concatenate(zs), np.concatenate(own)
    import base64
    return dict(n=int(len(X)), xz=base64.b64encode(np.c_[X, Zz].round().astype('<i4').tobytes()).decode(), own=base64.b64encode(O.astype('<u2').tobytes()).decode(), src=src)


def mapdata():
    Rg = register(); P = Rg['plants']; names = sorted(P, key=lambda k: -P[k]['mw']); idx = {k: i for i, k in enumerate(names)}
    plants = [dict(name=k, lat=round(P[k]['lat'], 4), lon=round(P[k]['lon'], 4), mw=P[k]['mw'], n=P[k]['n'], model=P[k]['model'] or 'not tagged',
                   rotor=P[k]['rotor'], hub=P[k]['hub'], start=P[k]['start'], operator=P[k]['operator'], ver=P[k]['ver'],
                   wsite=P[k]['wsite'], fam=family(P[k]['model']), portfolio=k == PORTFOLIO_WIND) for k in names]
    turbines = [dict(lat=round(t['lat'], 5), lon=round(t['lon'], 5), p=idx[t['plant']], mw=t['mw'], img=img('turbine', i), approx=not t['placed'])
                for i, t in enumerate(Rg['turbines'])]
    F = json.loads((DERIVED / 'assets_fleet.json').read_text())
    solar = [dict(lat=round(s['lat'], 5), lon=round(s['lon'], 5), name=s['name'], mw=s['mw'], mw_osm=s['mw_osm'], mw_img=s['mw_img'], seen=s['seen'],
                  cell=s['cell'], img=img('solar', i)) for i, s in enumerate(F['solar'])]
    lines, subs = [], []
    for e in json.loads((RAW / 'osm_grid.json').read_text())['elements']:
        t, g = e['tags'], e.get('geometry') or []
        if not g:
            continue
        v = max(int(x) for x in t.get('voltage', '0').split(';') if x.isdigit()) // 1000
        if t.get('power') == 'line':
            lines.append(dict(v=v, p=[c for q in g for c in (round(q['lat'], 4), round(q['lon'], 4))]))
        else:
            la, lo = sum(q['lat'] for q in g) / len(g), sum(q['lon'] for q in g) / len(g)
            if in_estonia(la, lo):
                subs.append([round(la, 4), round(lo, 4), t.get('name'), v, [c for q in g for c in (round(q['lat'], 5), round(q['lon'], 5))]])
    cables = [dict(v=max(int(x) for x in e['tags'].get('voltage', '0').split(';') if x.isdigit()) // 1000, name=e['tags'].get('name'),
                   p=[c for q in e['geometry'] for c in (round(q['lat'], 5), round(q['lon'], 5))]) for e in json.loads((RAW / 'osm_cables.json').read_text())['elements'] if e.get('geometry')]
    B = json.loads((RAW / 'osm_border.json').read_text())['elements'][0]
    border = [[c for q in m['geometry'][::2] + m['geometry'][-1:] for c in (round(q['lat'], 3), round(q['lon'], 3))]
              for m in B['members'] if m['role'] == 'outer' and m.get('geometry')]
    W = [[k, v['lat'], v['lon']] for k, v in weather_sites().items()]
    db = json.loads((DERIVED / 'assets_detected.json').read_text())['battery']; cc = db['compound_center']  # what qest.detect found in the photo
    bat = []
    for i, b in enumerate(Rg['battery']):
        x = dict(name=b['name'], mw=b['mw'], mwh=b['mwh'], status=b['status'], dispatch=b['dispatch'], src=b['src'], img=img('battery', i) if b['dispatch'] else None)
        if b['dispatch']:  # Hertz 1: the compound and container field found in the photo, the operator's 54 containers
            x.update(lat=round(cc[0], 6), lon=round(cc[1], 6), field=[round(v, 6) for v in db['center']], axis=round(db['axis_deg'], 1), n=db['containers'],
                     outline=[round(v, 6) for ll in db['outline'] for v in ll])
        else:  # 20 ft blocks of ~3.4 MWh (the same density as Hertz 1: 200 MWh in 54 containers would be ~3.7)
            x.update(lat=round(b['lat'], 6), lon=round(b['lon'], 6), field=[round(b['lat'], 6), round(b['lon'], 6)], axis=0.0, n=int(round(b['mwh'] / 3.7)), outline=[])
        bat.append(x)
    other = [dict(name=o['name'], source=o['source'], mw=o['mw'], lat=round(o['lat'], 5), lon=round(o['lon'], 5),
                  poly=[c for q in (o['poly'] or [])[::max(1, len(o['poly'] or []) // 60)] for c in (round(q[0], 5), round(q[1], 5))]) for o in Rg['other']]
    # grid connection: the nearest 110-330 kV substation (an assumption: connection points are not public), except Hertz 1,
    # whose 330 kV cable to Elering's Kiisa substation is public (qest.sites.BATTERY)
    from qest.sites import _km
    def conn(lat, lon, name=None):
        k = next((i for i, x in enumerate(subs) if name and name in (x[2] or '') and x[3] >= 220), None)
        if k is None:
            k = min(range(len(subs)), key=lambda i: _km((lat, lon), (subs[i][0], subs[i][1])))
        return [k, round(_km((lat, lon), (subs[k][0], subs[k][1])), 2)]
    for x in plants:
        x['conn'] = conn(x['lat'], x['lon'])
    for x in solar:
        x['conn'] = conn(x['lat'], x['lon']) if (x['mw'] or 0) > 0 else None
    for x in bat:
        x['conn'] = conn(x['lat'], x['lon'], 'Kiisa')
    return dict(bounds=ortho_meta('estonia'), img='estonia.jpg', plants=plants, turbines=turbines, solar=solar, battery=bat, other=other, lines=lines, subs=subs, cables=cables,
                tables=tables3d(Rg, F),
                border=border, weather=W, fits=F['fits'], chip=[CHIP, PER], atlases=sorted({c['atlas'] for d in imagery().values() for c in d.values()}),
                pihlaka=dict(name=SOLAR['name'], lat=SOLAR['lat'], lon=SOLAR['lon']))


def sim():
    """The hourly scenario: per wind plant capacity factor, per solar cell per-MW output, fleet vs Elering, battery traces."""
    from qest.assets import hours
    H = hours(); S = pd.read_parquet(DERIVED / 'assets_sim.parquet').reindex(H); Rg = register(); P = Rg['plants']
    names = sorted(P, key=lambda k: -P[k]['mw']); F = json.loads((DERIVED / 'assets_fleet.json').read_text())['fits']
    cells = sorted({s['cell'] for s in json.loads((DERIVED / 'assets_fleet.json').read_text())['solar'] if s['mw'] > 0})
    wcf = np.stack([q8(S[f'wind|{k}'] / P[k]['mw'], 1) for k in names], 1)  # hour-major: [h, plant]
    placed = list(range(len(Rg['turbines'])))  # the map's turbine order: every register turbine
    def wdir(k):
        w = json.loads((RAW / f'era5_{k}.json').read_text())['hourly']
        return pd.Series(w['wind_direction_100m'], index=pd.to_datetime(w['time']).tz_localize('UTC')).reindex(H)
    wdp = np.stack([q8(wdir(P[k]['wsite']), 360) for k in names], 1)  # each plant's ERA5 wind direction: the 3D rotors face it
    tcf = np.stack([q8(S[f'turbine|{i}'] / Rg['turbines'][i]['mw'], 1) for i in placed], 1)  # each turbine's own output inside the wakes
    spu = np.stack([q8(S[f'solar|{c}'], 1) for c in cells], 1)
    tp = RESULTS / 'twin_traces.parquet'  # from `python -m qest.twin --traces`; without it the battery shows as not traded
    T = pd.read_parquet(tp) if tp.exists() else pd.DataFrame(dict(strategy=[], unit=[], t=pd.to_datetime([], utc=True), deliv=[], soc=[]))
    T = T[T.unit == 'battery'].copy(); T['h'] = pd.to_datetime(T.t).dt.floor('h')
    bat = {}
    for st in STRATS:
        x = T[T.strategy == st].groupby('h').agg(soc=('soc', 'last'), deliv=('deliv', 'mean')).reindex(H)
        bat[st] = dict(soc=b64(q8(x.soc, 200)), dl=b64(q8(x.deliv + 100, 200)))
    x = T[T.strategy == 'oracle'].groupby('h').agg(soc=('soc', 'last'), deliv=('deliv', 'mean')).reindex(H)
    hind = dict(soc=b64(q8(x.soc, 200)), dl=b64(q8(x.deliv + 100, 200)))  # the hindsight plan (perfect foresight), `--traces oracle`
    from qest.data import frame
    f = frame(); E = f[['ee_real_production', 'ee_real_consumption', 'nps_dayahead_eur_mwh', 'imbalance_eur_mwh']].groupby(f.index.floor('h')).mean().reindex(H)
    loc = H.tz_convert('Europe/Tallinn'); day = pd.Series(loc.date, index=H)
    dmean = lambda x: x.groupby(day.values).mean()
    wind_d, sol_d, px_d, imb_d = dmean(S['fleet|wind']), dmean(S['elering|solar']), dmean(E.nps_dayahead_eur_mwh), E.imbalance_eur_mwh.abs().groupby(day.values).max()
    on = f.solarpark_forecast_devices.gt(0).idxmax().tz_convert('Europe/Tallinn').date()
    ev = [('2026-02-03', 'Hertz 1 opens (Kiisa)'), (str(on), 'Pihlaka: first devices online'), ('2026-03-29', 'Clocks forward (EEST)'), ('2026-10-25', 'Clocks back · dataset ends'),
          (str(wind_d.idxmax()), f'Windiest day: fleet {wind_d.max() * F["wind"]["a"] + F["wind"]["b"]:.0f} MW mean'),
          (str(sol_d.idxmax()), f'Sunniest day: Estonia {sol_d.max():.0f} MW mean solar'),
          (str(px_d.idxmax()), f'Highest day-ahead: {px_d.max():.0f} EUR/MWh mean'), (str(px_d.idxmin()), f'Lowest day-ahead: {px_d.min():.0f} EUR/MWh mean'),
          (str(imb_d.idxmax()), f'Imbalance spike: {imb_d.max():.0f} EUR/MWh')]
    w, sl = F['wind'], F['solar']
    return dict(t0=str(H[0]), n=len(H), plants=names, wcf=b64(wcf), tcf=b64(tcf), wd=b64(wdp), nt=len(placed), cells=cells, spu=b64(spu), bat=bat, hind=hind,
                fleet=dict(wind=r(w['a'] * S['fleet|wind'] + w['b'], 0), solar=r(sl['k'] * S['fleet|solar'], 0),
                           wind_raw=r(S['fleet|wind'], 0), solar_raw=r(S['fleet|solar'], 0),
                           ren=r(S['elering|renewable'], 0), esol=r(S['elering|solar'], 0), prod=r(E.ee_real_production, 0), cons=r(E.ee_real_consumption, 0)),
                events=sorted(ev))


# ------------------------------------------------------------------ replay
def replay():
    from qest import market
    from qest.data import frame
    from qest.power import generation
    R = pd.concat([day_rows(s).assign(split=s) for s in ('train', 'val', 'test')])
    R = R[R.strategy.isin(STRATS)].copy(); R['day'] = pd.to_datetime(R.day)
    have = set(R.strategy); STRATS_ = {k: v for k, v in STRATS.items() if k in have}
    for k in STRATS:
        if k not in have:
            print('replay: no rows yet for', k, '(run twin --traces', k, ')')
    um = {u.name: u for u in market.portfolio(['battery', 'hybrid'], official=True)}
    R['month'] = R.day.dt.to_period('M')
    tar = R.groupby(['strategy', 'unit', 'month']).apply(lambda x: market.tariff(x.import_mwh.sum(), x.export_mwh.sum(), um[x.name[1]]), include_groups=False)
    comp = ['cap', 'da', 'mfrr', 'afrr', 'imbalance', 'penalty', 'leftover', 'cycle', 'tariff_mtu', 'total']
    D = R.groupby(['strategy', 'day'])[comp].sum()
    last = R.groupby('month').day.max()
    for (s, u, m), x in tar.items():  # monthly tariff booked on the month's last traded day
        D.loc[(s, last[m]), 'total'] += x
    days = sorted(R.day.unique()); split = R.drop_duplicates('day').set_index('day').split
    soc = R[R.unit == 'battery'].set_index(['strategy', 'day']).soc_end
    out = dict(days=[str(d.date()) for d in days], split=[split[d] for d in days], labels=STRATS_, strat={})
    for s in STRATS_:
        x = D.loc[s].reindex(days)
        out['strat'][s] = {c: r(x[c], 0) for c in ('total', 'cap', 'da', 'mfrr', 'afrr', 'imbalance')} | dict(soc=r(soc.loc[s].reindex(days), 1))
    O = pd.read_parquet(RESULTS / 'twin_oracle_days.parquet'); O['day'] = pd.to_datetime(O.day)
    out['hind'] = r(O.groupby('day').total.sum().reindex(days), 0)  # hindsight euros a day (monthly tariff not booked: official mode has none)
    rp = RESULTS / 'twin_reach_days.parquet'  # the reachable optimum: perfect DA and capacity prices, Chronos-2 activations (strategies.oracle_prices)
    if rp.exists():
        Q = pd.read_parquet(rp); Q['day'] = pd.to_datetime(Q.day); out['reach'] = r(Q.groupby('day').total.sum().reindex(days), 0)
    else:
        out['reach'] = None; print('replay: no results/twin_reach_days.parquet yet (scratch script reach.py / strategies.oracle_prices)')
    # hourly context (UTC hours from 1 Jan)
    f = frame(); g = generation(); H = f.index.floor('h')
    hr = f.assign(pv=g.pv_actual, wind=g.wind_actual).groupby(H)[['nps_dayahead_eur_mwh', 'imbalance_eur_mwh', 'pv', 'wind']].mean()
    t = pd.date_range(hr.index[0], hr.index[-1], freq='h'); hr = hr.reindex(t)

    def era(k, col):
        w = json.loads((RAW / f'era5_{k}.json').read_text())['hourly']
        return pd.Series(w[col], index=pd.to_datetime(w['time']).tz_localize('UTC')).reindex(t)
    ws = {k: r(era(k, 'wind_speed_100m'), 1) for k in [wind_key(n) for n in (PORTFOLIO_WIND, *WIND_3D)]}  # m/s
    wd = {k: r(era(k, 'wind_direction_100m'), 0) for k in ws}
    from qest.evaluate import forecasts
    FC = pd.concat([forecasts(sp, 'chronos') for sp in ('train', 'val', 'test')]); FC = FC.groupby(FC.index.floor('h')).mean().reindex(t)  # what the strategies saw at gate A
    out['hourly'] = dict(t0=str(t[0]), n=len(t), da=r(hr.nps_dayahead_eur_mwh), imb=r(hr.imbalance_eur_mwh), pv=r(hr.pv), wind=r(hr.wind),
                         da_q10=r(FC.da_q10), da_q50=r(FC.da_q50), da_q90=r(FC.da_q90), imb_q50=r(FC.imb_q50),
                         pv_d1=r(g.pv_d1.groupby(H).mean().reindex(t)), wind_d1=r(g.wind_d1.groupby(H).mean().reindex(t)),
                         ghi=r(era('solar_pihlaka', 'shortwave_radiation'), 0), cloud=r(era('solar_pihlaka', 'cloud_cover'), 0),
                         temp=r(era('battery_kiisa', 'temperature_2m')), ws=ws, wd=wd)
    return out


def main(out=None):
    out = Path(out or RESULTS / 'figs' / 'twin'); out.mkdir(parents=True, exist_ok=True)
    Image.open(RAW / 'ortho_estonia.jpg').save(out / 'estonia.jpg', quality=80)
    import shutil
    data = dict(map=mapdata(), replay=replay(), sim=sim())
    for a in data['map']['atlases']:
        shutil.copy(DERIVED / f'assets_chip_{a}.jpg', out / f'chips_{a}.jpg')
    shutil.copytree(Path(__file__).parent / 'vendor', out / 'vendor', dirs_exist_ok=True)  # three.js r182 + OrbitControls, the water twin's copy (MIT, THREE-LICENSE)
    (out / 'data.json').write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))
    page = (Path(__file__).parent / 'twin.html').read_text()
    (out / 'page.html').write_text(page)  # the fragment the Artifact publisher wraps
    (out / 'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                                    '<style>body{margin:0}img{max-width:100%}</style></head><body>' + page + '</body></html>')  # served locally (run.sh serve)
    print('wrote', out, {k: len(json.dumps(v)) // 1000 for k, v in data.items()}, 'kB')
    return data


def _check(d):
    R = d['replay']; m = d['map']
    assert len(R['days']) == len(R['split']) and all(len(v['total']) == len(R['days']) for v in R['strat'].values())
    assert R['hourly']['n'] == len(R['hourly']['da']) and all(len(v) == R['hourly']['n'] for v in R['hourly']['ws'].values())
    assert m['bounds']['south'] < SOLAR['lat'] < m['bounds']['north'] and m['bounds']['west'] < 28.2 < m['bounds']['east']
    import base64
    S = d['sim']; assert len(base64.b64decode(S['wcf'])) == S['n'] * len(S['plants']) and len(base64.b64decode(S['spu'])) == S['n'] * len(S['cells'])
    assert all(s['cell'] in S['cells'] for s in m['solar'] if s['mw'] > 0) and len(S['fleet']['ren']) == S['n']
    # the Q4 slice of the replay equals qest.qdecide's Q4 result for the baseline (EUR 1,812,548 with the tariff)
    q4 = sum(x for x, s in zip(R['strat']['milp@chronos']['total'], R['split']) if s == 'test')
    assert abs(q4 - 1_812_548) < 25, q4  # per-day totals are rounded to whole euros
    assert family('Delta4000') == 'nordex' and family('E-82') == 'enercon' and family('GE 2.5-100') == 'ge' and family('3M116') == 'eleon'
    print('twin check ok: Q4 baseline', round(q4))


def run_traces(names=None):
    """Re-run strategies over 2026 keeping each storage unit's MTU trace -> results/twin_traces.parquet (rows of other
    strategies kept). The three STRATS need CUDA-Q (CUDA_VISIBLE_DEVICES=1) and must reproduce the qdecide runs to the
    euro; 'oracle' is the hindsight plan (perfect foresight of prices, activations and output, the same settlement) and
    also writes its daily euros to results/twin_oracle_days.parquet."""
    from qest import market, qdecide
    from qest.evaluate import run
    names = names or list(STRATS); cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=cfg.official)
    qdecide.register('chronos', [n.split('@')[0] for n in names if '@' in n and n.split('@')[0] not in ('milp', *qdecide.SOLVERS)])
    rows, days = [], []
    for sp in ('test', 'val', 'train'):  # the shortest split first: a failure shows early
        T = {}; R, _ = run(sp, names, units, cfg, traces=T); days.append(R.assign(split=sp))
        mine = [n for n in names if n in STRATS]
        if mine:
            old = day_rows(sp); old = old[old.strategy.isin(mine)].groupby('strategy').total.sum()
            print(sp, 'max |total - qdecide run| EUR', float((R[R.strategy.isin(mine)].groupby('strategy').total.sum() - old).abs().max()), flush=True)
        new = [n for n in names if n in NQS]
        if new:  # the distilled strategies' day rows are written here (one pass over 2026 instead of two)
            p = RESULTS / f'days_{sp}_qdecide_chronos_nqs.parquet'
            old = pd.read_parquet(p).query('strategy not in @new') if p.exists() else pd.DataFrame()
            pd.concat([old, R[R.strategy.isin(new)]]).to_parquet(p)
        for (st, d), us in T.items():
            for u, x in us.items():
                rows.append(pd.DataFrame(dict(strategy=st, unit=u, t=x['idx'], deliv=x['deliv'].round(3), soc=x['soc'].round(3))))
    new = pd.concat(rows); p = RESULTS / 'twin_traces.parquet'
    if p.exists():
        new = pd.concat([pd.read_parquet(p).query('strategy not in @names'), new])
    new.to_parquet(p)
    if 'oracle' in names:
        D = pd.concat(days); D[D.strategy == 'oracle'].to_parquet(RESULTS / 'twin_oracle_days.parquet')


if __name__ == '__main__':
    if '--traces' in sys.argv:
        run_traces([a for a in sys.argv[sys.argv.index('--traces') + 1:] if not a.startswith('-')] or None)
    else:
        _check(main(sys.argv[1] if len(sys.argv) > 1 else None))
