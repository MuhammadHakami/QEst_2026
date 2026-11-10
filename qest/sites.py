"""Where every asset stands, and how each location was established.

SOLAR: the dataset's 42.3 MW park (4 devices, online March-April 2026) gives no location. Its forecast's cloud pattern
  (hourly anomalies, Apr-Oct) was correlated with ERA5 irradiance on a 0.25 deg grid over Estonia (`locate_solar`):
  the peak (r = 0.75) is north Estonia at 59.25-59.5 N, 25.25-25.5 E. There OSM has two newly mapped solar plants
  (ways 1568593429/30, 59.32 N 25.31 E) by Kehra: the Pihlaka park (KC Solar2, 45 MW, EBRD-financed 2024, built 2025),
  whose commissioning matches the dataset's devices coming online. Its D-1 noon timing alone gives 23.5 E (+-1, tracker
  geometry blurs it), consistent only with the west-to-central band, so the cloud match decides.
BATTERY: the toy's 100 MW / 200 MWh is Estonia's real Hertz 1 park (Evecon, Corsica Sole, Mirova; in operation
  3 Feb 2026), connected by 330 kV cable at Elering's Kiisa substation (OSM way 34935436, 59.2431 N 24.7007 E).
WIND: every mapped Estonian wind plant and turbine (OSM): rated power, rotor and hub per turbine where mapped, else the
  plant's output split over its turbines with the model's datasheet rotor. The portfolio's park is PORTFOLIO_WIND.
"""
import json
import math
import re
from functools import lru_cache

from qest.paths import RAW

SOLAR = dict(key='solar_pihlaka', name='Pihlaka solar park (Kehra)', lat=59.322, lon=25.307, mw=42.3, devices=4,
             evidence='ERA5 cloud-pattern match r=0.75 (qest.sites.locate_solar) + OSM ways 1568593429/30')
BATTERY = dict(key='battery_kiisa', name='Hertz 1 battery park (Kiisa)', lat=59.2431, lon=24.7007, mw=100.0, mwh=200.0,
               evidence='Kiisa 330 kV substation, OSM way 34935436; in operation 3 Feb 2026')
PORTFOLIO_WIND = 'Paldiski tuulepark'  # 45 MW on the north coast, the size of the solar park; any key of wind_plants()

# datasheet rotor (m) / hub (m) by OSM model string when the turbine node carries none
MODELS = {'delta4000': (163, 159), 'n163': (163, 159), 'wwd3': (100, 100), '3m116': (116, 122), 'e-70': (71, 85), 'e70': (71, 85),
          'v136': (136, 132), 'n-90': (90, 80), 'n90': (90, 80), 'e-82': (82, 78), 'e82': (82, 78), 'e-40': (44, 63), 'e-48': (48, 65)}


def _mw(v):
    m = re.match(r'([\d.]+)\s*(kW|MW|GW)?', (v or '').replace(',', '.'))
    if not m:
        return None
    x = float(m.group(1)); return {'kW': x / 1e3, 'GW': x * 1e3}.get(m.group(2), x)


def _km(a, b):
    return math.hypot((a[0] - b[0]) * 111.2, (a[1] - b[1]) * 111.2 * math.cos(math.radians(a[0])))


@lru_cache(None)
def wind_plants():
    """{plant name: dict(lat, lon, mw, start, turbines=[dict(lat, lon, mw, rotor, hub, model)])}: turbines go to the nearest
    plant centre within 4 km; a plant with no mapped turbines gets ceil(mw / 3) generic 3 MW, 100 m units at its centre."""
    E = json.load(open(RAW / 'osm_wind.json'))['elements']
    plants = {}
    for e in E:
        t = e.get('tags', {})
        if t.get('power') == 'plant' and e.get('center'):
            plants[t.get('name') or f"plant {e['id']}"] = dict(lat=e['center']['lat'], lon=e['center']['lon'], mw=_mw(t.get('plant:output:electricity')),
                                                               start=t.get('start_date'), operator=t.get('operator'), turbines=[])
    for e in E:
        t = e.get('tags', {})
        if e['type'] != 'node' or t.get('power') != 'generator':
            continue
        p = (e['lat'], e['lon']); name, d = min(((k, _km(p, (v['lat'], v['lon']))) for k, v in plants.items()), key=lambda x: x[1])
        if d > 4:
            continue
        model = (t.get('model') or t.get('generator:model') or '').lower().replace(' ', '')
        rotor, hub = (float(t['rotor:diameter']) if t.get('rotor:diameter') else None), (float(t['height:hub']) if t.get('height:hub') else None)
        guess = next((v for k, v in MODELS.items() if k in model), (100, 100))
        plants[name]['turbines'].append(dict(lat=p[0], lon=p[1], mw=_mw(t.get('generator:output:electricity')), rotor=rotor or guess[0],
                                             hub=hub or guess[1], model=t.get('model') or t.get('generator:model'), start=t.get('start_date')))
    for v in plants.values():
        ts = v['turbines']; known = sum(x['mw'] or 0 for x in ts)
        if not ts and v['mw']:
            n = math.ceil(v['mw'] / 3); ts += [dict(lat=v['lat'], lon=v['lon'], mw=v['mw'] / n, rotor=100, hub=100, model=None, start=None)] * n
        for x in ts:  # unmapped rating: the plant's remaining output split evenly
            if not x['mw']:
                x['mw'] = max((v['mw'] or 3 * len(ts)) - known, 0) / max(sum(1 for y in ts if not y['mw']), 1) or 3.0
        v['mw'] = v['mw'] or sum(x['mw'] for x in ts)
    return plants


def weather_sites():
    """Locations that get their own weather files: the solar park, the battery, and one per wind plant >= 5 MW
    (plants within 3 km of a kept one share its file)."""
    out = {SOLAR['key']: dict(lat=SOLAR['lat'], lon=SOLAR['lon']), BATTERY['key']: dict(lat=BATTERY['lat'], lon=BATTERY['lon'])}
    for name, p in sorted(wind_plants().items(), key=lambda kv: -(kv[1]['mw'] or 0)):
        if (p['mw'] or 0) < 5 or any(_km((p['lat'], p['lon']), (s['lat'], s['lon'])) < 3 for s in out.values()):
            continue
        out[wind_key(name)] = dict(lat=round(p['lat'], 4), lon=round(p['lon'], 4))
    return out


def wind_key(name):
    return 'wind_' + re.sub(r'[^a-z0-9]+', '_', name.lower().replace('ä', 'a').replace('ö', 'o').replace('õ', 'o').replace('ü', 'u')).strip('_')


def wind_site_of(name):
    """The weather file a plant uses (its own, or the kept one within 3 km)."""
    p = wind_plants()[name]; ws = weather_sites()
    return min(((k, _km((p['lat'], p['lon']), (s['lat'], s['lon']))) for k, s in ws.items() if k.startswith('wind_')), key=lambda x: x[1])[0]


SITES = None  # filled lazily by qest.fetch_weather through weather_sites()


def locate_solar():
    """Re-run the cloud-pattern match: correlation of the park's hourly forecast anomalies with ERA5 GHI anomalies per grid cell."""
    import numpy as np
    import pandas as pd
    from qest.paths import DATASET
    s = pd.read_parquet(DATASET / 'solarpark_forecast_15min.parquet').set_index('datetime_utc').solarpark_forecast_mw
    h = s.resample('h').mean().dropna()['2026-04-15':]; h.index = h.index.tz_convert(None)
    anom = lambda x: x - x.groupby([x.index.month, x.index.hour]).transform('mean')
    out = []
    for g in json.load(open(RAW / 'ghi_grid.json')):
        v = pd.Series(g['hourly']['shortwave_radiation'], index=pd.to_datetime(g['hourly']['time']) - pd.Timedelta('1h')).astype(float).reindex(h.index)
        m = v.notna() & (h > .5)
        out.append((float(np.corrcoef(anom(v[m]), anom(h[m]))[0, 1]), g['latitude'], g['longitude']))
    return sorted(set(out), reverse=True)


if __name__ == '__main__':
    for r in locate_solar()[:5]: print('solar match r=%.3f at %.2f N %.2f E' % r)
    for k, v in weather_sites().items(): print(k, v)
    p = wind_plants()[PORTFOLIO_WIND]; print(PORTFOLIO_WIND, p['mw'], 'MW', len(p['turbines']), 'turbines', {(t['model'], t['rotor'], t['hub'], t['mw']) for t in p['turbines']})
