"""Estonian weather for the dataset's year (2026-01-01 .. 2026-10-25), keyless (Open-Meteo), into data/raw/.

- ghi_grid.json: ERA5 hourly global horizontal irradiance on a 0.25 deg grid over Estonia (locates the dataset's solar park:
  its forecast's cloud pattern correlates best with the irradiance where it stands, qest.sites.locate_solar).
- era5_<site>.json: ERA5 hourly 'what happened' at each portfolio site: irradiance (GHI, DNI, DHI), 2 m temperature, 10 m and
  100 m wind, 100 m direction, surface pressure, cloud cover.
- fc_<site>.json: the forecast issued the day before (Open-Meteo Previous Runs API, best-match model at 15 min where the model
  has it, hourly otherwise): what a trader knew at the D-1 gate; '_previous_day1' columns.
usage: .venv/bin/python -m qest.fetch_weather [--grid] [--sites]
"""
import json
import sys
import time
import urllib.request

import numpy as np

from qest.paths import RAW

START, END = '2026-01-01', '2026-10-25'
ERA = 'shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m,wind_speed_100m,wind_direction_100m,surface_pressure,cloud_cover'
FC = 'shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_direction_80m,surface_pressure,cloud_cover'


def get(u, t=600):
    for k in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'qest-2026-hackathon/0.1 (research)'}), timeout=t).read()
        except Exception as e:  # 429 / transient: back off
            print('retry', k, e, flush=True); time.sleep(30 * (k + 1))
    raise RuntimeError(u)


def grid():
    p = RAW / 'ghi_grid.json'
    if p.exists():
        return
    lats, lons = np.arange(57.5, 59.76, .25), np.arange(21.75, 28.26, .25)
    pts = [(a, o) for a in lats for o in lons]; out = []
    for i in range(0, len(pts), 25):  # 25 locations a request
        la = ','.join(f'{a:.2f}' for a, _ in pts[i:i + 25]); lo = ','.join(f'{o:.2f}' for _, o in pts[i:i + 25])
        r = json.loads(get(f'https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={START}&end_date={END}'
                           f'&hourly=shortwave_radiation&models=era5&timezone=GMT'))
        out += r if isinstance(r, list) else [r]; print('grid', i + 25, '/', len(pts), flush=True); time.sleep(5)
    p.write_text(json.dumps(out))


def sites():
    from qest.sites import weather_sites
    for k, s in weather_sites().items():
        la, lo = s['lat'], s['lon']
        for name, url in ((f'era5_{k}.json', f'https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={START}&end_date={END}'
                                              f'&hourly={ERA}&models=era5&timezone=GMT&wind_speed_unit=ms'),
                          (f'fc_{k}.json', f'https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&start_date={START}&end_date={END}'
                                           f'&hourly={",".join(v + "_previous_day1" for v in FC.split(","))}&timezone=GMT&wind_speed_unit=ms')):
            p = RAW / name
            if not p.exists():
                p.write_bytes(get(url)); print('fetched', name, flush=True); time.sleep(3)


if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    if '--grid' in sys.argv or len(sys.argv) == 1: grid()
    if '--sites' in sys.argv or len(sys.argv) == 1: sites()
