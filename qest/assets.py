"""Every generating and storage asset in Estonia at its real position, with its model, simulated hour by hour over 2026,
and checked against close-up aerial photos and Elering's national measurements.

REGISTER (`register`): wind turbines from OSM, corrected by public records (VERIFIED: model, hub, rotor; the four GE
  turbines OSM files under Pakri go back to Paldiski); every OSM solar plant; the Hertz 1 battery.
IMAGERY (`chips`): for each asset a close-up from the Land Board orthophoto tiles (0.3-0.6 m / px for turbines and small
  plants, coarser only where a plant's outline needs it), with an automatic check, every rule visible here:
  turbine: a white nacelle / blade blob (max > 215, saturation < .12, 3-400 m2) within 40 m of the mapped position;
  solar: blue-grey module pixels (qest.detect's colour rule) with parallel-row texture (structure-tensor coherence > .4
  over 7 m) inside the plant's box; module area x 21 % efficiency = DC rating seen in the photo;
  battery: qest.detect's compound and container-field rule.
  The photos are not dated per tile: an asset built after the flight shows as bare ground.
SIMULATION (`simulate`), hourly UTC, 1 Jan - 25 Oct 2026:
  wind: every turbine at its mapped position inside its park's wakes (PyWake TurbOPark, `wind_turbine_mw`; adapted from
  the sage_globe twin's energy/wind.py), its own rated power / rotor / hub, ERA5 100 m wind at the plant's weather point
  sheared to hub height at the hour's air density; 97 % availability, 2 % electrical; plant totals scaled to the stated
  output where OSM maps fewer turbines than were built;
  solar: each plant's DC rating (OSM tag, else the photo's) through pvlib at fixed south 25 deg (the Pihlaka fit:
  56 MW per unit of plane-of-array irradiance for 42.3 MW AC, so AC = DC / 1.32, clipped), irradiance from the
  0.25 deg ERA5 grid cell the plant sits in (GHI split by Erbs), air temperature from the nearest ERA5 point.
  Sim to real: the wind fleet against Elering's measured renewable production (which excludes solar) as
  a x wind + b (b = biomass, hydro, waste), fitted on Jan-Jun and scored on Jul-Oct; the solar fleet against
  Elering's measured solar production, a x solar, the same split (OSM misses most rooftops, so a > 1 is expected).
usage: .venv/bin/python -m qest.assets   (fetches tiles into data/raw/tiles, writes data/derived/assets_*.json/.parquet)
"""
import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from qest.fetch_ortho import URL, tile_ll, tile_xy
from qest.paths import DERIVED, RAW
from qest.sites import BATTERY, SOLAR, _km, _mw, weather_sites, wind_plants, wind_site_of

CHIP, PER = 384, 10  # close-up size (px) and atlas grid (PER x PER chips a file)
TILES = RAW / 'tiles'
DC_AC = 56.0 / 42.3  # the Pihlaka fit (qest.power.pv_fit): DC per AC MW
EFF = .21
CELL_M = 4.0  # module-cell grid for the 3D tables (m)
# public sources checked 8 Nov 2026; None = not published, the OSM value is kept
VERIFIED = {
    'Paldiski tuulepark': dict(model='GE 2.5-100', n=18, mw=2.5, rotor=100, hub=None,
                               src='https://www.ge.com/news/press-releases/ge-bringing-45-megawatts-advanced-wind-turbine-technology-estonia',
                               note='OSM has all 18; it files 4 of them under Pakri (nearest plant centre). The register moves them back.'),
    'Narva tuhavälja tuulepark': dict(model='Enercon E-82 E2', n=17, mw=2.3, rotor=82, hub=107,
                                      src='https://cleantechnica.com/2013/09/11/39-mw-wind-farm-opens-in-narva-estonia-on-site-of-old-oil-shale-ash-depository/',
                                      note='OSM carries no model; the trading simulator uses a generic 100 m rotor / 100 m hub here.'),
    'Saarde tuulepark': dict(model='Vestas V150-4.2 (4.3 MW mode)', n=9, mw=4.3, rotor=150, hub=None,
                             src='https://www.vestas.com/en/media/company-news/2022/vestas-secures-39-mw-order-in-estonia-c3593455',
                             note='OSM maps 4 of 9 (one 4 km south); output is scaled to the 39 MW plant. Hub height not published.'),
    'Sopi-Tootsi tuulepark': dict(model='Nordex N163/6.X', n=38, mw=6.7, rotor=163, hub=159,
                                  src='https://enefitgreen.ee/en/uudised/Sopi-Tootsi-tuulepargi-tuulikud-toodab-Nordex', note=None),
    'Aidu Tuulepark': dict(model='Eleon 3M116 (direct drive)', n=None, mw=None, rotor=116, hub=None,
                           src='https://en.wind-turbine-models.com/turbines/2329-eleon-3m116',
                           note='Rating reported as 3.0-3.4 MW; how many of the mapped turbines stand today is not confirmed.'),
    'Pakri Tuulepark': dict(model='Nordex N90/2300', n=8, mw=2.3, rotor=90, hub=80, src='https://thewindpower.net/windfarm_en_791_pakri.php',
                            note="OSM's 4 extra turbines here are Paldiski's GE 2.5-100s; the register moves them back."),
}


# ------------------------------------------------------------------ register
@lru_cache(None)
def estonia():
    """Estonia's border (OSM admin_level 2 relation) as closed rings of (lon, lat): the outer ways chained end to end."""
    from matplotlib.path import Path
    B = json.loads((RAW / 'osm_border.json').read_text())['elements'][0]
    ways = [[(q['lon'], q['lat']) for q in m['geometry']] for m in B['members'] if m['role'] == 'outer' and m.get('geometry')]
    rings = []
    while ways:
        ring = ways.pop(0)
        while ring[0] != ring[-1]:
            k = next((i for i, w in enumerate(ways) if w[0] == ring[-1] or w[-1] == ring[-1]), None)
            if k is None:
                break
            w = ways.pop(k); ring += w[1:] if w[0] == ring[-1] else w[::-1][1:]
        rings.append(Path(ring))
    return rings


def in_estonia(lat, lon):
    return any(r.contains_point((lon, lat)) for r in estonia())


def sp_rotor(mw, w_m2=350.0):
    """Rotor diameter (m) of an untagged turbine from its rating at a typical onshore specific power (W per swept m2;
    assumption, 300-400 for Europe's onshore fleet). The OSM reader's default 100 m made 0.1-0.8 MW turbines run at 60-87 %."""
    return round(float(np.sqrt(4 * mw * 1e6 / (np.pi * w_m2))), 1)



@lru_cache(None)
def register():
    P = wind_plants(); plants, turbines = {}, []
    moved = [t for t in P['Pakri Tuulepark']['turbines'] if t['model'] is None and t['mw'] == 2.5]
    for name, p in P.items():
        v = VERIFIED.get(name) or {}
        T = [t for t in p['turbines'] if not (name == 'Pakri Tuulepark' and t in moved)] + (moved if name == 'Paldiski tuulepark' else [])
        T = [dict(t, model=v.get('model') or t['model'], rotor=v.get('rotor') or (t['rotor'] if t['model'] else sp_rotor(v.get('mw') or t['mw'])),
                  hub=v.get('hub') or (t['hub'] if t['model'] else max(30.0, round(sp_rotor(v.get('mw') or t['mw']))) if not v.get('rotor') else t['hub']),
                  mw=v.get('mw') or t['mw'], placed=(t['lat'], t['lon']) != (p['lat'], p['lon'])) for t in T]
        mw = p['mw'] if name != 'Pakri Tuulepark' else sum(t['mw'] for t in T)
        plants[name] = dict(name=name, lat=p['lat'], lon=p['lon'], mw=round(mw, 2), start=p['start'], operator=p.get('operator'),
                            wsite=wind_site_of(name), n=len(T), model=T[0]['model'], rotor=T[0]['rotor'], hub=T[0]['hub'], ver=VERIFIED.get(name))
        turbines += [dict(plant=name, **t) for t in T]
    solar = []
    for e in json.loads((RAW / 'osm_solar.json').read_text())['elements']:
        t, b = e.get('tags', {}), e.get('bounds')
        if not b or not in_estonia((b['minlat'] + b['maxlat']) / 2, (b['minlon'] + b['maxlon']) / 2):
            continue
        m = _mw(t.get('plant:output:electricity'))
        solar.append(dict(id=e['id'], name=t.get('name'), lat=(b['minlat'] + b['maxlat']) / 2, lon=(b['minlon'] + b['maxlon']) / 2, bounds=b,
                          mw_osm=m if m and m < 200 else None, operator=t.get('operator')))
    geo = {e['id']: [(q['lat'], q['lon']) for q in e['geometry']] for e in json.loads((RAW / 'osm_solar_geom.json').read_text())['elements'] if e.get('geometry')}
    for x in solar:
        x['poly'] = geo.get(x['id'])
    kirik = next((x for x in solar if 'Kirikmäe' in (x['name'] or '')), None)
    battery = [dict(name=BATTERY['name'], lat=BATTERY['lat'], lon=BATTERY['lon'], mw=BATTERY['mw'], mwh=BATTERY['mwh'], status='operating since 3 Feb 2026', dispatch=True,
                    src='https://www.energy-storage.news/evecon-mirova-and-corsica-sole-inaugurate-first-of-two-100mw-200mwh-bess-in-estonia/'),
               dict(name='Auvere battery (Eesti Energia)', lat=59.2700, lon=27.8975, mw=26.5, mwh=53.1, status='operating since 1 Feb 2025; placed at the Auvere plant (pad not mapped)',
                    dispatch=False, src='https://enefit.ee/en/-/uudised/eesti-energia-avas-auveres-eesti-suurima-akusalvesti'),
               dict(name='Hertz 2 battery (Aruküla)', lat=59.3612, lon=25.0450, mw=100.0, mwh=200.0, status='under construction, due late 2026; placed by Aruküla substation',
                    dispatch=False, src='https://www.ess-news.com/2025/03/27/estonia-inagurates-its-largest-battery-storage-project/')]
    if kirik:
        battery.append(dict(name='Kirikmäe battery', lat=kirik['lat'], lon=kirik['lon'] + .004, mw=55.0, mwh=250.0, status='under construction since Jan 2026; placed beside the Kirikmäe solar park',
                            dispatch=False, src='https://www.pv-magazine.com/2026/02/20/estonia-advances-1-1-gwh-heat-storage-250-mwh-bess-and-800-mwh-battery-park'))
    other = []
    for e in json.loads((RAW / 'osm_plants.json').read_text())['elements']:
        t, g = e.get('tags', {}), e.get('geometry') or []
        pts = [(q['lat'], q['lon']) for q in g] or ([(e['lat'], e['lon'])] if 'lat' in e else [])
        if not pts:
            continue
        la, lo = sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts)
        if in_estonia(la, lo) and t.get('power') == 'plant':
            other.append(dict(name=t.get('name'), source=t.get('plant:source'), mw=_mw(t.get('plant:output:electricity')), lat=la, lon=lo, poly=pts if len(pts) > 2 else None))
    return dict(plants=plants, turbines=turbines, solar=solar, battery=battery, other=other)


# ------------------------------------------------------------------ imagery
def tile(z, x, y):
    p = TILES / f'{z}/{x}/{y}.jpg'
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        for k in range(4):
            try:
                req = urllib.request.Request(URL.format(z=z, x=x, y=2 ** z - 1 - y), headers={'User-Agent': 'krattrade-qest'})
                p.write_bytes(urllib.request.urlopen(req, timeout=30).read()); time.sleep(.15); break  # polite: the server blocks bursts
            except Exception:
                time.sleep(2 * (k + 1))
    try:
        return Image.open(p).convert('RGB')
    except Exception:
        return Image.new('RGB', (256, 256))


def chip(lat, lon, half_m, zmax=18):
    """A square close-up centred on (lat, lon), half_m metres each way, at the finest zoom (<= zmax) keeping it near CHIP px."""
    mpp0 = 156543.03 * math.cos(math.radians(lat))
    z = int(min(zmax, max(12, round(math.log2(mpp0 * CHIP * 1.5 / (2 * half_m))))))  # native pixels >= 1.5 x CHIP: turbines at 0.3 m / px (z18)
    cx, cy = tile_xy(lat, lon, z); r = half_m / (mpp0 / 2 ** z) / 256  # half-size in tiles
    x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
    X, Y = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    im = Image.new('RGB', (256 * len(X), 256 * len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            im.paste(tile(z, x, y), (256 * i, 256 * j))
    box = [round((x0 - X[0]) * 256), round((y0 - Y[0]) * 256), round((x1 - X[0]) * 256), round((y1 - Y[0]) * 256)]
    im = im.crop(box)
    n, w = tile_ll(x0, y0, z); s, e = tile_ll(x1, y1, z)
    return im, dict(north=n, south=s, west=w, east=e, z=z, mpp=mpp0 / 2 ** z)


def check_turbine(A, mpp):
    V = A.max(-1); sat = (V - A.min(-1)) / (V + 1e-9)
    white = ndi.binary_opening((V > 215) & (sat < .12), np.ones((2, 2))); lab, n = ndi.label(white)
    if not n:
        return dict(ok=False, offset_m=None)
    cy, cx = np.array(ndi.center_of_mass(white, lab, range(1, n + 1))).T; area = ndi.sum(white, lab, range(1, n + 1)) * mpp ** 2
    h, w = V.shape; d = np.hypot(cx - w / 2, cy - h / 2) * mpp; d[(area < 3) | (area > 400)] = np.inf
    j = int(np.argmin(d)); return dict(ok=bool(d[j] < 40), offset_m=round(float(d[j]), 1) if np.isfinite(d[j]) else None)


def check_solar(A, mpp, inner):
    """inner: boolean mask of the plant's box in the chip. Module pixels with row texture -> share of the box, DC MWp."""
    R, G, B = A[..., 0], A[..., 1], A[..., 2]; V = A.max(-1)
    panel = (B >= G - 4) & (B >= R - 10) & (V < 150) & (V > 25)
    Y = A.mean(-1); gx, gy = ndi.sobel(Y, 1), ndi.sobel(Y, 0); s = max(1.5, 7 / mpp / 2)
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(a, s) for a in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
    mod = panel & (coh > .4) & inner; a = float(mod.sum() * mpp ** 2); box = float(inner.sum() * mpp ** 2)
    k = max(1, int(round(CELL_M / mpp))); h, w = mod.shape  # module cells on a ~4 m grid: where the 3D view puts its tables
    cm = mod[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean((1, 3)) > .5; yy, xx = np.nonzero(cm)
    return dict(ok=bool(a > max(60, .05 * box)), module_m2=round(a), share=round(a / max(box, 1), 3), mwp=round(a * EFF / 1e3, 3),
                cells_px=[((xx + .5) * k / w).round(4).tolist(), ((yy + .5) * k / h).round(4).tolist()], cell_m=round(k * mpp, 2))


def chips():
    """Every asset's close-up and check -> data/derived/assets_imagery.json + assets_chip_<i>.jpg atlases (16 x 16 chips)."""
    Rg = register(); jobs = []
    for i, t in enumerate(Rg['turbines']):
        if t['placed']:
            jobs.append(('turbine', i, t['lat'], t['lon'], 80, 18))
    for i, s in enumerate(Rg['solar']):
        b = s['bounds']; half = max((b['maxlat'] - b['minlat']) * 111_320, (b['maxlon'] - b['minlon']) * 111_320 * math.cos(math.radians(s['lat']))) / 2
        jobs.append(('solar', i, s['lat'], s['lon'], max(30, half * 1.2 + 15), 18))
    for i, b in enumerate(Rg['battery']):
        jobs.append(('battery', i, 59.2375, 24.7050, 160, 18))  # the compound (qest.detect), 1.5 km from the substation point

    def one(j):
        kind, i, lat, lon, half, zmax = j
        im, M = chip(lat, lon, half, zmax); A = np.asarray(im).astype(float)
        if kind == 'turbine':
            c = check_turbine(A, M['mpp'])
        elif kind == 'solar':
            b = Rg['solar'][i]['bounds']; h, w = A.shape[:2]
            ys = (np.arange(h) + .5) / h; xs = (np.arange(w) + .5) / w
            lat_ = M['north'] + (M['south'] - M['north']) * ys; lon_ = M['west'] + (M['east'] - M['west']) * xs
            inner = ((lat_ >= b['minlat']) & (lat_ <= b['maxlat']))[:, None] & ((lon_ >= b['minlon']) & (lon_ <= b['maxlon']))[None, :]
            c = check_solar(A, M['mpp'], inner)
        else:
            d = json.loads((DERIVED / 'assets_detected.json').read_text())['battery']
            c = dict(ok=True, compound_m2=round(d['compound_m2']), note='compound detected; the photo predates completion (pads, a first row of containers)')
        return kind, i, im.resize((CHIP, CHIP), Image.LANCZOS), dict(M, **c)
    with ThreadPoolExecutor(2) as ex:  # the Land Board server drops clients that open many connections
        res = list(ex.map(one, jobs))
    out = dict(turbine={}, solar={}, battery={}); per = PER
    for a in range(0, len(res), per * per):
        atlas = Image.new('RGB', (per * CHIP, per * CHIP))
        for k, (kind, i, im, meta) in enumerate(res[a:a + per * per]):
            atlas.paste(im, ((k % per) * CHIP, (k // per) * CHIP)); out[kind][i] = dict(meta, atlas=a // (per * per), cell=k)
        atlas.save(DERIVED / f'assets_chip_{a // (per * per)}.jpg', quality=78)
    (DERIVED / 'assets_imagery.json').write_text(json.dumps(out))
    return out


@lru_cache(None)
def imagery():
    """The close-ups' checks; empty when they have not been fetched (then solar capacity is OSM-tagged only)."""
    p = DERIVED / 'assets_imagery.json'
    if not p.exists():
        return dict(turbine={}, solar={}, battery={})
    return {k: {int(i): v for i, v in d.items()} for k, d in json.loads(p.read_text()).items()}


# ------------------------------------------------------------------ simulation
def hours():
    from qest.data import frame
    f = frame(); return pd.date_range(f.index[0].floor('h'), f.index[-1].floor('h'), freq='h')


TI, AVAIL, ELEC = .10, .97, .02  # onshore turbulence, availability, electrical losses: the sage_globe twin's onshore park (energy/wind.py)
WS_G, WD_G = np.arange(0, 30.01, .5), np.arange(0, 360, 5.0)


def hub_wind(key, hub):
    """ERA5 100 m at the plant's weather point, sheared to hub height with the hour's 10-100 m shear, as the IEC
    density-equivalent speed; and the direction (qest.power and the twin's energy/wind.hub_wind)."""
    from qest.power import om
    w = om(f'era5_{key}.json')
    a = np.clip(np.log(w.wind_speed_100m.clip(.3) / w.wind_speed_10m.clip(.3)) / np.log(10), 0, .45); v = w.wind_speed_100m * (hub / 100) ** a
    T = w.temperature_2m + 273.15 - .0065 * hub; pr = w.surface_pressure * 100 * np.exp(-9.81 * hub / (287.05 * T))
    return v * (pr / (287.05 * T) / 1.225) ** (1 / 3), w.wind_direction_100m % 360


def wind_turbine_mw():
    """Every turbine's MW each hour inside its park's wakes (DTU PyWake 2.6, TurbOPark / Nygaard 2022, adapted from the
    sage_globe twin's energy/wind.py): the park's real layout (OSM positions), each turbine's own type (PyWake's
    GenericWindTurbine from its rated power, rotor and hub: Cp max 0.49 with drivetrain losses), a (direction, speed)
    table per turbine read at the hour's hub wind. Plants without mapped turbines run free-stream at their centre.
    Returns (per-turbine MW, index = register turbines; per-plant MW scaled to the stated plant output; wake loss per plant)."""
    from scipy.interpolate import RegularGridInterpolator
    from py_wake.literature.turbopark import Nygaard_2022
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbines
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
    Rg = register(); H = hours(); TM = {}; PM = {}; loss = {}
    for name, p in Rg['plants'].items():
        idx = [i for i, t in enumerate(Rg['turbines']) if t['plant'] == name]; T = [Rg['turbines'][i] for i in idx]
        kinds = sorted({(t['model'] or 'generic', t['rotor'], t['hub'], t['mw']) for t in T})
        wts = WindTurbines.from_WindTurbine_lst([GenericWindTurbine(f'{m} {d:.0f}m {h:.0f}m {mw}MW', d, h, mw * 1e3, turbulence_intensity=TI, ws_cutin=3, ws_cutout=25)
                                                 for m, d, h, mw in kinds])
        ty = np.array([kinds.index((t['model'] or 'generic', t['rotor'], t['hub'], t['mw'])) for t in T])
        lat0, lon0 = np.mean([t['lat'] for t in T]), np.mean([t['lon'] for t in T])
        x = np.array([(t['lon'] - lon0) * 111_320 * math.cos(math.radians(lat0)) for t in T]); y = np.array([(t['lat'] - lat0) * 110_574 for t in T])
        mean_hub = float(np.mean([t['hub'] for t in T])); v, wd = hub_wind(p['wsite'], mean_hub)
        free = np.stack([wts.power(np.asarray(v), type=k) / 1e6 for k in ty], 1)
        if all(t['placed'] for t in T) and len(T) > 1 and np.ptp(x) + np.ptp(y) > 1:
            sim = Nygaard_2022(UniformSite(ti=TI), wts)(x, y, type=ty, wd=WD_G, ws=WS_G)
            Pw = sim.Power.transpose('wt', 'wd', 'ws').values / 1e6; wdg = np.r_[WD_G, 360.0]
            pts = np.c_[np.asarray(wd), np.clip(np.asarray(v), 0, WS_G[-1])]
            mw = np.stack([RegularGridInterpolator((wdg, WS_G), np.vstack([Pw[k], Pw[k][:1]]), bounds_error=False, fill_value=0.0)(pts) for k in range(len(T))], 1)
        else:
            mw = free
        mw = mw * AVAIL * (1 - ELEC); free = free * AVAIL * (1 - ELEC)
        for k, i in enumerate(idx):
            TM[i] = pd.Series(mw[:, k], index=v.index)
        scale = p['mw'] / sum(t['mw'] for t in T)  # unmapped turbines (Saarde: 4 of 9 mapped) carried by the plant's stated rating
        PM[name] = pd.Series(mw.sum(1) * scale, index=v.index); loss[name] = round(float(1 - mw.sum() / max(free.sum(), 1e-9)), 4)
    return pd.DataFrame(TM).reindex(H), pd.DataFrame(PM).reindex(H), loss


def solar_cell_profiles(cells):
    """Per-MW-AC output for each 0.25 deg grid cell (index into ghi_grid.json), hourly."""
    import pvlib
    from qest.power import pv_raw
    G = json.loads((RAW / 'ghi_grid.json').read_text()); H = hours(); ws = weather_sites(); out = {}
    temps = {k: pd.DataFrame(json.loads((RAW / f'era5_{k}.json').read_text())['hourly']).assign(time=lambda x: pd.to_datetime(x.time, utc=True)).set_index('time')
             for k in ws}
    for c in cells:
        g = G[c]; lat, lon = g['latitude'], g['longitude']
        w = pd.DataFrame(g['hourly']).assign(time=lambda x: pd.to_datetime(x.time, utc=True)).set_index('time').astype(float)
        near = min(ws, key=lambda k: _km((lat, lon), (ws[k]['lat'], ws[k]['lon'])))
        w['temperature_2m'] = temps[near].temperature_2m.reindex(w.index).astype(float); w['wind_speed_10m'] = temps[near].wind_speed_10m.reindex(w.index).astype(float)
        mid = w.index - pd.Timedelta('30min')  # radiation is the hour's mean ending at the stamp
        sp = pvlib.solarposition.get_solarposition(mid, lat, lon); sp.index = w.index
        e = pvlib.irradiance.erbs(w.shortwave_radiation.clip(0), sp.zenith, w.index.dayofyear)
        w['direct_normal_irradiance'] = np.nan_to_num(np.asarray(e['dni'], float)); w['diffuse_radiation'] = np.nan_to_num(np.asarray(e['dhi'], float))
        w.index = mid  # pv_raw puts the sun at index + 7.5 min; close enough to the hour's middle at hourly resolution
        raw = pv_raw(w, 25, False, lat, lon); raw.index = raw.index - pd.Timedelta('30min')  # -> the start of the hour it covers (frame hours)
        out[c] = np.minimum(56.0 / 42.3 * raw, 1.0).reindex(H)
    return pd.DataFrame(out)


def solar_cell_of(lat, lon, G):
    return int(np.argmin([(g['latitude'] - lat) ** 2 + ((g['longitude'] - lon) * .52) ** 2 for g in G]))


def simulate():
    """-> data/derived/assets_sim.parquet (hourly: per wind plant MW, per solar cell per-MW), assets_fleet.json (fits)."""
    from qest.data import frame
    Rg = register(); IM = imagery(); H = hours()
    G = json.loads((RAW / 'ghi_grid.json').read_text())
    for i, s in enumerate(Rg['solar']):
        c = IM['solar'].get(i, {}); s['cell'] = solar_cell_of(s['lat'], s['lon'], G)
        s['mw_img'] = round(c.get('mwp', 0) / DC_AC, 3); s['seen'] = bool(c.get('ok'))
        s['mw'] = s['mw_osm'] if s['mw_osm'] else (s['mw_img'] if s['seen'] else 0.0)  # photo capacity only where panels are confirmed
    TW, W, wake_loss = wind_turbine_mw(); cells = sorted({s['cell'] for s in Rg['solar'] if s['mw'] > 0}); S = solar_cell_profiles(cells)
    cap = pd.Series(0.0, index=cells)
    for s in Rg['solar']:
        if s['mw'] > 0:
            cap[s['cell']] += s['mw']
    f = frame(); E = f[['ee_real_production_renewable', 'ee_real_solar_energy_production']].groupby(f.index.floor('h')).mean().reindex(H)
    es = E.ee_real_solar_energy_production; E['ee_real_solar_energy_production'] = es.where(es.groupby(es.index.date).transform('max') > 0)  # whole days of 0: Elering's gaps (3 Oct, 20-25 Oct)
    wind = W.sum(1); solar = (S * cap).sum(1); tr = H < pd.Timestamp('2026-07-01', tz='UTC'); te = ~tr
    m = tr & E.ee_real_production_renewable.notna(); a, b = np.polyfit(wind[m], E.ee_real_production_renewable[m], 1)
    ms = tr & E.ee_real_solar_energy_production.notna(); k = float((solar[ms] * E.ee_real_solar_energy_production[ms]).sum() / (solar[ms] ** 2).sum())

    def score(sim, real, mask):
        mm = mask & real.notna() & sim.notna(); return dict(r=round(float(np.corrcoef(sim[mm], real[mm])[0, 1]), 3), mae=round(float((sim[mm] - real[mm]).abs().mean()), 1),
                                                            mean_real=round(float(real[mm].mean()), 1))
    fits = dict(wind=dict(a=round(float(a), 3), b=round(float(b), 1), fit=score(a * wind + b, E.ee_real_production_renewable, tr), test=score(a * wind + b, E.ee_real_production_renewable, te),
                          nameplate_mw=round(sum(p['mw'] for p in Rg['plants'].values()), 1), wake_loss=wake_loss),
                solar=dict(k=round(k, 3), fit=score(k * solar, E.ee_real_solar_energy_production, ms), test=score(k * solar, E.ee_real_solar_energy_production, te),
                           osm_mw=round(float(cap.sum()), 1), plants_used=sum(1 for s in Rg['solar'] if s['mw'] > 0), plants=len(Rg['solar'])))
    out = pd.concat({'wind': W, 'turbine': TW.rename(columns=str), 'solar': S.rename(columns=str)}, axis=1)
    out[('fleet', 'wind')] = wind; out[('fleet', 'solar')] = solar
    out[('elering', 'renewable')] = E.ee_real_production_renewable; out[('elering', 'solar')] = E.ee_real_solar_energy_production
    out.columns = ['|'.join(c) for c in out.columns]; out.to_parquet(DERIVED / 'assets_sim.parquet')
    (DERIVED / 'assets_fleet.json').write_text(json.dumps(dict(fits=fits, solar=[{k: v for k, v in s.items() if k != 'bounds'} for s in Rg['solar']]), default=float))
    return out, fits


def _check():
    Rg = register(); P = Rg['plants']
    assert P['Paldiski tuulepark']['n'] == 18 and abs(P['Paldiski tuulepark']['mw'] - 45) < .01 and P['Pakri Tuulepark']['n'] == 8
    assert P['Narva tuhavälja tuulepark']['hub'] == 107 and P['Narva tuhavälja tuulepark']['rotor'] == 82
    assert in_estonia(59.437, 24.745) and in_estonia(58.25, 22.5) and not in_estonia(57.75, 22.6) and not in_estonia(60.17, 24.94)  # Tallinn, Saaremaa; Kolka (LV), Helsinki
    A = np.zeros((64, 64, 3)); A[30:34, 28:36] = 250  # a 4 x 8 px white nacelle in the centre
    assert check_turbine(A, .6)['ok'] and not check_turbine(np.zeros((64, 64, 3)), .6)['ok']
    print('assets check ok:', len(Rg['turbines']), 'turbines,', len(Rg['solar']), 'solar plants')


if __name__ == '__main__':
    _check(); t = time.time(); im = chips(); print({k: (len(v), sum(x['ok'] for x in v.values())) for k, v in im.items()}, f'{time.time() - t:.0f} s')
    imagery.cache_clear(); _, fits = simulate(); print(json.dumps(fits, indent=1))
