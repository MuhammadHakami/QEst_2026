"""What the PV and wind assets produced (actual) and what was forecast at the D-1 gate, per 15-min MTU, MW.

PV (Pihlaka, 42.3 MW): actual = the dataset's latest forecast (issued close to delivery: the best record of production
  the dataset holds) from 7 Apr; before that the park was not built (devices 0). The 20 MW hybrid PV uses the same
  per-MW profile. Where the profile is missing (Q1, gaps) and for every D-1 forecast the dataset lacks (it carries one only
  for Jul-Aug, 'v1-recon', and Oct, 'v1'), pvlib turns Open-Meteo weather at the park into power: orientation (fixed south
  at a fitted tilt, or single-axis tracking) and the DC/AC scale are fitted to the dataset on train days only.
Wind (PORTFOLIO_WIND, e.g. Paldiski 45 MW): hub-height wind from ERA5 100 m (actual) or the D-1 forecast's 80/120 m
  (shear to hub), the IEC air-density correction, a generic pitch-regulated power curve from each turbine's rated power and
  rotor (Cp max 0.45 with drivetrain losses, cut-in 3 m/s, cut-out 25 m/s), 8 % wake + availability + electrical losses,
  then one level factor for all of Estonia's wind (wind_scale: the fleet vs Elering's measured renewables, train days).
usage: .venv/bin/python -m qest.power   (fits PV, writes data/derived/generation.parquet, prints skill)
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from qest.data import SPLITS, frame
from qest.paths import DERIVED, RAW
from qest.sites import PORTFOLIO_WIND, SOLAR, wind_plants, wind_site_of

LOSS_WIND = .08


def om(name):
    """An Open-Meteo hourly file as a UTC DataFrame; values are the hour's mean ending at the stamp (radiation) or
    instantaneous (wind): both moved to the 15-min grid by time interpolation onto MTU midpoints."""
    j = json.load(open(RAW / name))['hourly']
    df = pd.DataFrame(j).assign(time=lambda x: pd.to_datetime(x.time, utc=True)).set_index('time').astype(float)
    df.columns = [c.replace('_previous_day1', '') for c in df.columns]
    return df


def to15(h, mean_ending=False):
    """Hourly -> 15-min MTU starts: values at MTU midpoints by linear interpolation (a mean-ending hour sits at its middle)."""
    h = h.copy()
    if mean_ending:
        h.index = h.index - pd.Timedelta('30min')
    idx = pd.date_range(h.index.min().floor('h'), h.index.max().ceil('h'), freq='15min', tz='UTC')
    mid = idx + pd.Timedelta('7.5min')
    out = h.reindex(h.index.union(mid)).interpolate('time', limit_area='inside').reindex(mid)
    out.index = idx
    return out


# ---------- PV ----------
def pv_raw(w, tilt, tracking, lat=SOLAR['lat'], lon=SOLAR['lon']):
    """pvlib: plane-of-array irradiance -> DC power per kWp-ish unit (relative), with cell temperature."""
    import pvlib
    loc = pvlib.location.Location(lat, lon)
    sp = loc.get_solarposition(w.index + pd.Timedelta('7.5min')); sp.index = w.index  # sun at the MTU midpoint
    ghi = w.shortwave_radiation.clip(0); dhi = w.diffuse_radiation.clip(0).fillna(0)
    dni = w.direct_normal_irradiance.clip(0).fillna(0)
    if tracking:
        tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180, max_angle=55, backtrack=True, gcr=.35)
        st, sa = tr.surface_tilt.fillna(0), tr.surface_azimuth.fillna(180)
    else:
        st, sa = tilt, 180
    poa = pvlib.irradiance.get_total_irradiance(st, sa, sp.apparent_zenith, sp.azimuth, dni, ghi, dhi, albedo=.2)['poa_global'].fillna(0)
    tc = pvlib.temperature.faiman(poa, w.temperature_2m, w.get('wind_speed_10m', 2))
    return (poa / 1000 * (1 - .004 * (tc - 25))).clip(0)


@lru_cache(None)
def pv_fit():
    """Choose orientation + scale on train days (Apr-Jun, park online): least squares of the dataset's latest forecast on
    pvlib(ERA5), clipped at 42.26 MW (the dataset's maximum: the inverter limit)."""
    f = frame(); y = f.solarpark_forecast_mw
    w = to15(om(f'era5_{SOLAR["key"]}.json'), mean_ending=True)
    tr = (f.split == 'train') & y.notna()
    best = None
    for tracking, tilt in [(True, 0)] + [(False, t) for t in (15, 25, 35, 45)]:
        x = pv_raw(w, tilt, tracking).reindex(f.index)
        m = tr & x.notna()
        for k in np.linspace(20, 80, 61):
            e = float(((np.minimum(k * x[m], 42.26) - y[m]) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, tracking, tilt, k)
    return dict(rmse=best[0] ** .5, tracking=best[1], tilt=best[2], scale=best[3])


def pv_mw(kind):
    """Pihlaka MW per MTU: kind 'actual' or 'd1'."""
    f = frame(); P = pv_fit()
    name = f'era5_{SOLAR["key"]}.json' if kind == 'actual' else f'fc_{SOLAR["key"]}.json'
    w = to15(om(name), mean_ending=True)
    model = np.minimum(P['scale'] * pv_raw(w, P['tilt'], P['tracking']), 42.26).reindex(f.index)
    data = f.solarpark_forecast_mw if kind == 'actual' else f.solarpark_forecast_d1_mw
    online = f.solarpark_forecast_devices > 0
    return data.where(data.notna(), model).where(online | (kind == 'model'), 0.0).fillna(0.0).rename(f'pv_{kind}'), model.fillna(0.0)


# ---------- wind ----------
def power_curve(v, rated_mw, rotor_m, rho=1.225):
    """Generic pitch-regulated curve: P = 0.5 rho A Cp v^3 up to rated; cut-in 3, cut-out 25 m/s."""
    A = np.pi * (rotor_m / 2) ** 2
    p = .5 * rho * A * .45 * np.asarray(v, float) ** 3 / 1e6
    return np.where((v < 3) | (v > 25), 0.0, np.minimum(p, rated_mw))


def wind_mw(plant=PORTFOLIO_WIND, kind='actual', calibrated=True, turbines=None, mw=None):
    """The plant's MW per MTU from ERA5 (actual) or the D-1 forecast, scaled to the plant's stated output.
    turbines / mw: a corrected turbine list and plant rating (qest.assets); default the OSM record."""
    P = dict(wind_plants()[plant]); key = wind_site_of(plant)
    P['turbines'] = turbines if turbines is not None else P['turbines']; P['mw'] = mw or P['mw']
    w = to15(om(f'{"era5" if kind == "actual" else "fc"}_{key}.json'))
    T = w.temperature_2m + 273.15; p = w.surface_pressure * 100
    out = 0
    for t in P['turbines']:
        if kind == 'actual':
            a = np.clip(np.log(w.wind_speed_100m.clip(.3) / w.wind_speed_10m.clip(.3)) / np.log(10), 0, .45); v = w.wind_speed_100m * (t['hub'] / 100) ** a
        else:
            a = np.clip(np.log(w.wind_speed_120m.clip(.3) / w.wind_speed_80m.clip(.3)) / np.log(1.5), 0, .45); v = w.wind_speed_120m * (t['hub'] / 120) ** a
        rho = p * np.exp(-9.81 * t['hub'] / (287.05 * T)) / (287.05 * T)
        out = out + power_curve(v * (rho / 1.225) ** (1 / 3), t['mw'], t['rotor'])
    tot = sum(t['mw'] for t in P['turbines'])
    s = pd.Series(np.asarray(out) * (P['mw'] / tot) * (1 - LOSS_WIND), index=w.index)
    return (s.reindex(frame().index).ffill().fillna(0.0) * (wind_scale() if calibrated else 1.0)).rename(f'wind_{kind}')


@lru_cache(None)
def wind_scale():
    """Sim2real level for every wind plant: Estonia's whole OSM fleet through the same curves vs Elering's measured
    renewable production, least squares through the origin on the training days (shape: r = 0.88 per MTU, 0.93 per day;
    the generic Cp curve and ERA5 100 m wind run high by ~1.5x)."""
    from qest.sites import wind_plants
    f = frame(); a = sum(wind_mw(n, 'actual', calibrated=False) for n, v in wind_plants().items() if (v['mw'] or 0) >= 1)
    e = f.ee_real_production_renewable; m = (f.split == 'train') & a.notna() & e.notna()
    return float((a[m] * e[m]).sum() / (a[m] ** 2).sum())


def build():
    f = frame()
    pa, model_a = pv_mw('actual'); pd1, model_f = pv_mw('d1')
    g = pd.DataFrame({'pv_actual': pa, 'pv_d1': pd1, 'pv_model_era5': model_a, 'pv_model_fc': model_f,
                      'wind_actual': wind_mw(kind='actual'), 'wind_d1': wind_mw(kind='d1')}, index=f.index)
    g['pv_actual_per_mw'] = g.pv_actual / SOLAR['mw']; g['pv_d1_per_mw'] = g.pv_d1 / SOLAR['mw']
    # the hypothetical 20 MW hybrid PV exists all year: the park's per-MW profile where the park ran, pvlib(ERA5) before
    g['hybrid_pv_actual_per_mw'] = np.where(f.solarpark_forecast_devices >= 4, g.pv_actual_per_mw, g.pv_model_era5 / SOLAR['mw'])
    g['hybrid_pv_d1_per_mw'] = np.where(f.solarpark_forecast_d1_mw.notna(), g.pv_d1_per_mw, g.pv_model_fc / SOLAR['mw'])
    g.to_parquet(DERIVED / 'generation.parquet')
    return g


def generation():
    p = DERIVED / 'generation.parquet'
    return pd.read_parquet(p) if p.exists() else build()


def skill(g=None):
    f = frame(); g = generation() if g is None else g
    out = {}
    for split in SPLITS:
        m = f.split == split
        both = m & f.solarpark_forecast_d1_mw.notna() & (f.solarpark_forecast_devices > 0)
        out[split] = dict(pv_d1_mae=round(float((g.pv_d1 - g.pv_actual)[m & (f.solarpark_forecast_devices > 0)].abs().mean()), 2),
                          pv_dataset_d1_mae=round(float((f.solarpark_forecast_d1_mw - g.pv_actual)[both].abs().mean()), 2) if both.any() else None,
                          pv_model_d1_mae=round(float((g.pv_model_fc - g.pv_actual)[m & (f.solarpark_forecast_devices > 0)].abs().mean()), 2),
                          wind_d1_mae=round(float((g.wind_d1 - g.wind_actual)[m].abs().mean()), 2), wind_cf=round(float(g.wind_actual[m].mean() / 45), 3))
    return out


if __name__ == '__main__':
    print('PV fit (train):', pv_fit())
    g = build(); print(skill(g))
