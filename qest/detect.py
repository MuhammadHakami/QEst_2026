r"""Assets detected in the Land Board orthophotos (qest.fetch_ortho), for the three.js ground view and as a check on the
capacities the dataset and operators state. Plain image processing (SciPy), every rule visible here:
  SOLAR (Pihlaka, 0.61 m / px): module pixels are blue-grey (B >= G - 4, B >= R - 10, 25 < max < 150; grass and
    forest have G > B); forest shadow passes that colour test too, so a pixel also needs parallel-row texture: the
    structure tensor's orientation coherence > .4 (12 px window) and module density > .15 (31 px). Closing + hole fill
    gives the park; module area / cos(tilt) x 21 % module efficiency x 1 kW/m2 gives the DC rating (tilt 0-35 deg).
  BATTERY (Hertz 1, Kiisa, 0.31 m / px): the concrete compound (bright, unsaturated, the largest such region) and the
    container field inside it (the smoothest 40 m stretch of slab: switchyard gear and buildings are dense edges). The orthophoto predates completion (pads, a first row of
    containers), so the 54 containers the operator reports are laid out on the detected compound's axis.
  WIND (Paldiski, 1.22 m / px): each OSM turbine is confirmed by the white rotor / nacelle blob (max > 215,
    saturation < .12, 3-400 m2) nearest to it within 60 m; the blob's centre is the detected position.
usage: .venv/bin/python -m qest.detect   -> data/derived/assets_detected.json, data/derived/ortho_*_mask.png
"""
import json
import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from qest.paths import DERIVED, RAW

EFF, CELL_M, HERTZ1_CONTAINERS = .21, 2.5, 54


def load(key):
    A = np.asarray(Image.open(RAW / f'ortho_{key}.jpg')).astype(float); M = json.loads((RAW / f'ortho_{key}.json').read_text())
    return A, M


def px2ll(M, x, y):
    lat = M['north'] + (M['south'] - M['north']) * np.asarray(y) / M['height']  # linear in the small box (< 1.5 km)
    lon = M['west'] + (M['east'] - M['west']) * np.asarray(x) / M['width']
    return lat, lon


def coherence(Y, w):
    gx, gy = ndi.sobel(Y, 1), ndi.sobel(Y, 0)
    Jxx, Jyy, Jxy = (ndi.gaussian_filter(a, w) for a in (gx * gx, gy * gy, gx * gy))
    return np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9), .5 * np.arctan2(2 * Jxy, Jxx - Jyy)


def solar():
    A, M = load('solar_pihlaka'); R, G, B = A[..., 0], A[..., 1], A[..., 2]; V = A.max(-1); mpp = M['m_per_px']
    panel = (B >= G - 4) & (B >= R - 10) & (V < 150) & (V > 25)
    coh, ang = coherence(A.mean(-1), 12); dens = ndi.uniform_filter(panel.astype(float), 31)
    park = ndi.binary_fill_holes(ndi.binary_closing(panel & (coh > .4) & (dens > .15), np.ones((15, 15)), iterations=2))
    lab, n = ndi.label(park); sz = ndi.sum(park, lab, range(1, n + 1)); park = np.isin(lab, 1 + np.flatnonzero(sz * mpp ** 2 > 5000))
    mod = panel & park; a = mod.sum() * mpp ** 2
    theta = float(np.degrees(np.median(ang[mod])))  # gradient direction: across the rows
    k = max(1, int(round(CELL_M / mpp))); h, w = mod.shape
    cells = mod[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean((1, 3)) > .5
    yy, xx = np.nonzero(cells); lat, lon = px2ll(M, (xx + .5) * k, (yy + .5) * k)
    Image.fromarray((mod * 255).astype(np.uint8)).save(DERIVED / 'ortho_solar_pihlaka_mask.png')
    return dict(park_ha=park.sum() * mpp ** 2 / 1e4, module_m2=a, mwp={t: a / math.cos(math.radians(t)) * EFF / 1e3 for t in (0, 25, 35)},
                stated_mw=42.3, row_gradient_deg=theta, cell_m=k * mpp, cells=np.c_[lat, lon].round(6).tolist())


def battery():
    A, M = load('battery_kiisa'); V = A.max(-1); sat = (V - A.min(-1)) / (V + 1e-9); mpp = M['m_per_px']
    pad = ndi.binary_opening((V > 175) & (sat < .12), np.ones((5, 5))); lab, n = ndi.label(pad)
    pad = lab == 1 + np.argmax(ndi.sum(pad, lab, range(1, n + 1)))
    pad = ndi.binary_fill_holes(ndi.binary_closing(pad, np.ones((25, 25))))
    yy, xx = np.nonzero(pad); ev, evec = np.linalg.eigh(np.cov(np.c_[xx, yy].T))
    axis = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))  # image degrees, x right, y down
    # the container field: the smoothest stretch of slab (switchyard gear, buildings and vehicles are dense thin edges;
    # the pad field is flat concrete with faint frames): the minimum of the mean gradient magnitude over a 40 m window,
    # among windows wholly inside the compound; the containers follow the compound's axis
    Y = A.mean(-1); k = int(40 / mpp); g = ndi.uniform_filter(np.hypot(ndi.sobel(Y, 1), ndi.sobel(Y, 0)), k)
    full = ndi.uniform_filter(pad.astype(float), int(1.6 * k)) > .995; g[~full] = np.inf
    gy, gx = np.unravel_index(np.argmin(g), g.shape); glat, glon = px2ll(M, gx, gy); grid_axis = axis
    yy0, xx0 = np.nonzero(pad); fx, fy = (gx - xx0.min()) / np.ptp(xx0), (gy - yy0.min()) / np.ptp(yy0)
    edge = pad & ~ndi.binary_erosion(pad, np.ones((3, 3))); ey, ex = np.nonzero(edge)
    order = np.argsort(np.arctan2(ey - yy.mean(), ex - xx.mean()))[::max(1, len(ey) // 120)]
    lat, lon = px2ll(M, ex[order], ey[order]); clat, clon = px2ll(M, xx.mean(), yy.mean())
    return dict(compound_m2=pad.sum() * mpp ** 2, axis_deg=grid_axis, compound_axis_deg=axis, center=[float(glat), float(glon)],
                compound_center=[float(clat), float(clon)], field_in_bbox=[round(float(fx), 2), round(float(fy), 2)], outline=np.c_[lat, lon].round(6).tolist(),
                containers=HERTZ1_CONTAINERS, containers_source='operator (Corsica Sole / ERR, Feb 2026); imagery predates completion',
                stated_mw=100.0, stated_mwh=200.0)


def wind():
    from qest.sites import PORTFOLIO_WIND, wind_plants
    A, M = load('wind_paldiski'); V = A.max(-1); sat = (V - A.min(-1)) / (V + 1e-9); mpp = M['m_per_px']
    white = ndi.binary_opening((V > 215) & (sat < .12), np.ones((2, 2))); lab, n = ndi.label(white)
    cy, cx = np.array(ndi.center_of_mass(white, lab, range(1, n + 1))).T; area = ndi.sum(white, lab, range(1, n + 1)) * mpp ** 2
    ok = (area > 3) & (area < 400); cy, cx = cy[ok], cx[ok]; out = []
    for t in wind_plants()[PORTFOLIO_WIND]['turbines']:
        x = (t['lon'] - M['west']) / (M['east'] - M['west']) * M['width']; y = (t['lat'] - M['north']) / (M['south'] - M['north']) * M['height']
        d = np.hypot(cx - x, cy - y) * mpp; j = int(np.argmin(d)) if len(d) else -1
        hit = j >= 0 and d[j] < 60; lat, lon = px2ll(M, cx[j], cy[j]) if hit else (t['lat'], t['lon'])
        out.append(dict(lat=float(lat), lon=float(lon), osm=[t['lat'], t['lon']], detected=bool(hit), offset_m=float(d[j]) if hit else None,
                        mw=t['mw'], rotor=t['rotor'], hub=t['hub']))
    return dict(park=PORTFOLIO_WIND, turbines=out, detected=sum(o['detected'] for o in out), stated_mw=45.0)


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    res = dict(solar=solar(), battery=battery(), wind=wind(),
               imagery={k: json.loads((RAW / f'ortho_{k}.json').read_text()) for k in ('solar_pihlaka', 'battery_kiisa', 'wind_paldiski')})
    (DERIVED / 'assets_detected.json').write_text(json.dumps(res))
    s, b, w = res['solar'], res['battery'], res['wind']
    print(f"solar: park {s['park_ha']:.1f} ha, modules {s['module_m2'] / 1e4:.1f} ha -> {s['mwp'][0]:.1f}-{s['mwp'][35]:.1f} MWp (stated 42.3 MW), {len(s['cells'])} cells")
    print(f"battery: compound {b['compound_m2']:.0f} m2, compound axis {b['compound_axis_deg']:.0f}, field centre {b['center']}, at {b['field_in_bbox']} of the compound box (x right, y down)")
    print(f"wind: {w['detected']}/{len(w['turbines'])} turbines confirmed, offsets m:", [round(t['offset_m']) for t in w['turbines'] if t['detected']])


if __name__ == '__main__':
    main()
