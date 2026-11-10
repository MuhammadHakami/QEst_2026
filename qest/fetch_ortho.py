"""Orthophoto mosaics of the portfolio's real sites from the Estonian Land Board (Maa-amet, CC BY 4.0, 'Maa-amet
ortofoto'): the TMS service foto@GMC (Web Mercator, y counted from the south), 0.3 m / px at zoom 18 at 59 N.
Each mosaic is saved with its corner coordinates so pixels map back to lat / lon (qest.detect, the three.js view).
usage: .venv/bin/python -m qest.fetch_ortho   -> data/raw/ortho_<site>.jpg + ortho_<site>.json
"""
import io
import json
import math
import time
import urllib.request

from qest.paths import RAW

URL = 'https://tiles.maaamet.ee/tm/tms/1.0.0/foto@GMC/{z}/{x}/{y}.jpg'
# site: (lat, lon, half-width m, zoom)
SITES = dict(solar_pihlaka=(59.322, 25.307, 700, 17), battery_kiisa=(59.2375, 24.7050, 300, 18),
             wind_paldiski=(59.366, 24.087, 1300, 16))


def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile_ll(x, y, z):
    n = 2 ** z
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))), x / n * 360 - 180


def mosaic(lat, lon, half_m, z):
    dlat = half_m / 111_320; dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    return mosaic_bbox(lat + dlat, lat - dlat, lon - dlon, lon + dlon, z)


def mosaic_bbox(n, s, w, e, z):
    from PIL import Image
    lat = (n + s) / 2
    x0, y0 = tile_xy(n, w, z); x1, y1 = tile_xy(s, e, z)
    X, Y = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    im = Image.new('RGB', (256 * len(X), 256 * len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            req = urllib.request.Request(URL.format(z=z, x=x, y=2 ** z - 1 - y), headers={'User-Agent': 'krattrade-qest'})
            for k in range(3):
                try:
                    im.paste(Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert('RGB'), (256 * i, 256 * j)); break
                except Exception:
                    time.sleep(2 * (k + 1))
    north, west = tile_ll(X[0], Y[0], z); south, east = tile_ll(X[-1] + 1, Y[-1] + 1, z)
    return im, dict(north=north, south=south, west=west, east=east, zoom=z, width=im.width, height=im.height,
                    m_per_px=156543.03 * math.cos(math.radians(lat)) / 2 ** z, source='Maa-amet ortofoto (CC BY 4.0), foto@GMC')


WIND_3D = ('Sopi-Tootsi tuulepark', 'Aidu Tuulepark', 'Aulepa Tuulepark', 'Narva tuhavälja tuulepark', 'Saarde tuulepark')  # largest parks, for qest.twin


def wind_sites():
    """WIND_3D plants as SITES entries: centred on their mapped turbines, wide enough for all of them plus 400 m."""
    from qest.sites import _km, wind_key, wind_plants
    out = {}
    for name in WIND_3D:
        T = wind_plants()[name]['turbines']; lat = sum(t['lat'] for t in T) / len(T); lon = sum(t['lon'] for t in T) / len(T)
        half = 1000 * max(_km((lat, lon), (t['lat'], t['lon'])) for t in T) + 400
        out[wind_key(name)] = (lat, lon, half, 16 if half < 1500 else 15)
    return out


ESTONIA = (59.72, 57.48, 21.70, 28.25, 9)  # the whole country at ~150 m / px: the asset map's base (qest.twin)


def main():
    if not (RAW / 'ortho_estonia.jpg').exists():
        im, meta = mosaic_bbox(*ESTONIA); im.save(RAW / 'ortho_estonia.jpg', quality=85); (RAW / 'ortho_estonia.json').write_text(json.dumps(meta, indent=1))
        print('estonia', im.size)
    for key, (lat, lon, half, z) in {**SITES, **wind_sites()}.items():
        if (RAW / f'ortho_{key}.jpg').exists():
            continue
        im, meta = mosaic(lat, lon, half, z)
        im.save(RAW / f'ortho_{key}.jpg', quality=90); (RAW / f'ortho_{key}.json').write_text(json.dumps(meta, indent=1))
        print(key, im.size, round(meta['m_per_px'], 3), 'm/px')


if __name__ == '__main__':
    main()
