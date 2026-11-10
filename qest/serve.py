"""Local server for the twin page: static files from results/figs/twin, plus the Land Board orthophoto through a disk cache.

  /tiles/maaamet/{z}/{x}/{y}.jpg   XYZ tile; read from data/raw/tiles (the cache qest.assets fills), else fetched once and kept
  /tiles/ping                      200 when the cache is there (the page then routes Land Board tiles through it)

The Land Board (CC BY 4.0) blocks bursts, so at most 6 fetches run upstream at once; tiles it doesn't have (sea, abroad: a 404 or its tiny
black tile) are remembered as empty {y}.404 files and the page falls back to Esri. Esri is display-stream only and is not cached here.
Browsers keep served tiles for a year. --warm fills the cache in the background, one fetch at a time: Estonia at zoom 8-12, and
zoom 13-17 around every wind plant, battery and the 150 largest solar plants.

usage: .venv/bin/python -m qest.serve [port] [--warm]
"""
import os
import sys
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from math import asinh, cos, floor, pi, radians, tan

from .paths import RAW, RESULTS

TILES = RAW / 'tiles'
URL = 'https://tiles.maaamet.ee/tm/tms/1.0.0/foto@GMC/{z}/{x}/{y}.jpg'  # TMS: y counts from the south
UP = threading.BoundedSemaphore(6)  # ponytail: one global cap, the browser's own 6 per host (the domain blocked 6 bulk python threads once; warm takes 1 slot)
EST = (57.5, 59.75, 21.7, 28.25)
BLANK = 400  # bytes: the Land Board answers 200 with a ~116-byte black tile where it has no photo (sea, abroad)  # lat0, lat1, lon0, lon1


def get(z, x, y):
    """The tile's bytes, b'' when the Land Board has none, None when the fetch failed (not cached: try again later)."""
    p = TILES / f'{z}/{x}/{y}.jpg'
    if p.exists():
        b = p.read_bytes()
        return b if len(b) > BLANK else b''
    if p.with_suffix('.404').exists():
        return b''
    with UP:
        if p.exists():  # another request fetched it while this one waited
            b = p.read_bytes()
            return b if len(b) > BLANK else b''
        try:
            req = urllib.request.Request(URL.format(z=z, x=x, y=2 ** z - 1 - y), headers={'User-Agent': 'krattrade-qest'})
            b = urllib.request.urlopen(req, timeout=20).read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                p.parent.mkdir(parents=True, exist_ok=True); p.with_suffix('.404').touch()
                return b''
            return None
        except Exception:
            return None
    p.parent.mkdir(parents=True, exist_ok=True)
    if len(b) <= BLANK:  # its no-data tile: 256 px of black palette, ~116 bytes
        p.with_suffix('.404').touch()
        return b''
    tmp = p.with_suffix(f'.{threading.get_ident()}.part'); tmp.write_bytes(b); os.replace(tmp, p)  # atomic: readers never see half a tile
    return b


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if not self.path.startswith('/tiles/'):
            return super().do_GET()
        if self.path.startswith('/tiles/ping'):
            return self._send(200, b'ok', 'text/plain', 'no-store')
        try:
            src, z, x, y = self.path.split('?')[0].split('/')[2:6]
            z, x, y = int(z), int(x), int(y.split('.')[0])
            assert src == 'maaamet' and 0 <= z <= 19 and 0 <= x < 2 ** z and 0 <= y < 2 ** z
        except Exception:
            return self._send(400, b'bad tile', 'text/plain', 'no-store')
        b = get(z, x, y)
        if b is None:
            return self._send(502, b'upstream failed', 'text/plain', 'no-store')
        if not b:
            return self._send(404, b'no tile', 'text/plain', 'max-age=86400')
        self._send(200, b, 'image/jpeg', 'public, max-age=31536000, immutable')

    def _send(self, code, body, ctype, cache):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', cache); self.send_header('Access-Control-Allow-Origin', '*'); self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # the page dropped a tile it no longer wants
            pass


def txy(lat, lon, z):
    n = 2 ** z
    return floor((lon + 180) / 360 * n), floor((1 - asinh(tan(radians(lat))) / pi) / 2 * n)


def warm_list():
    out = []
    for z in range(8, 13):
        x0, y0 = txy(EST[1], EST[2], z); x1, y1 = txy(EST[0], EST[3], z)
        out += [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    import json
    M = json.loads((RESULTS / 'figs' / 'twin' / 'data.json').read_text())['map']
    sites = [(p['lat'], p['lon']) for p in M['plants']] + [(b['lat'], b['lon']) for b in M['battery']]
    sites += [(s['lat'], s['lon']) for s in sorted(M['solar'], key=lambda s: -(s.get('mw') or 0))[:150]]
    for lat, lon in sites:
        for z in range(13, 18):
            cx, cy = txy(lat, lon, z)
            out += [(z, cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    return list(dict.fromkeys(out))


def warm():
    todo = [t for t in warm_list() if not (TILES / f'{t[0]}/{t[1]}/{t[2]}.jpg').exists() and not (TILES / f'{t[0]}/{t[1]}/{t[2]}.404').exists()]
    print('warm:', len(todo), 'tiles to fetch', flush=True)
    for i, t in enumerate(todo):
        get(*t)  # one at a time: the page's own requests keep the other slots
        if i % 500 == 499:
            print('warm:', i + 1, flush=True)
    print('warm: done', flush=True)


def _check():
    from math import atan, degrees, sinh
    x, y = txy(59.4370, 24.7536, 12)  # Tallinn: the tile's edges must hold the point
    edge = lambda v: degrees(atan(sinh(pi * (1 - 2 * v / 4096))))
    assert edge(y + 1) <= 59.4370 <= edge(y) and x / 4096 * 360 - 180 <= 24.7536 <= (x + 1) / 4096 * 360 - 180, (x, y)
    b = get(12, x, y)
    assert b is None or b == b'' or b[:2] == b'\xff\xd8', 'not a JPEG'
    assert get(16, 37290, 18935) in (b'', None), 'the no-data tile (Gulf of Finland) must read as missing'
    print('serve check ok:', 'cached' if (TILES / f'12/{x}/{y}.jpg').exists() else 'upstream down', len(warm_list()), 'warm tiles listed')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--check' in sys.argv:
        _check(); sys.exit()
    if '--warm' in sys.argv:
        threading.Thread(target=warm, daemon=True).start()
    port = int(args[0]) if args else 18977
    srv = ThreadingHTTPServer(('0.0.0.0', port), partial(Handler, directory=str(RESULTS / 'figs' / 'twin')))
    print(f'twin on http://localhost:{port}/', flush=True)
    srv.serve_forever()
