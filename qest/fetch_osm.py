"""Every mapped solar plant and wind turbine in Estonia, from OpenStreetMap (Overpass API), into data/raw/osm_*.json.

Wind: each turbine node (power=generator, generator:source=wind) with its tags (rated power, rotor, hub height, operator,
start date when mapped) + every wind plant (power=plant, plant:source=wind) outline with its name and output.
Grid: 110-330 kV lines and substations; border: Estonia's admin_level 2 relation.
Solar: every solar plant (power=plant, plant:source=solar) outline with name, operator and plant:output:electricity.
usage: .venv/bin/python -m qest.fetch_osm
"""
import json
import time
import urllib.parse
import urllib.request

from qest.paths import RAW

MIRRORS = ['https://overpass.private.coffee/api/interpreter', 'https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter',
           'https://maps.mail.ru/osm/tools/overpass/api/interpreter']
AREA = ''  # a bounding box is far lighter for the servers than the country area; non-EE hits are dropped by the reader
BB = '(57.45,21.6,59.75,28.25)'
QUERIES = {
    'osm_wind.json': AREA + '(node["power"="generator"]["generator:source"="wind"]'+BB+';'
                            'nwr["power"="plant"]["plant:source"="wind"]'+BB+';);out tags center;',
    'osm_solar.json': AREA + '(nwr["power"="plant"]["plant:source"="solar"]'+BB+';'
                             'nwr["power"="generator"]["generator:source"="solar"]["generator:output:electricity"~"MW"]'+BB+';);'
                             'out tags center bb;',
    'osm_battery.json': AREA + '(nwr["power"="plant"]["plant:source"="battery"]'+BB+';'
                               'nwr["power"="generator"]["generator:source"="battery"]'+BB+';'
                               'nwr["power"="substation"]["name"~"Kiisa|Aruküla"]'+BB+';);out tags center bb;',
    # transmission grid: Elering's 110 / 330 kV lines (geometry) and substations, for the asset map (qest.twin)
    'osm_grid.json': '(way["power"="line"]["voltage"~"(110000|220000|330000)"]'+BB+';'
                     'nwr["power"="substation"]["voltage"~"(110000|220000|330000)"]'+BB+';);out tags geom;',
    'osm_border.json': 'relation["ISO3166-1"="EE"]["admin_level"="2"];out geom;',
    # outlines for the 3D twin: every solar plant's polygon, every other power plant (oil shale, CHP, hydro, biomass, storage), substation outlines
    'osm_solar_geom.json': '(way["power"="plant"]["plant:source"="solar"]'+BB+';way["power"="generator"]["generator:source"="solar"]["generator:output:electricity"~"MW"]'+BB+';);out tags geom;',
    'osm_plants.json': '(nwr["power"="plant"]["plant:source"!~"^(solar|wind)$"]'+BB+';nwr["power"="generator"]["generator:source"="battery"]'+BB+';);out tags geom;',
    # underground / subsea cables of 110 kV and up, north to Finland for EstLink 1-2 (HVDC)
    'osm_cables.json': 'way["power"="cable"]["voltage"~"(110000|150000|330000|350000|450000)"](57.45,21.6,60.4,28.25);out tags geom;',
}


def overpass(q):
    for attempt in range(6):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            data = urllib.parse.urlencode({'data': '[out:json][timeout:180];' + q}).encode()
            req = urllib.request.Request(url, data=data, headers={'User-Agent': 'qest-2026-hackathon/0.1 (research)'})
            return urllib.request.urlopen(req, timeout=240).read()
        except Exception as e:  # Overpass mirrors are flaky: rotate, back off
            print('overpass', url, e, flush=True); time.sleep(10 * (attempt + 1))
    raise RuntimeError('every Overpass mirror failed')


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        p = RAW / name
        if p.exists():
            continue
        body = overpass(q); json.loads(body); p.write_bytes(body)
        print('fetched', name, len(json.loads(body)['elements']), 'elements', flush=True)


if __name__ == '__main__':
    main()
