"""ECB euro reference rate EUR -> USD (keyless SDMX API), for the deck's USD figures. -> data/raw/ecb_eurusd.csv
usage: .venv/bin/python -m qest.fetch_fx"""
import urllib.request

from qest.paths import RAW

URL = 'https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2026-01-01&format=csvdata'


def eurusd():
    """Mean and latest ECB EUR/USD reference rate over 2026 (fetched once)."""
    import pandas as pd
    p = RAW / 'ecb_eurusd.csv'
    if not p.exists():
        req = urllib.request.Request(URL, headers={'User-Agent': 'qest-hackathon/1.0'})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    d = pd.read_csv(p)[['TIME_PERIOD', 'OBS_VALUE']].dropna()
    return dict(latest=float(d.OBS_VALUE.iloc[-1]), latest_date=str(d.TIME_PERIOD.iloc[-1]), mean_2026=float(d.OBS_VALUE.mean()))


if __name__ == '__main__':
    print(eurusd())
