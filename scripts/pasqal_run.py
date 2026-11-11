"""A scheduling instance in Pasqal Fresnel's native form: maximum independent set by Rydberg blockade (analog, global drive).

Instance per delivery day: the hours whose forecast-free, realised mFRR UP premium is in the day's top 40 % are candidate
discharge hours; the 200 MWh battery needs RECOVER hours between two full discharges, so two candidates closer than that
conflict. "Sell in as many premium hours as the battery can recover between" = maximum independent set of an interval graph.
Atoms sit on a line, one trap per hour (5 um apart), a candidate hour = an atom; the Rabi frequency is set so the blockade
radius spans RECOVER hours: the device's physics is the constraint. Adiabatic sweep (rise, detuning sweep, fall), global
channel only (all Fresnel SA1 offers), so the problem is unweighted. Exact answer by dynamic programming; control = random
independent sets at the same number of shots. Retrospective prices, as the organisers' quantum track (slide 9).
usage: .venv/bin/python scripts/pasqal_run.py --local [--days 5]      # pulser QutipEmulator
       .venv/bin/python scripts/pasqal_run.py --submit [--days 5 --runs 200]   -> results/pasqal_batches.json
       .venv/bin/python scripts/pasqal_run.py --fetch                          -> results/pasqal_mis.csv
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qest.paths import RESULTS, ROOT

SP, RECOVER, T_US = 6.0, 1, 4000  # um per hour, hours a discharge blocks, sequence length (ns). Fresnel SA1 needs an average amplitude >= 0.57 rad/us, i.e. a blockade radius <= ~10 um: one hour of recovery at 6 um per hour (two hours would need atoms closer than the 5 um minimum)
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def instances(n):
    from qest.data import frame
    f = frame(); f = f[f.split == 'test']; out = []
    for d in sorted(f.day.unique())[2::max(1, 23 // n)][:n]:
        g = f[f.day == d]; prem = (g.mfrr_up_eur_mwh.fillna(g.nps_dayahead_eur_mwh) - g.nps_dayahead_eur_mwh).groupby(g.local.dt.hour).mean()
        hrs = sorted(prem[prem >= prem.quantile(.6)].index.tolist()); out.append((str(pd.Timestamp(d).date()), hrs))
    return out


def mis_size(hrs):  # exact, interval graph: greedy by position is optimal
    k, last = 0, -99
    for h in hrs:
        if h - last > RECOVER: k, last = k + 1, h
    return k


def independent(bits, hrs):
    on = [h for h, b in zip(hrs, bits) if b]; return all(b - a > RECOVER for a, b in zip(on, on[1:]))


def sequence(hrs, device):
    from pulser import Pulse, Sequence
    from pulser.register.register_layout import RegisterLayout
    from pulser.waveforms import InterpolatedWaveform
    layout = RegisterLayout([[SP * (h - 11.5), 0.0] for h in range(24)] + [[SP * (h - 11.5), SP * 2] for h in range(24)])  # second row: empty traps (filling <= 50 %)
    reg = layout.define_register(*layout.get_traps_from_coordinates(*[[SP * (h - 11.5), 0.0] for h in hrs]), qubit_ids=[f'h{h}' for h in hrs])  # trap ids are sorted by position, not by our list
    rb = SP * (RECOVER + .5); omega = device.interaction_coeff / rb ** 6  # blockade radius between RECOVER and RECOVER + 1 hours
    seq = Sequence(reg, device); seq.declare_channel('ryd', 'rydberg_global')
    d0 = 4 * omega; seq.add(Pulse(InterpolatedWaveform(T_US, [1e-9, omega, omega, 1e-9]), InterpolatedWaveform(T_US, [-d0, -d0 / 2, d0 / 2, d0]), 0), 'ryd')
    return seq, omega


def score(day, hrs, counts, name):
    rng = np.random.default_rng(0); n = sum(counts.values()); opt = mis_size(hrs); best, hit, valid = 0, 0, 0
    for s, c in counts.items():
        b = [int(x) for x in s]
        if independent(b, hrs): valid += c; best = max(best, sum(b)); hit += c * (sum(b) == opt)
    rb = 0
    for _ in range(n):
        b = (rng.random(len(hrs)) < .5).astype(int)
        if independent(b, hrs): rb = max(rb, int(b.sum()))
    return dict(day=day, device=name, atoms=len(hrs), runs=n, mis=opt, best_valid=best, found_opt=best == opt, p_opt=hit / n, valid_share=valid / n, random_best=rb, random_opt=rb == opt)


def connect():
    for line in (ROOT / '.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
    from pasqal_cloud import PasqalCloudConnection
    return PasqalCloudConnection(username=os.environ['PASQAL_EMAIL'], password=os.environ['PASQAL_PASS'], project_id=os.environ['PASQAL_PROJECT_ID'], region='sa')


if __name__ == '__main__':
    I = instances(arg('--days', 5)); rows = []
    if '--local' in sys.argv:
        from pulser.devices import MockDevice as AnalogDevice  # no radius limit locally; Fresnel SA1 allows 72 um
        from pulser_simulation import QutipEmulator
        for day, hrs in I:
            seq, om = sequence(hrs, AnalogDevice); c = QutipEmulator.from_sequence(seq).run().sample_final_state(arg('--runs', 200)); rows.append(score(day, hrs, dict(c), 'qutip')); print(rows[-1], flush=True)
    elif '--submit' in sys.argv:
        c = connect(); dev = c.fetch_available_devices()['FRESNEL_SA1']; B = {}
        for day, hrs in I:
            seq, om = sequence(hrs, dev); rr = c.submit(seq, job_params=[dict(runs=arg('--runs', 200), variables={})])
            bid = getattr(rr, 'batch_id', None) or getattr(rr, '_batch_id', None); B[day] = dict(batch=bid, hrs=hrs); print(day, len(hrs), 'atoms, omega', round(om, 3), 'batch', bid, flush=True)
            json.dump(B, open(RESULTS / 'pasqal_batches.json', 'w'))
        json.dump(B, open(RESULTS / 'pasqal_batches.json', 'w'))
    elif '--fetch' in sys.argv:
        from pulser.backend.remote import RemoteResults
        c = connect(); B = json.load(open(RESULTS / 'pasqal_batches.json'))
        for day, v in B.items():
            rr = RemoteResults(v['batch'], c); st = rr.get_batch_status(); print(day, st, flush=True)
            if 'DONE' in str(st):
                for r in rr.results:
                    cnt = dict(getattr(r, 'bitstring_counts', None) or r.final_bitstrings); rows.append(score(day, v['hrs'], cnt, 'FRESNEL_SA1'))
    if rows:
        T = pd.DataFrame(rows); T.to_csv(RESULTS / f'pasqal_mis_{rows[0]["device"]}.csv', index=False); print(T.to_string())
