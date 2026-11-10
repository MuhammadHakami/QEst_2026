r"""Distil the QPU into a small conditional transformer, then let the partitioned loop query the model instead of the QPU.

After Provazza, Knitter, ..., Roetteler, Zhao (IonQ), "Accelerating QC-AFQMC with shadow-trained neural quantum states",
arXiv:2611.08129 (Nov 2026). There a quantum trial state guides classical Monte Carlo walkers, read through ~60k
matchgate shadows that every walker step must sum over (13,200 GPU h). They distil the shadows ONCE into a ~13k-parameter
autoregressive transformer NQS (one causal block, embedding 16, 2 heads, feed-forward 64), symmetry-masked and noise-aware
(a calibration factor a: the device's state = a x the ideal one + noise), and the NQS replaces the shadows (4 GPU h).
Here, mapped term by term:
  trial state        LR-QAOA on each 12-qubit window of the partitioned day QUBO (qest.qpart; frozen schedule)
  walkers            the classical block-descent loop over windows (qpart.solve), which asks for candidates per window
  shadows            the QAOA measurement samples, collected once, on Q3 days only (the training set)
  NQS                one CONDITIONAL autoregressive transformer p(y | window QUBO) shared by every window, day and unit
                     (the paper trains one per molecule; our windows are new every day, so the model is amortised)
  frame              each window's QUBO divided by its rms before the model sees it (as the circuit's angles)
  noise-aware        trained on noisy CUDA-Q trajectories (2-qubit depolarising P2 per CNOT); likelihood of what was
                     measured = a p_model + (1 - a) uniform, the paper's uniform response (exact for global
                     depolarising); a is LEARNED with the weights under a ridge toward 1 (their Eq. 15), the ridge
                     weight scanned and the model picked by a committee rule (lowest held-out energy = their Rayleigh energy)
  committee          3 seeds per setting, lowest held-out energy kept (ideal, unaware and twin models too)
  guide, not replace in the paper the quantum state guides the classical solver (AFQMC is approximate: a better trial
                     state lowers its energy); our MILP is exact on its forecast objective, so a pattern fixed from any
                     sampler can only lower the planned objective (tested: the pick-the-better rule chose the free MILP
                     on all 50 Q4 day-units and reproduced the baseline to the euro). The quantum pattern can only differ
                     in REALISED euros, which is what the 2026 replay measures.
  cost               --cost prices a 2026 season of each route at AWS list prices (their 13,200 GPU h -> 4 GPU h table)
  symmetry mask      none needed: MW / SoC / reserve limits stay in the LP (qest.qdecide), so every bit string is valid;
                     window padding is masked instead (positions past the window's length)
  payoff             at decision time the loop samples the model: zero QPU calls, and the cost no longer grows with shots
Controls: 'twin' = the same model trained on classical Boltzmann samples of each window at the temperature matching the
ideal QAOA mean energy (is the QUBO data, not the quantum source, doing the work?); 'uniform' = random candidates.
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qdistill [--check | --cost]
  -> results/qdistill_{data_val.pkl, *.pt, holdout.csv, days_test.csv, cost.csv}
"""
import functools
import pickle
import sys
import time

import numpy as np
import torch
import torch._dynamo  # noqa: F401  before CUDA-Q: both bundle LLVM, and the other order aborts (tested)
import torch.nn.functional as F
from torch import nn

from qest import qdecide, qpart
from qest.paths import RESULTS

B = qdecide.PART_B; O = max(2, B // 4); SHOTS = qpart.SHOTS
P2 = .003  # 2-qubit depolarising per CNOT: a 99.7 % two-qubit gate, IQM / IBM class in 2026
LAMS = (0., .3, 3., 30.)  # ridge weight on the learned calibration scalar a (the paper's regularisation scan; prior a = 1, a noiseless device)
SEEDS = (0, 1, 2)  # committee: one model per seed and setting, the one with the lowest held-out energy is kept (the paper's selection rule)
D, HEADS, FF, EPOCHS, BATCH = 16, 2, 64, 8, 8192
# List prices, Nov 2026. Braket (aws.amazon.com/braket/pricing): $0.30 a task + per shot; IBM Quantum pay-as-you-go $96 a QPU minute (billed by the
# second). EC2 us-east-1: one H100 = p5.48xlarge / 8, one vCPU-hour = c7i.48xlarge / 192.
USD_TASK, USD_GPU_H, USD_CPU_H = .30, 55.04 / 8, 8.568 / 192
USD_SHOT = {'IonQ Forte': .08, 'IQM Emerald 54q': .0016, 'IQM Garnet 20q': .00145, 'Rigetti Cepheus 108q': .000425}
USD_IBM_S = 96 / 60  # IBM Heron per QPU second; a 1,000-shot 12-qubit job takes about 1-4 s of QPU time (assumption, not measured)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
VARIANTS = ('ideal', 'unaware', 'aware', 'twin')


@functools.lru_cache
def noise_model():
    cudaq = qdecide._cudaq(); nm = cudaq.NoiseModel()
    nm.add_all_qubit_channel('x', cudaq.Depolarization2(P2), num_controls=1); return nm


def allbits(n):
    return (np.arange(1 << n)[:, None] >> np.arange(n)) & 1


def feats(qb):
    """(B, 2B + 1) tokens: own linear term, own coupling row, all linear terms; rms frame; padded to B rows."""
    n = qb['H']; Bs = qb['B'] + qb['B'].T; s = np.sqrt(np.mean(qb['a'] ** 2) + np.mean(Bs ** 2)) + 1e-12
    X = np.zeros((B, 2 * B + 1), np.float32); a = qb['a'] / s
    X[:n, 0] = a; X[:n, 1:1 + n] = Bs / s; X[:n, 1 + B:1 + B + n] = a
    return X


class NQS(nn.Module):
    """Causal transformer: p(y_i = 1 | y_<i, window QUBO). The paper's block sizes."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(2 * B + 1, D); self.tok = nn.Embedding(3, D); self.pos = nn.Embedding(B, D)
        self.blk = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0., batch_first=True); self.out = nn.Linear(D, 1)
        self.register_buffer('mask', nn.Transformer.generate_square_subsequent_mask(B))

    def forward(self, X, y):
        prev = torch.cat([torch.full_like(y[:, :1], 2), y[:, :-1]], 1)  # token 2 = start
        return self.out(self.blk(self.inp(X) + self.tok(prev) + self.pos.weight, src_mask=self.mask, is_causal=True)).squeeze(-1)

    def logp(self, X, y, m):
        return -(F.binary_cross_entropy_with_logits(self(X, y), y.float(), reduction='none') * m).sum(1)

    @torch.no_grad()
    def sample(self, X, n, seed=0):
        g = torch.Generator(device=X.device).manual_seed(seed); Xb = X.expand(n, -1, -1)
        y = torch.zeros(n, B, dtype=torch.long, device=X.device)
        for i in range(B):
            y[:, i] = (torch.rand(n, generator=g, device=X.device) < torch.sigmoid(self(Xb, y)[:, i])).long()
        return y


def boltzmann(qb, e_target, rng):
    """Classical twin: exact Boltzmann samples of the window at the temperature whose mean energy is e_target."""
    E = qdecide.energy(qb, allbits(qb['H'])); lo, hi = -8., 8.
    for _ in range(60):  # bisection on log T
        T = np.exp((lo + hi) / 2); w = np.exp(-(E - E.min()) / T); w /= w.sum()
        lo, hi = (lo, (lo + hi) / 2) if w @ E > e_target else ((lo + hi) / 2, hi)
    z = rng.choice(len(E), SHOTS, p=w); u, c = np.unique(z, return_counts=True)
    return allbits(qb['H'])[u], c


def collect(split='val'):
    """Windows the partitioned loop visits on each day QUBO, with ideal, noisy and twin samples. Cached."""
    path = RESULTS / f'qdistill_data_{split}.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    sched = qpart.schedule(B); data = []; rng = np.random.default_rng(0); orig = qpart.window_solve
    for d, u, q in qpart.day_qubos(split):
        seen = []
        qpart.window_solve = lambda qb, *a, **k: (seen.append(qb), orig(qb, *a, **k))[1]
        try:
            qpart.solve(q, B, O, sched, 'qaoa', 'mult', device=24)
        finally:
            qpart.window_solve = orig
        for w, qb in enumerate(seen):
            kw = dict(p=sched['p'], delta=sched['delta'], shots=SHOTS, seed=w, counts=True)
            ideal = qdecide.solve_qaoa(qb, **kw); noisy = qdecide.solve_qaoa(qb, noise=noise_model(), **kw)
            e = qdecide.energy(qb, ideal[0]) @ ideal[1] / ideal[1].sum()
            data.append(dict(day=d, unit=u, qb=qb, X=feats(qb), ideal=ideal, noisy=noisy, twin=boltzmann(qb, e, rng)))
        print(d.date(), u, len(seen), 'windows', flush=True)
    path.write_bytes(pickle.dumps(data)); return data


def tensors(data, key):
    Xs, ys, ws, ms = [], [], [], []
    for i, r in enumerate(data):
        Y, c = r[key]; n = Y.shape[1]; y = np.zeros((len(Y), B), np.int64); y[:, :n] = Y
        Xs.append(np.full(len(Y), i)); ys.append(y); ws.append(c / c.sum()); ms.append(np.repeat((np.arange(B) < n)[None], len(Y), 0))
    t = lambda v, dt: torch.as_tensor(np.concatenate(v), dtype=dt, device=DEV)
    return torch.as_tensor(np.stack([r['X'] for r in data]), device=DEV), t(Xs, torch.long), t(ys, torch.long), t(ws, torch.float32), t(ms, torch.float32)


def train(data, key, lam=None, seed=0):
    """Maximum likelihood on the measured samples. lam = None: the model is the data (noise-unaware). Else the measured
    distribution is a p_model + (1 - a) 2^-n (the paper's uniform calibration, exact for global depolarising) with the
    scalar a LEARNED alongside the weights under a ridge penalty lam (a - 1)^2 toward a noiseless device; as in the paper
    the ridge keeps the fit from lowering the loss through a alone, and a carries no weight decay. net.a = the fitted a."""
    torch.manual_seed(seed); X, wi, y, w, m = tensors(data, key); net = NQS().to(DEV); n = m.sum(1)
    alpha = torch.tensor(2., device=DEV, requires_grad=True)  # a = sigmoid(alpha), starts at .88
    opt = torch.optim.AdamW([dict(params=net.parameters(), weight_decay=1e-4), dict(params=[alpha], weight_decay=0.)], lr=3e-3)
    steps = EPOCHS * int(np.ceil(len(y) / BATCH)); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(EPOCHS):
        for b in torch.randperm(len(y), device=DEV).split(BATCH):
            lp = net.logp(X[wi[b]], y[b], m[b]); loss = 0.
            if lam is not None:
                a = torch.sigmoid(alpha); lp = torch.logaddexp(lp + torch.log(a), torch.log1p(-a) - n[b] * np.log(2)); loss = lam * (a - 1) ** 2
            loss = loss - (w[b] * lp).sum() / w[b].sum(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.a = 1. if lam is None else float(torch.sigmoid(alpha)); return net.eval()


def committee(tr, ho, key, lam=None, tag=''):
    """The paper's selection: train one model per seed, keep the one with the lowest held-out normalised energy."""
    rows = []
    for sd in SEEDS:
        net = train(tr, key, lam, sd); _NETS['_'] = net
        rows.append(dict(source=f'model {tag}', seed=sd, lam=lam, a=round(net.a, 3), q=model_quality(ho, '_'), net=net))
        print(f'  {tag} seed {sd}: a = {net.a:.3f}, held-out energy {rows[-1]["q"]:.4f}', flush=True)
    return min(rows, key=lambda r: r['q']), rows


_NETS = {}


def window_samples(qb, variant='aware', seed=0):
    if variant not in _NETS:
        net = NQS().to(DEV); net.load_state_dict(torch.load(RESULTS / f'qdistill_{variant}.pt', map_location=DEV)); _NETS[variant] = net.eval()
    y = _NETS[variant].sample(torch.as_tensor(feats(qb), device=DEV), SHOTS, seed)[:, :qb['H']].cpu().numpy()
    return np.unique(y, axis=0)


def norm_energy(qb, Y, c=None):
    """Mean sample energy on [0 = window optimum, 1 = uniform average]."""
    E = qdecide.energy(qb, allbits(qb['H'])); e = qdecide.energy(qb, Y); c = np.ones(len(Y)) if c is None else c
    return (e @ c / c.sum() - E.min()) / (E.mean() - E.min())


def model_quality(rows, variant):
    return float(np.mean([norm_energy(r['qb'], *np.unique(_NETS[variant].sample(torch.as_tensor(r['X'], device=DEV), SHOTS)[:, :r['qb']['H']].cpu().numpy(),
                                                          axis=0, return_counts=True)) for r in rows]))


def day_eval(qs, methods):
    """Partitioned loop on each day QUBO with each candidate source; hit = exact optimum reached."""
    rows = []; sched = qpart.schedule(B)
    for d, u, q in qs:
        emin = qdecide.energy(q, qdecide.solve_exact(q)[0][:1])[0]
        for meth in methods:
            t = time.time(); y, st = qpart.solve(q, B, O, sched, meth, 'mult', device=24); e = qdecide.energy(q, y)[0]
            rows.append(dict(day=str(d.date()), unit=u, method=meth, gap=e - emin, hit=bool(np.isclose(e, emin)), windows=st['circuits'],
                             qpu_circuits=0 if meth.startswith('nqs') or meth == 'uniform' else st['circuits'], s=time.time() - t))
    return rows


def main():
    import pandas as pd
    data = collect('val'); dd = sorted({r['day'] for r in data}); cut = dd[int(.8 * len(dd))]
    tr = [r for r in data if r['day'] < cut]; ho = [r for r in data if r['day'] >= cut]
    print(len(tr), 'training windows,', len(ho), 'held-out windows (Q3 days from', cut.date(), ')', flush=True)
    H = [dict(source='ideal QAOA samples', q=np.mean([norm_energy(r['qb'], *r['ideal']) for r in ho])),
         dict(source='noisy QAOA samples', q=np.mean([norm_energy(r['qb'], *r['noisy']) for r in ho])),
         dict(source='twin Boltzmann samples', q=np.mean([norm_energy(r['qb'], *r['twin']) for r in ho]))]
    t0 = time.time(); nets, allrows = {}, []
    for v, key in (('ideal', 'ideal'), ('unaware', 'noisy'), ('twin', 'twin')):
        nets[v], rows = committee(tr, ho, key, None, v); allrows += rows
    cands = []
    for lam in LAMS:  # regularisation scan x seeds = the committee for the noise-aware model
        best, rows = committee(tr, ho, 'noisy', lam, f'aware lam={lam}'); cands.append(best); allrows += rows
    nets['aware'] = min(cands, key=lambda r: r['q']); t_train = time.time() - t0
    for v in VARIANTS:
        torch.save(nets[v]['net'].state_dict(), RESULTS / f'qdistill_{v}.pt'); _NETS[v] = nets[v]['net']
    Hd = pd.DataFrame(H + [{k: r[k] for k in r if k != 'net'} for r in allrows]); Hd.to_csv(RESULTS / 'qdistill_holdout.csv', index=False)
    print(f'held-out Q3 windows, normalised mean energy (0 = optimum, 1 = uniform); aware: lam = {nets["aware"]["lam"]}, a = {nets["aware"]["a"]}, '
          f'seed {nets["aware"]["seed"]} | params {sum(p.numel() for p in nets["ideal"]["net"].parameters())} | {len(allrows)} models in {t_train / 60:.1f} min')
    print(Hd.round(4).to_string(index=False), flush=True)
    (RESULTS / 'qdistill_train_s.txt').write_text(str(t_train))
    R = pd.DataFrame(day_eval(qpart.day_qubos('test'), ['qaoa', 'nqaoa'] + [f'nqs-{v}' for v in VARIANTS] + ['uniform']))
    R.to_csv(RESULTS / 'qdistill_days_test.csv', index=False)
    print(R.groupby('method')[['hit', 'gap', 'windows', 'qpu_circuits', 's']].mean().round(3).to_string())


def cost():
    """The paper's accounting for our pipeline, across vendors: what a season (every traded 2026 day, two units) costs on the live
    route against the distilled one. QPU time at list prices, classical time at EC2 prices. Measured inputs: circuits per
    day-unit and seconds per day-unit from results/qdistill_days_test.csv, training windows and training time from this
    module's run. Shots: our SHOTS per window (and a 200-shot variant, untested for decision quality). -> results/qdistill_cost.csv"""
    import pandas as pd
    T = pd.read_csv(RESULTS / 'qdistill_days_test.csv'); g = T.groupby('method')[['windows', 's']].mean()
    du = sum(pd.read_parquet(RESULTS / f'days_{sp}_qdecide_chronos.parquet').query("strategy == 'milp@chronos'").shape[0] for sp in ('train', 'val', 'test'))
    nw = len(collect('val')); t_train = float((RESULTS / 'qdistill_train_s.txt').read_text()) if (RESULTS / 'qdistill_train_s.txt').exists() else float('nan')
    part = du * g.loc['qaoa', 'windows']
    routes = [('live partitioned LR-QAOA, 12-qubit windows, every day', part, SHOTS, True), ('live partitioned, 200 shots a window (untested)', part, 200, True),
              ('live monolithic LR-QAOA, 24 qubits, one circuit a day-unit', du, qdecide.SHOTS, True), ('distilled: Q3 training windows, once', nw, SHOTS, False)]
    rows = []
    for route, n, sh, rec in routes:
        r = dict(route=route, circuits=round(n), shots=round(n * sh), recurring=rec)
        r.update({f'usd_{k}': n * (USD_TASK + v * sh) for k, v in USD_SHOT.items()})
        r.update({f'usd_IBM_{t}s': n * t * USD_IBM_S for t in (1, 4)}); rows.append(r)
    rows.append(dict(route='distilled: the season on the model, no QPU (4 vCPU)', circuits=0, shots=0, recurring=True,
                     **{k: du * g.loc['nqs-aware', 's'] / 3600 * USD_CPU_H * 4 for k in rows[0] if k.startswith('usd_')}))
    C = pd.DataFrame(rows); C.to_csv(RESULTS / 'qdistill_cost.csv', index=False)
    pd.set_option('display.width', 250); print(f'{du} day-units in the 2026 season; $0.30 a task + per-shot list prices (Braket), IBM $96 a QPU minute at 1 s or 4 s a job')
    print(C.round(0).to_string(index=False))
    per = {k: part / du * (USD_TASK + v * SHOTS) for k, v in USD_SHOT.items()}
    print('per decision (one day-unit, live partitioned):', {k: round(v, 1) for k, v in per.items()}, f'IBM {part / du * USD_IBM_S:.1f}-{part / du * 4 * USD_IBM_S:.1f}')
    print(f'classical compute: live {du * g.loc["qaoa", "s"] / 3600 * USD_GPU_H:.2f} USD a season on one H100, distilled {C.iloc[-1]["usd_IonQ Forte"]:.3f} USD; training once {t_train / 3600 * USD_GPU_H:.2f} USD')
    # the paper's sentence ("13,200 GPU hours become 4 GPU hours of training plus 6,680 CPU core-hours"), for our season
    cpu_s = 0.34  # s per day-unit for the distilled loop on one CPU core, 1 thread (measured 9 Nov: 34 ms a window)
    for name, k in (('one decision a day-unit (gate A)', 1), ('a re-decision every MTU (T-25 re-plan, 96 a day)', 96)):
        gpu_live = du * k * g.loc['qaoa', 's'] / 3600; print(f'{name}: live simulation {gpu_live:.2f} GPU h a season -> training {t_train / 3600:.3f} GPU h once + '
                                                            f'{du * k * cpu_s / 3600:.2f} CPU core h a season; QPU shots {part * k * SHOTS / 1e6:.1f} M live -> {nw * SHOTS / 1e6:.1f} M once')
    return C


def _check():
    """The model is a normalised distribution over the window (padding masked), and training on one window's samples
    moves its mass onto them."""
    rng = np.random.default_rng(0); qb = qpart.synthetic(5, rng); net = NQS().to(DEV); X = torch.as_tensor(feats(qb), device=DEV)
    Y = np.zeros((32, B), np.int64); Y[:, :5] = allbits(5); m = torch.as_tensor(np.repeat((np.arange(B) < 5)[None], 32, 0), dtype=torch.float32, device=DEV)
    p = net.logp(X.expand(32, -1, -1), torch.as_tensor(Y, device=DEV), m).exp().sum().item(); assert abs(p - 1) < 1e-4, p
    E = qdecide.energy(qb, allbits(5)); top = allbits(5)[np.argsort(E)[:2]]
    r = dict(day=0, qb=qb, X=feats(qb), ideal=(top, np.array([700, 300])))
    globals()['EPOCHS'] = 200; net = train([r], 'ideal'); S = net.sample(X, 1000)[:, :5].cpu().numpy()
    hit = np.mean([any((s == t).all() for t in top) for s in S]); assert hit > .9, hit
    mixed = (allbits(5), np.full(32, 10) + np.isin(np.arange(32), np.argsort(E)[:2]) * 300)  # 2 signal rows on a uniform floor
    net = train([dict(day=0, qb=qb, X=feats(qb), ideal=mixed)], 'ideal', lam=0.); assert .2 < net.a < .9, net.a  # a learns the signal fraction (~.6)
    print('qdistill check ok: normalised, padding masked, fits its samples (', round(float(hit), 3), '), learned a =', round(net.a, 2))


if __name__ == '__main__':
    _check() if '--check' in sys.argv else cost() if '--cost' in sys.argv else main()
