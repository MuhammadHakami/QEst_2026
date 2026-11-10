r"""Quantum architecture search (NAS) and post-training architecture search (PostNAS) for the slide-11 puzzle.

Chronos-2 + MILP is the classical baseline; this module searches the quantum circuit itself, hardware-aware.

NAS (Optuna, multi-objective TPE): maximise P(optimum) in the final state, minimise two-qubit gates. The space:
  mixer   'x'  : plain QAOA (Farhi 2014), |+>^n start, RX mixer, the whole 2^n space (P1/P2 only; P3 is 2^30);
          'xy' : quantum alternating operator ansatz (Hadfield et al., Algorithms 12, 34, 2019) with an XY mixer per
                 decision, so the state never leaves the one-hot plans: 5^3 = 125 states at P2, 5^6 = 15,625 at P3.
                 The one-hot penalty couplers vanish on that subspace and drop out of the circuit. Ring or complete XY
                 graph, Trotterised in parity order as on hardware, 1-2 Trotter steps.
  start   uniform over the plans (W state per decision), or warm (each decision's options weighted by their value,
          Egger et al., Quantum 5, 479, 2021).
  depth   p = 1..6.
  angles  'train'    per instance by Adam from a linear ramp;
          'ramp'     fixed linear ramp, no training (LR-QAOA, Montanez-Barrera & Michielsen, 2024-25, run on IBM to 109
                     qubits): the hardware-friendly choice;
          'transfer' trained on Q3 hours, median angles reused unchanged (angle concentration).
  loss    mean energy, CVaR_alpha (Barkoutsos et al., Quantum 4, 256, 2020) or Gibbs (Li et al., PRR 2, 023074, 2020).
PostNAS (after Jet-Nemotron's PostNAS, Gu et al., NeurIPS 2025: keep the trained weights frozen, search what to keep):
  start from the NAS winner with its angles trained, freeze them, search which layers to drop and which couplers to
  prune (|J_ij| below tau x max), optionally a short fine-tune of the kept angles; same two objectives.
Search on Q3 (validation) hours only, prices known (slide 9, retrospective); the picks are scored once on Q4.
Gate counts are CZ-equivalents before routing: ZZ 2, XY pair 2, a k-option decision's start state 3 (k - 1).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qnas --level P2 [--trials 60] [--post 40]
  -> results/qnas_<level>_{nas,postnas,test}.csv, results/qnas_<level>.json
"""
import argparse
import itertools
import json
import time

import numpy as np
import optuna
import pandas as pd
import torch

from qest import qubo
from qest.paths import RESULTS
from qest.qsota import DEV, qaoa_state

SHOTS, STEPS, LR = 4096, 150, .05


def load(split, level, every=1, ndays=0):
    inst = qubo.instances(split, level, 'retro', ndays=ndays, every=every)
    P = [qubo.build(H, level) for _, _, H, _ in inst]
    HJ = [qubo.arrays(V, lin, quad) for V, lin, quad, off, lam in P]
    k = len(qubo.OPTS[level]); m = len(P[0][0]); G = m // k
    sub = torch.zeros(k ** G, m, dtype=torch.float64, device=DEV)  # every one-hot plan, first decision most significant
    for r, c in enumerate(itertools.product(range(k), repeat=G)):
        sub[r, [g * k + ci for g, ci in enumerate(c)]] = 1
    full = ((torch.arange(1 << m, device=DEV)[:, None] >> torch.arange(m, device=DEV)) & 1).double() if m <= 20 else None
    t = lambda a: torch.as_tensor(np.stack(a), device=DEV, dtype=torch.float64)
    same = torch.as_tensor(np.equal.outer(np.arange(m) // k, np.arange(m) // k), device=DEV)
    return dict(inst=inst, P=P, h=t([a for a, _ in HJ]), J=t([b for _, b in HJ]), off=t([p[3] for p in P]),
                lam=t([p[4] for p in P]), k=k, G=G, m=m, sub=sub, full=full, same=same, level=level)


def energies(h, J, off, X):
    return off[:, None] + h @ X.T + torch.einsum('nm,bmk,nk->bn', X, J, X)


def counted(arch, d, J):
    """The couplers the circuit must apply: all of them in the full space, only between decisions in the XY subspace."""
    return J * ~d['same'] if arch['space'] == 'xy' else J


def pairs(k, graph):
    if graph == 'full':
        return list(itertools.combinations(range(k), 2))
    if k == 2:
        return [(0, 1)]
    return [(i, i + 1) for i in range(0, k - 1, 2)] + [(i, i + 1) for i in range(1, k - 1, 2)] + [(k - 1, 0)]


def mixer(beta, k, graph, r):
    """(B, k, k): exp(-i beta (XX + YY) / 2) over the decision's XY graph, Trotterised, on its k one-hot states."""
    I = torch.eye(k, dtype=torch.complex64, device=DEV); U = I.expand(len(beta), k, k)
    c = torch.cos(beta / r).to(torch.complex64)[:, None, None]; s = torch.sin(beta / r).to(torch.complex64)[:, None, None]
    for _ in range(r):
        for a, b in pairs(k, graph):
            D = torch.zeros(k, k, dtype=torch.complex64, device=DEV); D[a, a] = D[b, b] = 1
            O = torch.zeros(k, k, dtype=torch.complex64, device=DEV); O[a, b] = O[b, a] = 1
            U = (I + (c - 1) * D - 1j * s * O) @ U
    return U


def start(arch, d, sel):
    B, N = len(sel), d['k'] ** d['G']
    if arch.get('init', 'w') == 'w':
        return torch.full((B, N), N ** -.5, dtype=torch.complex64, device=DEV)
    k = d['k']; v = -(d['h'][sel] + d['lam'][sel, None])  # an option's value; infeasible options sit at -lam
    ok = v > -.5 * d['lam'][sel, None]; c = ok.sum(1, keepdim=True); mv = torch.where(ok, v, 0).sum(1, keepdim=True) / c
    sc = (torch.where(ok, (v - mv) ** 2, 0).sum(1, keepdim=True) / c).sqrt().clamp(min=1e-9)  # spread of the feasible values
    lp = torch.log_softmax((v / (arch['temp'] * sc)).view(B, d['G'], k), -1).view(B, -1)
    return torch.exp(.5 * lp @ d['sub'].T).to(torch.complex64)


def state(arch, d, Ph, gam, bet, psi0):
    if arch['space'] == 'x':
        return qaoa_state(Ph, gam, bet)
    B, N = Ph.shape; k, G = d['k'], d['G']; psi = psi0
    for l in range(gam.shape[1]):
        psi = psi * torch.exp(-1j * gam[:, l, None] * Ph)
        U = mixer(bet[:, l], k, arch['graph'], arch['trotter'])
        for g in range(G):
            psi = torch.einsum('bij,bajc->baic', U, psi.reshape(B, k ** g, k, k ** (G - g - 1))).reshape(B, N)
    return psi


def loss(arch, pr, T, order):
    if arch['loss'] == 'cvar':  # mean energy of the best alpha share of the probability mass
        a = arch['alpha']; ps = torch.gather(pr, 1, order); Ts = torch.gather(T, 1, order)
        w = torch.minimum(ps, (a - (ps.cumsum(1) - ps)).clamp(min=0)); return (w * Ts).sum() / a
    if arch['loss'] == 'gibbs':
        return -torch.logsumexp(torch.log(pr + 1e-30) - arch['eta'] * T, 1).sum()
    return (pr * T).sum()


def cz(arch, d, J, layers):
    """Mean CZ-equivalent two-qubit gates of one circuit. ponytail: no SWAP routing; heavy-hex / square lattices add more."""
    nnz = (counted(arch, d, J) != 0).sum((1, 2)).double().mean().item()
    if arch['space'] == 'x':
        return layers * 2 * nnz
    k, G = d['k'], d['G']
    return 3 * (k - 1) * G + layers * (2 * nnz + 2 * len(pairs(k, arch['graph'])) * arch['trotter'] * G)


def solve(arch, d, sel, angles=None, tau=0.0, keep=None, finetune=0, chunk=160, seed=0):
    """Run the circuit on instances sel. angles: None (ramp start; trained if arch says so), (p,) pair for all, or (B, p)
    pair per instance. tau / keep / finetune: PostNAS pruning of couplers and layers, frozen angles, short fine-tune.
    Returns per-instance P(optimum), whether the best of 4096 shots is optimal, its plan, the angles, two-qubit gates."""
    g = torch.Generator(device=DEV).manual_seed(seed); p = arch['p']; keep = list(range(p)) if keep is None else keep
    out = dict(popt=[], hit=[], x=[], gam=[], bet=[]); X = d['full'] if arch['space'] == 'x' else d['sub']
    for i in range(0, len(sel), chunk):
        s = sel[i:i + chunk]; B = len(s); h, J, off = d['h'][s], d['J'][s], d['off'][s]
        E = energies(h, J, off, X); mu, sd = E.mean(1, keepdim=True), E.std(1, keepdim=True); T = ((E - mu) / sd).float()
        if tau > 0:
            C = counted(arch, d, J).abs(); Jp = torch.where(C >= tau * C.amax((1, 2), keepdim=True), J, 0 * J)
            Jp = Jp if arch['space'] == 'x' else Jp + J * d['same']  # in-group couplers: zero on the subspace anyway
            Ph = ((energies(h, Jp, off, X) - mu) / sd).float()
        else:
            Ph = T
        psi0 = start(arch, d, s) if arch['space'] == 'xy' else None; order = T.argsort(1)
        if angles is None:
            ramp = (torch.arange(p, device=DEV) + .5) / p
            dg, db = (arch.get('dg', .6), arch.get('db', .6)) if arch['angles'] == 'ramp' else (.6, .6)
            gam, bet = (dg * ramp).repeat(B, 1), (db * (1 - ramp)).repeat(B, 1)
            if arch['angles'] == 'ramp':  # the start is the mixer's TOP eigenstate, so anneal to the cost's top unless its sign flips
                gam = -gam
            steps = STEPS if arch['angles'] in ('train', 'transfer') else 0
        else:
            gam, bet = (torch.as_tensor(np.asarray(a), device=DEV, dtype=torch.float32) for a in angles)
            gam, bet = ((x.repeat(B, 1) if x.ndim == 1 else x[i:i + chunk]) for x in (gam, bet)); steps = 0
        gam, bet = gam.clone().requires_grad_(), bet.clone().requires_grad_()
        if steps:
            opt = torch.optim.Adam([gam, bet], lr=LR)
            for _ in range(steps):
                opt.zero_grad(); loss(arch, state(arch, d, T, gam, bet, psi0).abs() ** 2, T, order).backward(); opt.step()
        out['gam'].append(gam.detach().cpu().numpy()); out['bet'].append(bet.detach().cpu().numpy())
        gk, bk = gam.detach()[:, keep].clone().requires_grad_(), bet.detach()[:, keep].clone().requires_grad_()
        if finetune:
            opt = torch.optim.Adam([gk, bk], lr=LR)
            for _ in range(finetune):
                opt.zero_grad(); loss(arch, state(arch, d, Ph, gk, bk, psi0).abs() ** 2, T, order).backward(); opt.step()
        with torch.no_grad():
            pr = state(arch, d, Ph, gk, bk, psi0).abs() ** 2; pr = pr / pr.sum(1, keepdim=True)
            emin = E.min(1).values; isopt = E <= emin[:, None] + 1e-9 * emin.abs()[:, None].clamp(min=1)
            dr = torch.multinomial(pr, SHOTS, replacement=True, generator=g); best = dr[torch.arange(B), torch.gather(E, 1, dr).argmin(1)]
            out['popt'].append((pr * isopt).sum(1).cpu().numpy()); out['hit'].append(isopt[torch.arange(B), best].cpu().numpy())
            out['x'].append(X[best].cpu().numpy().astype(int))
    R = {k_: np.concatenate(v) for k_, v in out.items()}; R['cz'] = cz(arch, d, d['J'][sel] if tau == 0 else _pruned(arch, d, sel, tau), len(keep))
    return R


def _pruned(arch, d, sel, tau):
    J = d['J'][sel]; C = counted(arch, d, J).abs(); return torch.where(C >= tau * C.amax((1, 2), keepdim=True), J, 0 * J)


def angles_for(arch, d, sel):
    """Transfer: train per instance on sel, keep the median angles."""
    R = solve(arch, d, sel); return np.median(R['gam'], 0), np.median(R['bet'], 0)


def suggest(trial, level):
    space = 'xy' if level == 'P3' else trial.suggest_categorical('space', ['x', 'xy'])
    A = dict(space=space, p=trial.suggest_int('p', 1, 6), angles=trial.suggest_categorical('angles', ['train', 'ramp', 'transfer']))
    if space == 'xy':
        A.update(graph=trial.suggest_categorical('graph', ['ring', 'full']), trotter=trial.suggest_int('trotter', 1, 2),
                 init=trial.suggest_categorical('init', ['w', 'warm']))
        if A['init'] == 'warm':
            A['temp'] = trial.suggest_float('temp', .1, 10, log=True)
    if A['angles'] == 'ramp':
        A.update(dg=trial.suggest_float('dg', .05, 2, log=True), db=trial.suggest_float('db', .05, 2, log=True))
    else:
        A['loss'] = trial.suggest_categorical('loss', ['mean', 'cvar', 'gibbs'])
        if A['loss'] == 'cvar':
            A['alpha'] = trial.suggest_float('alpha', .02, .5, log=True)
        if A['loss'] == 'gibbs':
            A['eta'] = trial.suggest_float('eta', .5, 10, log=True)
    A.setdefault('loss', 'mean')
    return A


def evaluate(arch, d, A, B):
    """Score an architecture on instances B (angles: per instance on B, ramp, or transferred from A)."""
    ang = angles_for(arch, d, A) if arch['angles'] == 'transfer' else None
    return solve(arch, d, B, angles=ang), ang


def pareto(st):
    return sorted(st.best_trials, key=lambda t: t.values[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--level', default='P2'); ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--post', type=int, default=40); a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING); torch.manual_seed(0); t0 = time.time()
    val = load('val', a.level, every=8)  # 12 Q3 days spread over the quarter
    n = len(val['inst']); A, Bv = np.arange(0, n, 3), np.setdiff1d(np.arange(n), np.arange(0, n, 3))  # A: transfer source
    print(a.level, n, 'Q3 instances', f'{time.time() - t0:.0f} s', flush=True)

    def nas(trial):
        arch = suggest(trial, a.level); R, _ = evaluate(arch, val, A, Bv)
        trial.set_user_attr('arch', arch); trial.set_user_attr('hit', float(R['hit'].mean()))
        return float(R['popt'].mean()), R['cz']
    st = optuna.create_study(directions=['maximize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12))
    base = [dict(p=1, angles='train', loss='mean'), dict(p=3, angles='train', loss='mean')]
    for b in base:  # the hand-built circuits we ran before (x mixer) and their XY twins, as the first trials
        if a.level != 'P3':
            st.enqueue_trial(dict(space='x', **b))
        st.enqueue_trial(dict(space='xy', graph='ring', trotter=1, init='w', **b))
    st.optimize(nas, n_trials=a.trials, callbacks=[lambda s, t: print('nas', t.number, [round(v, 4) for v in t.values], t.user_attrs['arch'], flush=True)])
    T = st.trials_dataframe(); T['arch'] = [json.dumps(t.user_attrs['arch']) for t in st.trials]; T['pareto'] = T.number.isin([t.number for t in st.best_trials])
    T.to_csv(RESULTS / f'qnas_{a.level}_nas.csv', index=False)
    F = pareto(st); top = max(F, key=lambda t: t.values[0])
    lean = min([t for t in F if t.values[0] >= .8 * top.values[0]], key=lambda t: t.values[1])
    uni = max([t for t in st.trials if t.user_attrs['arch'].get('init', 'w') == 'w'], key=lambda t: t.values[0])  # no classical warm start
    arch = top.user_attrs['arch']; print('NAS pick', top.values, arch, '| lean', lean.values, lean.user_attrs['arch'], flush=True)

    # PostNAS: freeze the winner's trained angles, search layers to drop and couplers to prune, optional fine-tune
    frozen = (lambda R: (R['gam'], R['bet']))(solve(arch, val, Bv)) if arch['angles'] == 'train' else \
        (angles_for(arch, val, A) if arch['angles'] == 'transfer' else None)
    def post(trial):
        keep = [l for l in range(arch['p']) if trial.suggest_categorical(f'keep{l}', [1, 0])] or [arch['p'] - 1]
        tau = trial.suggest_float('tau', 1e-3, .9, log=True) if trial.suggest_categorical('prune', [0, 1]) else 0.0
        ft = trial.suggest_categorical('finetune', [0, 20]); R = solve(arch, val, Bv, angles=frozen, tau=tau, keep=keep, finetune=ft)
        trial.set_user_attr('cfg', dict(keep=keep, tau=tau, finetune=ft)); return float(R['popt'].mean()), R['cz']
    sp = optuna.create_study(directions=['maximize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=10))
    sp.enqueue_trial({**{f'keep{l}': 1 for l in range(arch['p'])}, 'prune': 0, 'finetune': 0})  # the unpruned circuit
    sp.optimize(post, n_trials=a.post, callbacks=[lambda s, t: print('post', t.number, [round(v, 4) for v in t.values], t.user_attrs['cfg'], flush=True)])
    P_ = sp.trials_dataframe(); P_['cfg'] = [json.dumps(t.user_attrs['cfg']) for t in sp.trials]; P_['pareto'] = P_.number.isin([t.number for t in sp.best_trials])
    P_.to_csv(RESULTS / f'qnas_{a.level}_postnas.csv', index=False)
    full_popt = sp.trials[0].values[0]
    cut = min([t for t in sp.best_trials if t.values[0] >= .95 * full_popt] or [sp.trials[0]], key=lambda t: t.values[1])
    cfg = cut.user_attrs['cfg']; print('PostNAS pick', cut.values, cfg, flush=True)

    # Q4, once: the picks against the hand-built circuits
    test = load('test', a.level); nt = len(test['inst']); Q3 = np.arange(n)
    rows = []
    cands = ([('qaoa_x_p1', dict(space='x', p=1, angles='train', loss='mean'), {}), ('qaoa_x_p3', dict(space='x', p=3, angles='train', loss='mean'), {})] if a.level != 'P3' else []) + \
        [('xy_ring_p1', dict(space='xy', graph='ring', trotter=1, init='w', p=1, angles='train', loss='mean'), {}),
         ('nas_best', arch, {}), ('nas_lean', lean.user_attrs['arch'], {}), ('nas_uniform', uni.user_attrs['arch'], {}), ('postnas', arch, cfg)]
    cands.append(('random_onehot', dict(space='xy', graph='ring', trotter=1, init='w', p=0, angles='ramp', loss='mean'), {}))  # control: 4096 uniform valid plans
    for nm, ar in (('nas_best', arch), ('nas_lean', lean.user_attrs['arch'])):  # control: the warm start alone, no circuit
        if ar.get('init') == 'warm' and not any(c[0] == 'warm_only' for c in cands):
            cands.append(('warm_only', {**ar, 'p': 0, 'angles': 'ramp'}, {}))
    for name, ar, pc in cands:
        t = time.time(); ang = angles_for(ar, val, Q3) if ar['angles'] == 'transfer' else None
        if pc:
            fz = ang if ang is not None else ((lambda R: (R['gam'], R['bet']))(solve(ar, test, np.arange(nt))) if ar['angles'] == 'train' else None)
            R = solve(ar, test, np.arange(nt), angles=fz, tau=pc['tau'], keep=pc['keep'], finetune=pc['finetune'])
        else:
            R = solve(ar, test, np.arange(nt), angles=ang)
        eur = [qubo.realised(x, V, HR, a.level) if qubo.feasible(x, V, lin, quad, off, H, a.level) else 0.0
               for x, (V, lin, quad, off, lam), (_, _, H, HR) in zip(R['x'], test['P'], test['inst'])]
        rows.append(dict(name=name, arch=json.dumps(ar), post=json.dumps(pc), popt=R['popt'].mean(), hit=R['hit'].mean(), eur=sum(eur),
                         cz=R['cz'], s=time.time() - t))
        print('test', rows[-1], flush=True)
    if a.level != 'P3':  # the strongest earlier circuit, BF-DCQO (qest.qsota), on the full space: 4 steps of first-order CD
        from qest.qsota import bfdcqo  # ponytail: per circuit, every 2-body Pauli term gives two Y_i Z_j rotations of 2 CZ each
        t = time.time(); Ef = energies(test['h'], test['J'], test['off'], test['full']); z, stt = bfdcqo(Ef)
        xb = ((z.cpu().numpy()[:, None] >> np.arange(test['m'])) & 1)
        eur = [qubo.realised(x, V, HR, a.level) if qubo.feasible(x, V, lin, quad, off, H, a.level) else 0.0
               for x, (V, lin, quad, off, lam), (_, _, H, HR) in zip(xb, test['P'], test['inst'])]
        rows.append(dict(name='bfdcqo', arch='{}', post='{}', popt=float(np.mean(stt['p_opt'])), hit=float(np.mean(stt['best_sample_is_opt'])),
                         eur=sum(eur), cz=4 * 4 * (test['J'] != 0).sum((1, 2)).double().mean().item(), s=time.time() - t)); del Ef
        print('test', rows[-1], flush=True)
    E = energies(test['h'], test['J'], test['off'], test['sub']); xo = test['sub'][E.argmin(1)].cpu().numpy().astype(int)
    ex = sum(qubo.realised(x, V, HR, a.level) for x, (V, *_), (_, _, H, HR) in zip(xo, test['P'], test['inst']))
    D = pd.DataFrame(rows); D['eur_pct_exact'] = D.eur / ex * 100
    D['chance_full'] = (1.0 / 2 ** test['m']); D['chance_onehot'] = 1.0 / test['k'] ** test['G']
    D.to_csv(RESULTS / f'qnas_{a.level}_test.csv', index=False)
    (RESULTS / f'qnas_{a.level}.json').write_text(json.dumps(dict(nas=dict(values=top.values, arch=arch), lean=dict(values=lean.values, arch=lean.user_attrs['arch']),
                                                                   uniform=dict(values=uni.values, arch=uni.user_attrs['arch']),
                                                                   postnas=dict(values=cut.values, cfg=cfg, unpruned=full_popt), exact_eur=ex), indent=1))
    print(D.drop(columns=['arch', 'post']).round(4).to_string(), f'\n{time.time() - t0:.0f} s')


def _check():
    """The XY mixer keeps every state one-hot and unitary; the subspace optimum equals exact enumeration."""
    U = mixer(torch.tensor([.3, 1.1], device=DEV), 5, 'ring', 2)
    assert torch.allclose(U @ U.conj().transpose(1, 2), torch.eye(5, dtype=torch.complex64, device=DEV).expand(2, 5, 5), atol=1e-5)
    d = load('val', 'P2', every=30, ndays=1)
    E = energies(d['h'], d['J'], d['off'], d['sub'])
    for b in range(3):
        V, lin, quad, off, _ = d['P'][b]; assert np.isclose(E[b].min().item(), qubo.exact(V, lin, quad, off, 'P2')[0])
    arch = dict(space='xy', graph='ring', trotter=1, init='w', p=2, angles='ramp', loss='mean')
    R = solve(arch, d, np.arange(4)); assert R['x'].sum(1).tolist() == [d['G']] * 4 and 0 < R['popt'].min() <= 1
    print('qnas check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()
