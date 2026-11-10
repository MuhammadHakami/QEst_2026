r"""Quantum and quantum-inspired solvers on dense energy tables E (B, 2^n), simulated exactly on the GPU (run with
CUDA_VISIBLE_DEVICES=1: GPU 0 hosts vLLM). Ported unchanged from our Makkah entry (energy/qsota.py), where they ran on
the portfolio's hour problems; here they take the slide-11 puzzle (qest.qubo, P1 9 / P2 15 qubits).

'bfdcqo': bias-field digitized counterdiabatic quantum optimisation (Cadavid et al., arXiv:2409.04477, Commun. Phys.
  2025; used on trapped ions for large portfolios in arXiv:2602.23976, 2026; the cardinality-portfolio notebook
  research/third_party/bf-dcqo-qiskit, 2026). Non-variational: no angles to train. H_ad(l) = (1 - l) H_i + l H_f with the
  biased driver H_i = sum_i (-X_i - hb_i Z_i) and H_f the hour's cost in Pauli form. Only the first-order
  counterdiabatic term is applied (impulse regime): A = i alpha [H_i, H_f], which for a cost term c_S Z_S is
  sum_(i in S) 2 alpha (-1) c_S Y_i Z_(S \ i), with alpha(l) = -G1/G2, G1 = |[H_i, H_f]|_F^2, G2 = |[H_ad, [H_i, H_f]]|_F^2.
  The repository's closed form for alpha covers 2-body costs only; here G1, G2 come from exact Pauli algebra (symplectic
  products), so 3- and 4-body terms count. After each run the bias hb_i <- <sigma^z_i> over the best 10 % of the shots
  starts the next run nearer the good states.
'pce': Pauli correlation encoding (Sciorilli et al., Nat. Commun. 16, 476, 2025; for unit commitment arXiv:2605.17145 and
  arXiv:2609.36999, for power-demand portfolios arXiv:2607.24722, all 2026). The m bits of an hour are signs of 2-body
  correlators X_aX_b, Y_aY_b, Z_aZ_b on k qubits (m <= 3 k(k-1)/2: 17 bits on 4 qubits, 45 bits on 6), s_v = tanh(a <P_v>).
  The loss is the cost's exact multilinear form in s (equal to the cost at every bitstring), trained by Adam through a
  layered RY-RZ-CZ circuit, many restarts at once; signs are read off, then the one-bit polish. Shot-aware option
  (Boussahi, arXiv:2610.19544, Oct 2026): a QPU reads each correlator from S shots, and the sharpness alpha = 0.851 sqrt(S)
  makes the relaxed loss track the cut decoded from S shots; decode_shots reads the signs from S simulated shots.

October 2026 additions:
'falqon': feedback-based quantum optimisation (Magann et al., PRL 129, 250502, 2022; for unit commitment in
  research/third_party/Unit-Commitment-Quantum-Opt, 2026). QAOA layers whose mixer angle comes from a measured feedback
  law, beta_(k+1) = -gain <i[H_d, H_p]>_k, so the energy never rises: no training at all. With H_d = sum X and a diagonal
  H_p, <i[H_d, H_p]> = -2 Im <H_d psi | H_p psi>.
'pt': parallel tempering (replica exchange), the classical bar: on 180 QUBO benchmarks with moved optima it reached every
  optimum where branch-and-cut stalled (Hellstern et al., arXiv:2610.35126, Oct 2026).
"""
import itertools, math

import numpy as np
import torch

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def polish_qubo(h, J, X):
    """One-bit descent on QUBO coefficients: h (B, m), J (B, m, m) upper-triangular, X (B, m) in {0, 1} -> X."""
    S = J + J.transpose(1, 2)
    while True:
        d = (1 - 2 * X) * (h + torch.einsum('bmn,bn->bm', S, X))  # energy change of flipping each bit
        k = d.argmin(1); better = d[torch.arange(len(X)), k] < -1e-9
        if not better.any(): return X
        X = X.clone(); r = torch.nonzero(better).flatten(); X[r, k[r]] = 1 - X[r, k[r]]


def polish(E, z):
    """One-bit descent on the energy table: (B,) states -> (B,) states (from the Makkah repo's energy/qport.py)."""
    n = int(math.log2(E.shape[1])); b = torch.arange(len(z), device=DEV)
    while True:
        nb = z[:, None] ^ (1 << torch.arange(n, device=DEV))[None]; en = E[b[:, None], nb]; k = en.argmin(1)
        better = en[b, k] < E[b, z] - 1e-12
        if not better.any(): return z
        z = torch.where(better, nb[b, k], z)


def walsh(E):
    """Batch fast Walsh-Hadamard: E (B, 2^n) -> c (B, 2^n) with E(z) = sum_S c_S prod_(i in S) (1 - 2 z_i)."""
    a = E.clone(); B, N = a.shape; h = 1
    while h < N:
        a = a.view(B, -1, 2, h); a = torch.stack([a[:, :, 0] + a[:, :, 1], a[:, :, 0] - a[:, :, 1]], 2).reshape(B, N); h *= 2
    return a / N


# ---------- Pauli algebra on (x, z) bitmasks: P = prod_q (I, X, Z, Y for (x_q, z_q) = 00, 10, 01, 11) ----------
def _g(x1, z1, x2, z2):  # Aaronson-Gottesman phase exponent (powers of i) of P1 P2 = i^g P3, per qubit, summed
    g = np.where((x1 == 1) & (z1 == 1), z2 - x2, 0) + np.where((x1 == 1) & (z1 == 0), z2 * (2 * x2 - 1), 0) + np.where((x1 == 0) & (z1 == 1), x2 * (1 - 2 * z2), 0)
    return g.sum(-1)


def _bits(m, n): return (m[:, None] >> np.arange(n)[None]) & 1


def commutator(A, B, n):
    """[A, B] for Pauli sums given as (x, z, coef) arrays -> combined (x, z, coef)."""
    (xa, za, ca), (xb, zb, cb) = A, B
    XA, ZA, XB, ZB = _bits(xa, n), _bits(za, n), _bits(xb, n), _bits(zb, n)
    anti = ((XA[:, None] & ZB[None]) ^ (ZA[:, None] & XB[None])).sum(-1) % 2 == 1  # anticommuting pairs
    i, j = np.nonzero(anti)
    if not len(i): return np.zeros(0, int), np.zeros(0, int), np.zeros(0, complex)
    ph = 1j ** (_g(XA[i], ZA[i], XB[j], ZB[j]) % 4)
    return _combine(xa[i] ^ xb[j], za[i] ^ zb[j], 2 * ph * ca[i] * cb[j])


def _combine(x, z, c):
    key = (x.astype(np.int64) << 32) | z.astype(np.int64); u, inv = np.unique(key, return_inverse=True); s = np.zeros(len(u), complex); np.add.at(s, inv, c)
    keep = np.abs(s) > 1e-14; return (u[keep] >> 32), (u[keep] & 0xFFFFFFFF), s[keep]


def frob2(P): return float(np.sum(np.abs(P[2]) ** 2))  # |O|_F^2 / 2^n


def alpha(zS, cS, hb, lam, n):
    """alpha_1(lam) = -G1/G2 for H_i = sum(-X_i - hb_i Z_i), H_f = sum c_S Z_S (zS: masks)."""
    I = np.arange(n); Hi = (np.r_[1 << I, np.zeros(n, int)], np.r_[np.zeros(n, int), 1 << I], np.r_[-np.ones(n), -np.asarray(hb, float)].astype(complex))
    Hf = (np.zeros(len(zS), int), np.asarray(zS, int), np.asarray(cS, float).astype(complex))
    O1 = commutator(Hi, Hf, n); Had = _combine(np.r_[Hi[0], Hf[0]], np.r_[Hi[1], Hf[1]], np.r_[(1 - lam) * Hi[2], lam * Hf[2]])
    g2 = frob2(commutator(Had, O1, n)); return -frob2(O1) / g2 if g2 > 1e-14 else 0.0


def _pauli_terms(c, tol=1e-9):
    """Walsh coefficients of one instance -> (masks, coefficients) of the non-constant terms."""
    c = c.cpu().numpy(); k = np.nonzero(np.abs(c) > tol * np.abs(c[1:]).max())[0]; k = k[k > 0]; return k, c[k]


def _yz_rotate(psi, i, R, theta, sgnR, n):
    """psi <- exp(-i theta Y_i Z_R) psi for a batch; theta (B,), sgnR (N,) = (-1)^|z & R|."""
    N = psi.shape[1]; idx = torch.arange(N, device=psi.device); zi = ((idx >> i) & 1).to(psi.real.dtype)
    P = 1j * (2 * zi - 1) * sgnR * psi[:, idx ^ (1 << i)]
    return torch.cos(theta)[:, None] * psi - 1j * torch.sin(theta)[:, None] * P


def bfdcqo(E, iters=6, steps=4, shots=4096, top=.1, seed=0):
    """BF-DCQO on each instance's energy table E (B, 2^n) -> best state per instance (after polish), stats."""
    B, N = E.shape; n = int(math.log2(N)); g = torch.Generator(device=DEV).manual_seed(seed)
    C = walsh(E); terms = []
    for b in range(B):
        k, c = _pauli_terms(C[b]); s = np.abs(c).max(); terms.append((k, c / s))
    idx = torch.arange(N, device=DEV); hb = np.zeros((B, n)); best = E.argmin(1) * 0; be = torch.full((B,), float('inf'), device=DEV, dtype=E.dtype); hist = []
    lam_t = [(np.sin(np.pi / 2 * np.sin(np.pi * (s_ + .5) / (2 * steps)) ** 2) ** 2) for s_ in range(steps)]
    dlam = np.diff(np.r_[0.0, [(np.sin(np.pi / 2 * np.sin(np.pi * (s_ + 1) / (2 * steps)) ** 2) ** 2) for s_ in range(steps)]])  # the step's share of d lambda
    sgn = {}
    for it in range(iters):
        th0 = np.arctan2(1.0, hb)  # ground state of -X - hb Z on each qubit: RY(th0)|0>
        cs, sn = torch.tensor(np.cos(th0 / 2), device=DEV), torch.tensor(np.sin(th0 / 2), device=DEV)
        bits = ((idx[:, None] >> torch.arange(n, device=DEV)[None]) & 1).bool()
        psi = torch.exp(torch.where(bits[None], sn.log()[:, None, :], cs.log()[:, None, :]).sum(-1)).to(torch.complex128)
        for lam, dl in zip(lam_t, dlam):
            al = np.array([alpha(k, c, hb[b], lam, n) for b, (k, c) in enumerate(terms)])
            rot = {}  # (i, R) -> theta per instance: A = sum 2 alpha (-1) c_S Y_i Z_(S\i), theta = d lambda x coefficient
            for b, (k, c) in enumerate(terms):
                for S, cS in zip(k, c):
                    for i in range(n):
                        if S >> i & 1: rot.setdefault((i, int(S) ^ (1 << i)), np.zeros(B))[b] += -2 * al[b] * cS * dl
            for (i, R), th in rot.items():
                if R not in sgn: sgn[R] = (1 - 2 * (sum(((idx & R) >> q) & 1 for q in range(n)) % 2)).double()  # (-1)^|z & R|
                psi = _yz_rotate(psi, i, R, torch.tensor(th, device=DEV), sgn[R], n)
        pr = (psi.abs() ** 2).real; pr = pr / pr.sum(1, keepdim=True)
        draws = torch.multinomial(pr.float(), shots, replacement=True, generator=g); e = torch.gather(E, 1, draws)
        k_ = e.argmin(1); cand = draws[torch.arange(B), k_]; ec = e[torch.arange(B), k_]; upd = ec < be; best = torch.where(upd, cand, best); be = torch.where(upd, ec, be)
        nt = max(1, int(top * shots)); o = e.argsort(1)[:, :nt]; sel = torch.gather(draws, 1, o)
        hb = (1 - 2 * ((sel[..., None] >> torch.arange(n, device=DEV)) & 1)).double().mean(1).cpu().numpy()  # <sigma^z> over the best shots
        emin = E.min(1).values; hist.append(dict(iter=it, p_opt=(pr * (E <= emin[:, None] + 1e-9 * emin.abs()[:, None].clamp(min=1))).sum(1).tolist()))
    emin = E.min(1).values; isopt = E <= emin[:, None] + 1e-9 * emin.abs()[:, None].clamp(min=1)
    stats = dict(p_opt=hist[-1]['p_opt'], p_opt_random=(isopt.sum(1) / N).tolist(), best_sample_is_opt=(be <= emin + 1e-9 * emin.abs().clamp(min=1)).tolist(), p_opt_by_iter=[h['p_opt'] for h in hist])
    return polish(E, best), stats


def pce_layout(m, k=None):
    """Smallest k with 3 C(k, 2) >= m (or the k given), and variable v -> (pair a, b, Pauli 'X'|'Y'|'Z')."""
    if not k:  # one qubit above the minimum: on the Hajj hours 4 qubits gave the optimum in 14/24 hours, 5 in 16/24 (8 layers on 6: 17/24)
        k = 2
        while 3 * k * (k - 1) // 2 < m: k += 1
        k += 1
    slots = [(a, b, P) for P in 'ZXY' for a, b in itertools.combinations(range(k), 2)]
    return k, slots[:m]


def pce(E, layers=4, steps=300, restarts=24, lr=.05, seed=0, k=None, alpha=None, decode_shots=None, qubo=None, groups=None):
    """Pauli correlation encoding on each instance (B, 2^m) -> best state per instance (after polish), stats. alpha: the
    loss sharpness (default 1.5 k); decode_shots: read each correlator's sign from that many simulated shots.
    qubo=(h (B, m), J (B, m, m) upper-triangular): the cost given by its QUBO coefficients instead of the 2^m table, so m
    is not limited by memory (P3: 30 bits on 6 qubits); E is then ignored. groups: one-hot groups of bit indices (problem-aware
    decoding, after arXiv:2605.20163): in each group the bit with the most negative correlator is set, so every decoded plan is
    valid and the penalty terms never bite; the sign decode is kept for the bits outside groups."""
    if qubo is not None:
        h, J = (torch.as_tensor(a, device=DEV, dtype=torch.float32) for a in qubo); B, m = h.shape
        sc = h.abs().amax(1, keepdim=True).clamp(min=1e-9); h, J = h / sc, J / sc[..., None]
    else:
        B, N = E.shape; m = int(math.log2(N))
    k, slots = pce_layout(m, k); K = 1 << k; g = torch.Generator(device=DEV).manual_seed(seed)
    if qubo is None:
        C = walsh(E) / E.std(1, keepdim=True); Smask = torch.nonzero(C.abs().amax(0) > 1e-9 * C.abs().max()).flatten(); Smask = Smask[Smask > 0]
        terms = [(int(S), [i for i in range(m) if int(S) >> i & 1]) for S in Smask.tolist()]; coef = C[:, Smask].float()  # (B, T)
    R = restarts; th = (torch.rand((B, R, layers, k, 2), device=DEV, generator=g) * 2 * math.pi).requires_grad_()
    idx = torch.arange(K, device=DEV); zb = ((idx[:, None] >> torch.arange(k, device=DEV)[None]) & 1).float()  # (K, k)
    a_scale = alpha or 1.5 * k  # Sciorilli et al.: alpha ~ 1.5 n^(floor(k/2)) for 2-body correlators; 0.851 sqrt(S) for S shots (2610.19544)
    def state(th):
        psi = torch.zeros(B, R, K, dtype=torch.complex64, device=DEV); psi[..., 0] = 1
        for l in range(layers):
            for q in range(k):
                ry, rz = th[:, :, l, q, 0], th[:, :, l, q, 1]; c, s = torch.cos(ry / 2)[..., None], torch.sin(ry / 2)[..., None]
                v = psi.view(B, R, K >> (q + 1), 2, 1 << q); a0, a1 = v[..., 0, :], v[..., 1, :]
                cc, ss = c[..., None], s[..., None]
                psi = torch.stack([cc * a0 - ss * a1, ss * a0 + cc * a1], -2).reshape(B, R, K)
                ph = torch.exp(-.5j * rz)[..., None]; psi = torch.where(((idx >> q) & 1).bool(), psi * ph.conj(), psi * ph)
            for q in range(k): psi = psi * torch.where((((idx >> q) & 1) & ((idx >> ((q + 1) % k)) & 1)).bool(), -1.0, 1.0)  # CZ ring
        return psi
    def corr(psi):
        p = psi.abs() ** 2; out = []
        for a, b, P in slots:
            sa, sb = 1 - 2 * zb[:, a], 1 - 2 * zb[:, b]
            if P == 'Z': out.append((p * sa * sb).sum(-1))
            else:
                fl = psi[..., idx ^ ((1 << a) | (1 << b))]; amp = (psi.conj() * fl)
                out.append(amp.real.sum(-1) if P == 'X' else -(amp * (sa * sb)).real.sum(-1))
        return torch.stack(out, -1)  # (B, R, m)
    def loss_of(s):
        if qubo is not None:  # x = (1 - s) / 2: the QUBO's own quadratic form, exact at every sign pattern
            x = (1 - s) / 2; return (x * h[:, None]).sum(-1) + torch.einsum('brm,bmn,brn->br', x, J, x)
        tot = 0
        for t, (S, vs) in enumerate(terms): tot = tot + coef[:, t, None] * torch.prod(s[..., vs], -1)
        return tot  # (B, R): the cost's multilinear form (up to its constant)
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); loss_of(torch.tanh(a_scale * corr(state(th)))).sum().backward(); opt.step()
    with torch.no_grad():
        s = corr(state(th))
        if decode_shots:  # each correlator is the mean of S +-1 outcomes with P(+1) = (1 + c) / 2
            s = 2 * torch.distributions.Binomial(decode_shots, ((1 + s.clamp(-1, 1)) / 2)).sample() / decode_shots - 1
        if qubo is not None:
            X = (s < 0).float()
            if groups:  # problem-aware decoding: exactly one bit per one-hot group, the one the state leans to most
                X = torch.zeros_like(X)
                for g_ in groups:
                    gi = torch.as_tensor(g_, device=DEV); X[..., gi] = torch.nn.functional.one_hot(s[..., gi].argmin(-1), len(g_)).float()
            e = (X * h[:, None]).sum(-1) + torch.einsum('brm,bmn,brn->br', X, J, X)
            return polish_qubo(h, J, X[torch.arange(B), e.argmin(1)]), dict(qubits=k, bits=m, restarts=R)
        z = ((s < 0).long() * (1 << torch.arange(m, device=DEV))).sum(-1)  # s = +1 <-> bit 0
        e = torch.gather(E, 1, z); best = z[torch.arange(B), e.argmin(1)]
        emin = E.min(1).values; isopt = E <= emin[:, None] + 1e-9 * emin.abs()[:, None].clamp(min=1)
        stats = dict(qubits=k, bits=m, restarts=R, rounded_is_opt=(torch.gather(E, 1, best[:, None])[:, 0] <= emin + 1e-9 * emin.abs().clamp(min=1)).tolist(),
                     restarts_at_opt=(torch.gather(isopt, 1, z).float().mean(1)).tolist(), p_opt_random=(isopt.sum(1) / N).tolist())
    return polish(E, best), stats


def falqon(E, layers=300, dt=.1, gain=1.0, shots=4096, seed=0):
    """FALQON on each instance (B, 2^n): |+>^n, then layers of exp(-i beta_k dt H_d) exp(-i dt H_p) with beta from the
    feedback law; sample, best shot, one-bit polish. Returns states, stats."""
    B, N = E.shape; n = int(math.log2(N)); g = torch.Generator(device=DEV).manual_seed(seed)
    Es = ((E - E.mean(1, keepdim=True)) / E.std(1, keepdim=True)).float()
    psi = torch.full((B, N), N ** -.5, dtype=torch.complex64, device=DEV); beta = torch.zeros(B, device=DEV); ph = torch.exp(-1j * dt * Es)
    flip = lambda v, q: v.view(B, N >> (q + 1), 2, 1 << q).flip(2).reshape(B, N)
    for _ in range(layers):
        psi = psi * ph; c, s = torch.cos(beta * dt)[:, None], torch.sin(beta * dt)[:, None]
        for q in range(n): psi = c * psi - 1j * s * flip(psi, q)  # exp(-i beta dt X_q)
        hd = sum(flip(psi, q) for q in range(n)); beta = 2 * gain * (hd.conj() * (Es * psi)).sum(1).imag  # -gain <A>, <A> = -2 Im<H_d psi|H_p psi>
    pr = psi.abs() ** 2; pr = pr / pr.sum(1, keepdim=True)
    draws = torch.multinomial(pr, shots, replacement=True, generator=g); best = draws[torch.arange(B), torch.gather(E, 1, draws).argmin(1)]
    emin = E.min(1).values; tol = 1e-9 * emin.abs().clamp(min=1); isopt = E <= (emin + tol)[:, None]
    return polish(E, best), dict(p_opt=(pr * isopt).sum(1).tolist(), p_opt_random=(isopt.sum(1) / N).tolist(),
                                 best_sample_is_opt=(torch.gather(E, 1, best[:, None])[:, 0] <= emin + tol).tolist(), energy=(pr * Es).sum(1).tolist())


def ptemper(E, replicas=16, reads=8, sweeps=300, t_hot=3.0, t_cold=.02, seed=0):
    """Parallel tempering on the energy table (B, 2^n): replicas at fixed geometric temperatures (units: the median one-bit
    energy change, as energy.qport.anneal), Metropolis sweeps, neighbour swaps every sweep; best seen, one-bit polish."""
    B, N = E.shape; n = int(math.log2(N)); g = torch.Generator(device=DEV).manual_seed(seed); ar = torch.arange(B, device=DEV)
    probe = torch.randint(0, N, (B, 256), device=DEV, generator=g); fl = probe[..., None] ^ (1 << torch.arange(n, device=DEV))
    scale = (E[ar[:, None, None], fl] - E[ar[:, None], probe][..., None]).abs().flatten(1).median(1).values.clamp(min=1e-12)
    En = E / scale[:, None]; T = torch.tensor(np.geomspace(t_hot, t_cold, replicas), device=DEV, dtype=En.dtype)  # (R,)
    b = ar[:, None, None]; z = torch.randint(0, N, (B, reads, replicas), device=DEV, generator=g); best = z[..., -1].clone(); be = En[b[..., 0], best]
    for s in range(sweeps):
        for q in range(n):
            zf = z ^ (1 << q); d = En[b, zf] - En[b, z]
            z = torch.where((d < 0) | (torch.rand(d.shape, device=DEV, generator=g) < torch.exp(-d.clamp(min=0) / T)), zf, z)
        e = En[b, z]; o = s % 2; i = torch.arange(o, replicas - 1, 2, device=DEV)  # swap replicas i, i+1 with prob min(1, exp((1/T_i - 1/T_j)(e_i - e_j)))
        acc = torch.rand((B, reads, len(i)), device=DEV, generator=g) < torch.exp(((1 / T[i] - 1 / T[i + 1]) * (e[..., i] - e[..., i + 1])).clamp(max=0))
        zi, zj = z[..., i], z[..., i + 1]; z[..., i] = torch.where(acc, zj, zi); z[..., i + 1] = torch.where(acc, zi, zj)
        e = En[b, z]; k = e.flatten(1).argmin(1); zc = z.flatten(1)[ar, k]; ec = e.flatten(1)[ar, k]
        upd = ec < be.min(1).values; best[:, 0] = torch.where(upd, zc, best[:, 0]); be[:, 0] = torch.where(upd, ec, be[:, 0])
    return polish(E, best[ar, be.argmin(1)])


# --- batched GPU QAOA (linear-ramp start + Adam, optional warm start) and simulated quantum annealing (path-integral MC),
# --- from the Makkah repo energy/qport.py, unchanged
def qaoa_state(Es, gam, bet, warm=None):
    """QAOA state for a batch: Es (B, 2^n) cost, gam/bet (B, p); warm (B, n) probabilities start RY(th)|0> and tilt the
    mixer to RY(th) RZ(2 beta) RY(-th) (plain RX(2 beta) at th = pi/2). Qubit q = bit q of the index."""
    B, N = Es.shape; n = int(math.log2(N)); p = gam.shape[1]
    c = torch.full((B, n), .5, device=Es.device) if warm is None else torch.as_tensor(warm, device=Es.device, dtype=torch.float32).clamp(.1, .9)  # Egger's epsilon
    th = 2 * torch.asin(c.sqrt()); cs, sn = torch.cos(th / 2), torch.sin(th / 2)
    bits = ((torch.arange(N, device=Es.device)[:, None] >> torch.arange(n, device=Es.device)[None]) & 1).bool()
    psi = torch.exp(torch.where(bits[None], sn.log()[:, None, :], cs.log()[:, None, :]).sum(-1)).to(torch.complex64)
    for k in range(p):
        psi = psi * torch.exp(-1j * gam[:, k, None] * Es)
        eb, ebc, sb = torch.exp(-1j * bet[:, k]), torch.exp(1j * bet[:, k]), torch.sin(bet[:, k])
        for q in range(n):
            c2, s2, cq = cs[:, q] ** 2, sn[:, q] ** 2, cs[:, q] * sn[:, q]
            u00, u11, u01 = (c2 * eb + s2 * ebc)[:, None, None], (s2 * eb + c2 * ebc)[:, None, None], (-2j * cq * sb)[:, None, None]
            v = psi.view(B, N >> (q + 1), 2, 1 << q); a0, a1 = v[:, :, 0], v[:, :, 1]
            psi = torch.stack([u00 * a0 + u01 * a1, u01 * a0 + u11 * a1], 2).reshape(B, N)
    return psi


def qaoa(E, p=3, steps=150, lr=.05, warm=None, shots=4096, seed=0, angles=None):
    """QAOA on the exact diagonal, a batch of instances at once. E (B, 2^n) float64; warm (B, n) bit probabilities or None.
    Angles: linear-ramp start (Sack & Serbyn 2021), then Adam on the expectation through the simulation. Returns the best
    sampled state per instance (after a one-bit polish) and how often the state vector hits the exact optimum."""
    B, N = E.shape; g = torch.Generator(device=DEV).manual_seed(seed)
    Es = ((E - E.mean(1, keepdim=True)) / E.std(1, keepdim=True)).float()
    ramp = (torch.arange(p, device=DEV) + .5) / p
    gam = (.6 * ramp).repeat(B, 1).requires_grad_(); bet = (.6 * (1 - ramp)).repeat(B, 1).requires_grad_()
    if angles is not None:  # transferred from smaller instances of the same family (angle concentration): no training, forward only
        gam, bet = (torch.tensor(np.asarray(a, float), device=DEV, dtype=torch.float32) for a in angles); steps = 0
        if gam.ndim == 1: gam, bet = gam.repeat(B, 1), bet.repeat(B, 1)  # one set for all, or (B, p) each (learned per instance: energy.hourbench)
    evolve = lambda gm, bt: qaoa_state(Es, gm, bt, warm)

    opt = torch.optim.Adam([gam, bet], lr=lr) if steps else None
    for _ in range(steps):
        opt.zero_grad(); loss = (evolve(gam, bet).abs() ** 2 * Es).sum(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = evolve(gam, bet).abs() ** 2; pr = pr / pr.sum(1, keepdim=True)
        draws = torch.multinomial(pr, shots, replacement=True, generator=g); e = torch.gather(E, 1, draws)
        best = draws[torch.arange(B), e.argmin(1)]; z = polish(E, best)
        emin = E.min(1).values; isopt = (E <= emin[:, None] + 1e-9 * emin.abs()[:, None].clamp(min=1))
        stats = dict(p_opt=(pr * isopt).sum(1).tolist(), p_opt_random=(isopt.sum(1) / N).tolist(),
                     best_sample_is_opt=(torch.gather(E, 1, best[:, None])[:, 0] <= emin + 1e-9 * emin.abs().clamp(min=1)).tolist(),
                     angles=dict(gamma=gam.detach().cpu().numpy().round(3).tolist(), beta=bet.detach().cpu().numpy().round(3).tolist()))
    return z, stats


def anneal(E, quantum=True, P=16, reads=32, sweeps=300, seed=0):
    """Simulated quantum annealing (path-integral Monte Carlo, transverse field Gamma 3 -> 1e-3, T fixed) or, with
    quantum=False, classical annealing (one replica, T geometric 3 -> 1e-3), on the energy table. Units: the median size
    of a one-bit energy change, so one schedule fits every hour."""
    B, N = E.shape; n = int(math.log2(N)); g = torch.Generator(device=DEV).manual_seed(seed); b = torch.arange(B, device=DEV)[:, None, None]
    probe = torch.randint(0, N, (B, 256), device=DEV, generator=g); flips = probe[..., None] ^ (1 << torch.arange(n, device=DEV))
    scale = (E[torch.arange(B, device=DEV)[:, None, None], flips] - E[torch.arange(B, device=DEV)[:, None], probe][..., None]).abs().flatten(1).median(1).values.clamp(min=1e-12)
    En = E / scale[:, None]; P = P if quantum else 1
    z = torch.randint(0, N, (B, reads, P), device=DEV, generator=g); best = z[:, :, 0].clone(); be = En[b[:, :, 0], best]
    for s in range(sweeps):
        f = s / max(sweeps - 1, 1)
        if quantum: T, Gam = .05, 3.0 * (1e-3 / 3.0) ** f; Jp = -(P * T / 2) * math.log(math.tanh(Gam / (P * T)))
        else: T = 3.0 * (1e-3 / 3.0) ** f
        for q in torch.randperm(n, generator=torch.Generator().manual_seed(seed + s)).tolist():
            zf = z ^ (1 << q); d = (En[b, zf] - En[b, z]) / P
            if quantum:
                sp = 1 - 2 * ((z >> q) & 1).float(); d = d + 2 * Jp * sp * (sp.roll(1, -1) + sp.roll(-1, -1))
            acc = (d < 0) | (torch.rand(d.shape, device=DEV, generator=g) < torch.exp(-d.clamp(min=0) / T))
            z = torch.where(acc, zf, z)
        e = En[b, z]; k = e.argmin(-1); zc = torch.gather(z, -1, k[..., None])[..., 0]; ec = torch.gather(e, -1, k[..., None])[..., 0]
        upd = ec < be; best = torch.where(upd, zc, best); be = torch.where(upd, ec, be)
    return polish(E, best[torch.arange(B, device=DEV), be.argmin(1)])


TRAIN_MAX_Q = 19  # above this, training through the simulation needs > 100 GB for 24 hours at p = 3: angles are transferred
ANGLES = {}  # trained angles by solver, kept for transfer to larger instances

