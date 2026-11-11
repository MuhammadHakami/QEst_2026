"""The organisers' quantum-track puzzle on IBM hardware: LR-QAOA on the 15-qubit P2 instance of every hour of a Q4 day
(battery, hybrid battery, hybrid PV each pick one of sell / buy / idle / mFRR up / mFRR down), retrospective prices (slide 9).
One circuit per hour, fixed linear-ramp schedule (no angle training on the device), `--shots` shots; the best feasible sample
is the plan. Controls with the same number of shots: uniform random bit strings; the exact optimum by enumeration.
Credentials: IBM_QUANTUM_TOKEN (IBM Cloud API key) and IBM_QUANTUM_CRN in the gitignored .env; never printed.
usage: .venv/bin/python scripts/ibm_run.py --check                      # list backends, queue, no job
       .venv/bin/python scripts/ibm_run.py --local [--hours 24]         # same pipeline on the local statevector sampler
       .venv/bin/python scripts/ibm_run.py --backend ibm_fez --day 2026-10-14 [--hours 24 --shots 1024 --p 3]
  -> results/ibm_p2_<backend>.csv
"""
import argparse, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from qiskit import QuantumCircuit
from qest import qubo
from qest.paths import RESULTS, ROOT


def env():
    for line in (ROOT / '.env').read_text().splitlines() if (ROOT / '.env').exists() else []:
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get('IBM_QUANTUM_TOKEN'), os.environ.get('IBM_QUANTUM_CRN')


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok, crn = env()
    if not tok:
        sys.exit('IBM_QUANTUM_TOKEN is not set: put the IBM Cloud API key in .env (IBM_QUANTUM_TOKEN=...). The CRN alone does not authenticate.')
    return QiskitRuntimeService(channel='ibm_cloud', token=tok, instance=crn)


def lr_qaoa(h, J, p, dg=.6, db=.6):
    """Linear-ramp QAOA for the QUBO E(x) = h.x + x.J.x (J upper triangular), x = (1 - z) / 2; coefficients normalised by the largest."""
    n = len(h); Js = J + J.T; hz = -(h / 2 + Js.sum(1) / 4); Jz = np.triu(J, 1) / 4; sc = max(np.abs(hz).max(), np.abs(Jz).max(), 1e-9); hz, Jz = hz / sc, Jz / sc
    qc = QuantumCircuit(n); qc.h(range(n))
    for k in range(p):
        g, b = dg * (k + 1) / p, db * (1 - k / p)
        for i in range(n):
            if abs(hz[i]) > 1e-9: qc.rz(2 * g * hz[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Jz[i, j]) > 1e-9: qc.rzz(2 * g * Jz[i, j], i, j)
        qc.rx(-2 * b, range(n))  # exp(+i b X): |+> is the ground state of -X, so the ramp ends in the low-energy states
    qc.measure_all(); return qc


def best_plan(X, V, lin, quad, off, H, level):
    """Lowest-energy feasible sample of X (shots, n), or None."""
    best = None
    for x in np.unique(X, axis=0):
        if qubo.feasible(x, V, lin, quad, off, H, level):
            e = qubo.energy(x, V, lin, quad, off)
            if best is None or e < best[0]: best = (e, x)
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--local', action='store_true'); ap.add_argument('--backend')
    ap.add_argument('--day'); ap.add_argument('--hours', type=int, default=24); ap.add_argument('--shots', type=int, default=1024); ap.add_argument('--p', type=int, default=3); a = ap.parse_args()
    if a.check:
        svc = service()
        for b in svc.backends(operational=True, simulator=False):
            print(b.name, b.num_qubits, 'qubits, pending jobs', b.status().pending_jobs)
        sys.exit()
    inst = qubo.instances('test', 'P2', 'retro', ndays=0, hours=24)
    if a.day: inst = [i for i in inst if str(i[0])[:10] == a.day]
    inst = inst[:a.hours]; built = [qubo.build(H, 'P2') for _, _, H, _ in inst]; circs = []
    for V, lin, quad, off, lam in built:
        h, J = qubo.arrays(V, lin, quad); circs.append(lr_qaoa(np.asarray(h, float), np.asarray(J, float), a.p))
    t0 = time.time()
    if a.local:
        from qiskit.primitives import StatevectorSampler
        res = StatevectorSampler(seed=0).run(circs, shots=a.shots).result(); name = 'local'; depth2 = [c.count_ops().get('rzz', 0) for c in circs]
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
        svc = service(); be = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=15); name = be.name
        isa = generate_preset_pass_manager(optimization_level=3, backend=be).run(circs); depth2 = [sum(v for k, v in c.count_ops().items() if k in ('cz', 'ecr', 'cx', 'rzz')) for c in isa]
        job = SamplerV2(mode=be).run(isa, shots=a.shots); print('job', job.job_id(), 'on', name, flush=True); res = job.result()
    rng = np.random.default_rng(0); rows = []
    for k, ((day, hour, H, HR), (V, lin, quad, off, lam)) in enumerate(zip(inst, built)):
        n = len(V); bits = res[k].data.meas.get_bitstrings(); X = np.array([[int(c) for c in s[::-1]] for s in bits])  # qiskit prints qubit 0 last
        eopt, xopt = qubo.exact(V, lin, quad, off, 'P2'); q = best_plan(X, V, lin, quad, off, H, 'P2')
        R = (rng.random((a.shots, n)) < .2).astype(int); r = best_plan(R, V, lin, quad, off, H, 'P2')
        val = lambda b: qubo.realised(b[1], V, HR, 'P2') if b else 0.0
        rows.append(dict(day=str(day)[:10], hour=hour, backend=name, shots=a.shots, p=a.p, two_qubit_gates=depth2[k], exact_eur=qubo.realised(xopt, V, HR, 'P2'), qpu_eur=val(q), random_eur=val(r),
                         qpu_opt=bool(q and np.isclose(q[0], eopt, rtol=1e-9, atol=1e-6)), random_opt=bool(r and np.isclose(r[0], eopt, rtol=1e-9, atol=1e-6)),
                         feasible_share=float(np.mean([qubo.valid(x, V, 'P2') for x in X])), p_opt=float(np.mean((X == np.asarray(xopt)).all(1)))))
    T = pd.DataFrame(rows); T.to_csv(RESULTS / f'ibm_p2_{name}.csv', index=False)
    print(f'{name}: {len(T)} hours, {a.shots} shots, p={a.p}, {time.time() - t0:.0f} s; 2-qubit gates per circuit {T.two_qubit_gates.mean():.0f}')
    print(f'  optimum returned: QPU {T.qpu_opt.mean():.2f} | random control {T.random_opt.mean():.2f};  EUR as % of exact: QPU {100 * T.qpu_eur.sum() / T.exact_eur.sum():.1f} | random {100 * T.random_eur.sum() / T.exact_eur.sum():.1f}')
    print(f'  valid (one-hot) share of shots {T.feasible_share.mean():.3f}; P(optimum) per shot {T.p_opt.mean():.4f} (uniform: {1 / 2 ** 15:.6f})')
