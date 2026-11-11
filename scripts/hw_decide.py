"""The day's charge / discharge QUBO (24-25 qubits, one per hour) on real QPUs, for the revenue loop.
  1. export:  QEST_WALK=1 .venv/bin/python -m qest.qdecide --split test --fc actrank --solvers hwexport --tag _hwx  -> results/hw_problems_test_actrank.json
  2. run:     .venv/bin/python scripts/hw_decide.py ibm results/hw_problems_test_actrank.json results/hw_samples_ibm.json [--backend ibm_fez --p 1 --shots 1024]
              .venv/bin/python scripts/hw_decide.py qasm ... out.json   (OpenQASM for scripts/iqm_run.py);  ... bits iqm_out.json results/hw_samples_iqm.json
  3. score:   QEST_HW_SAMPLES=results/hw_samples_ibm.json QEST_WALK=1 .venv/bin/python -m qest.qdecide --split test --fc actrank --solvers samples,hw,hw+ --tag _hw_ibm
LR-QAOA, fixed linear ramp (as qdecide.solve_qaoa: angles on an O(1) energy scale), no training on the device.
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from qiskit import QuantumCircuit

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def circuit(h, J, p, delta=.75):
    h, J = np.asarray(h, float), np.asarray(J, float); n = len(h); E = [(i, j, J[i, j] + J[j, i]) for i in range(n) for j in range(i + 1, n) if abs(J[i, j] + J[j, i]) > 1e-9]
    s = np.sqrt(np.mean(h ** 2) + np.mean([e[2] ** 2 for e in E] or [0])); ramp = (np.arange(p) + .5) / p; qc = QuantumCircuit(n); qc.h(range(n))
    for k in range(p):
        g, b = delta * ramp[k], delta * (1 - ramp[k])
        for i in range(n): qc.rz(2 * g * h[i] / s, i)
        for i, j, w in E: qc.rzz(2 * g * w / s, i, j)
        qc.rx(-2 * b, range(n))
    qc.measure_all(); return qc


if __name__ == '__main__':
    mode, src, out = sys.argv[1], sys.argv[2], sys.argv[3]; p = arg('--p', 1); shots = arg('--shots', 1024)
    if mode == 'bits':  # counts from scripts/iqm_run.py -> {key: [[bits]]}; keys from the problems file given with --problems
        keys = list(json.load(open(arg('--problems', '')))); J = json.load(open(src))
        json.dump({k: [[int(c) for c in s[::-1]] for s in b] for k, b in zip(keys, J['bits'])}, open(out, 'w')); print('saved', out, 'mean 2q gates', np.mean(J['two_qubit_gates'])); sys.exit()
    P = json.load(open(src)); keys = list(P); circs = [circuit(P[k]['h'], P[k]['J'], p) for k in keys]
    if mode == 'qasm':
        from qiskit import qasm2
        json.dump([qasm2.dumps(c) for c in circs], open(out, 'w')); print('exported', len(circs)); sys.exit()
    from scripts.ibm_run import service
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    svc = service(); be = svc.backend(arg('--backend', 'ibm_fez')); isa = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=0).run(circs)
    g2 = [sum(v for k, v in c.count_ops().items() if k in ('cz', 'ecr', 'cx', 'rzz')) for c in isa]; print(be.name, len(isa), 'circuits,', circs[0].num_qubits, 'qubits, 2-qubit gates', np.mean(g2), flush=True)
    job = SamplerV2(mode=be).run(isa, shots=shots); print('job', job.job_id(), flush=True); res = job.result()
    json.dump({k: [[int(c) for c in s[::-1]] for s in res[i].data.meas.get_bitstrings()] for i, k in enumerate(keys)}, open(out, 'w')); print('saved', out)
