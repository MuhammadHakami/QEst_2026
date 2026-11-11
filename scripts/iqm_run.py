"""Run exported OpenQASM circuits (scripts/ibm_run.py --export) on an IQM Resonance device; counts go back as JSON for
scripts/ibm_run.py --score. Runs in .venv-iqm (the IQM client pins an older qiskit). Token: IQM_MUHAMMAD_TOKEN in .env.
usage: .venv-iqm/bin/python scripts/iqm_run.py circuits.json out.json [--device garnet] [--shots 1024]
"""
import json, os, sys
for line in open('.env'):
    if '=' in line and not line.startswith('#'):
        k, v = line.strip().split('=', 1); os.environ.setdefault(k, v)
os.environ['IQM_TOKEN'] = os.environ.get('IQM_TOKEN') or os.environ['IQM_MUHAMMAD_TOKEN']
from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
src, out = sys.argv[1], sys.argv[2]; dev = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'garnet'; shots = int(sys.argv[sys.argv.index('--shots') + 1]) if '--shots' in sys.argv else 1024
be = IQMProvider('https://resonance.meetiqm.com', quantum_computer=dev).get_backend()
circs = [QuantumCircuit.from_qasm_str(q) for q in json.load(open(src))]; isa = transpile(circs, backend=be, optimization_level=3, seed_transpiler=0)
g2 = [sum(v for k, v in c.count_ops().items() if k in ('cz', 'move')) for c in isa]; print(dev, len(isa), 'circuits, 2-qubit gates', sum(g2) / len(g2), flush=True)
bits = []
for i in range(0, len(isa), 8):  # small batches
    job = be.run(isa[i:i + 8], shots=shots); res = job.result(); print('job', job.job_id(), flush=True)
    for k in range(len(isa[i:i + 8])):
        c = res.get_counts(k); bits.append([b.replace(' ', '') for b, n in c.items() for _ in range(n)])
json.dump(dict(name=f'iqm_{dev}', two_qubit_gates=g2, bits=bits), open(out, 'w')); print('saved', out)
