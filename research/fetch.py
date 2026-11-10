"""Clone the code and download the papers behind the 2026 quantum-for-energy competitions and conference results
(the 'Challenges and competitions' and 'Big conferences and journals' tables of our sources doc).
usage: .venv/bin/python research/fetch.py [--search]   -> research/third_party/<repo>, research/papers/<id>.pdf
--search queries the GitHub search API for entries the doc does not link (unauthenticated: 10 queries a minute).
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
UA = {'User-Agent': 'qest-research-fetch'}

REPOS = {  # repo: why it is here
    'qBraid/tn-quantum-hackathon': 'Tennessee Quantum Hackathon 2026, TVA plant scheduling (unit commitment) + sensor placement, IonQ Forte',
    'AshrafBoussahi/Rigetti_Quantum_Grid_Optimization_eQoSystem': 'Q-Volution 2026 Rigetti grid track, eQoSystem: PCE weighted Max-Cut on Ankaa-3',
    'SakxamShrestha/quant5-fisk-IQ-hackathon': 'Interstate Quantum Hackathon 2026 energy track: XY-mixer QAOA EV depot charging, 6.9 % optimum on IBM',
    'QuantumHackathon/quantum-grid-intelligence': 'hackathon entry 2026: multi-angle warm-start QAOA, grid fault-zone partitioning',
    'GaidaiIgor/PowerGridQuantumOpt': 'arXiv:2607.15543 benchmarking hybrid algorithms for AC-OPF unit commitment',
    'jiaqileng/QHDOPT': 'quantum Hamiltonian descent (QHDOPT), the method behind PNNL QCE26 dispatch result',
    'LSU-RAISE-LAB/DQAOA': 'arXiv:2608.04159 distributed QAOA in ADMM for unit commitment',
    'Cairo-Henrique/Unit-Commitment-Quantum-Opt': 'FALQON unit commitment, 2026',
    'AshrafBoussahi/qGridX': 'eQoSystem entry, DOE Global Industry Challenge 2026: N-1-secure storage siting + microgrid formation',
    'edwingeorgeshaji/Q-Site': 'team kazi entry, DOE Global Industry Challenge 2026 (energy infrastructure): quantum storage siting',
    'Armstrong66/resque-qrc': 'Global Industry Challenge 2026 entry: dynamic forecasting with quantum reservoir computing',
    'USN-Quantum-Research/parallel-coordinated-qaoa': 'distributed QUBO-QAOA for capacity expansion planning (PyPSA)',
    'ajliouat/quantumgrid': 'QAOA and VQE for unit commitment on ENTSO-E data',
    'jhoule42/xQAOA_unit_commitment': 'QAOA variants for utility-scale unit commitment',
    'TravisCao/qugrid': 'MATPOWER grid case to quantum algorithm',
    'austinamissah/quantum-solar': 'QAOA home-battery scheduling under time-of-use tariffs',
    'djzts/HQCMCBD-API': 'hybrid quantum-classical multi-cut Benders decomposition (D-Wave + Gurobi): the method family of the DOE winner, Qtyche (QAOA + Benders)',
}
PAPERS = {  # arXiv id or URL: what it is
    '2611.05030': 'Quantum-ready stochastic unit commitment (IEEE TII 2026)',
    '2607.15543': 'Benchmarking hybrid algorithms for grid optimisation (AC-OPF UC)',
    '2606.15083': 'REGRID-QAOA: resource-efficient hybrid QAOA for power system islanding',
    '2609.18847': 'Quantum computing in smart grid operations: review',
    '2607.19263': 'The limits of quantum computers for power flow (QCE26 HHL / DC-OPF verdict line)',
    '2508.02969': 'QHD-based augmented Lagrangian (Fan, Li: the PNNL / Houston QHD team)',
    '2303.01471': 'Quantum Hamiltonian descent (Leng et al. 2023)',
    '2602.06171': 'Quantum-enhanced MCMC for combinatorial optimisation, 117 IBM qubits (2026)',
    '2505.08663': 'Runtime quantum advantage with digital quantum optimisation (BF-DCQO on IBM)',
    '2409.04477': 'BF-DCQO for higher-order binary optimisation',
    '2606.12806': 'Quantum reservoir computing for short-term load forecasting (2026)',
    '2601.18811': 'VQC reinforcement learning for dynamic portfolio optimisation (2026)',
    '2605.17145': 'Quantum unit commitment with PCE (2026)',
    '2603.00260': 'Copula-QAOA unit commitment on 150 IBM qubits (2026)',
    '2608.04159': 'Distributed QAOA inside three-block ADMM for unit commitment',
    '2610.10772': 'QAOA on a 1,354-bus grid: Frank-Wolfe + <=50-qubit partitions on IBM (E.ON / Qoro P2P)',
    '2609.36999': 'PCE + SQD unit commitment',
    '2609.40327': 'Quantum Q-networks for distribution-grid reconfiguration',
    '2607.24722': 'PCE for power-demand portfolios on trapped ions',
    '2606.22542': 'Quantum RBM for unit commitment',
    '2605.10255': 'Qubit vs qudit encodings for EV charging',
    '2603.07823': 'Hierarchical quantum annealing MPC, homes with hydrogen storage',
    '2601.01777': 'Quantum unit commitment survey',
    'WEF_Quantum_for_Energy_and_Utilities_2026': 'https://reports.weforum.org/docs/WEF_Quantum_for_Energy_and_Utilities_2026.pdf',
    'PES_GM_2026_programme': 'https://pes-gm.org/wp-content/uploads/2026/07/2026-IEEE-PES-GM-Print-Program.pdf',
}
SEARCH = ['qgridx', 'eQoSystem storage siting', 'Q-Volution Rigetti grid', 'Global Industry Challenge 2026 quantum',
          'quantum Benders decomposition', 'QAOA Benders storage', 'quantum annealing battery scheduling', 'quantum microgrid dispatch 2026']


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def clone(repo):
    dst = ROOT / 'third_party' / repo.split('/')[1]
    if dst.exists():
        return 'kept'
    r = subprocess.run(['git', 'clone', '--depth', '1', f'https://github.com/{repo}.git', str(dst)], capture_output=True, text=True)
    return 'cloned' if r.returncode == 0 else f'failed: {r.stderr.strip()[-120:]}'


def paper(key, what):
    dst = ROOT / 'papers' / f'{key}.pdf'
    if dst.exists():
        return 'kept'
    url = what if what.startswith('http') else f'https://arxiv.org/pdf/{key}'
    try:
        b = get(url)
    except Exception as e:  # noqa: BLE001 - report and go on
        return f'failed: {e}'
    if not b.startswith(b'%PDF'):
        return 'failed: not a PDF'
    dst.write_bytes(b); time.sleep(3)  # arXiv asks for a pause between requests
    return f'{len(b) // 1024} KB'


def search():
    out = {}
    for q in SEARCH:
        u = 'https://api.github.com/search/repositories?' + urllib.parse.urlencode(dict(q=q, sort='updated', per_page=8))
        try:
            items = json.loads(get(u))['items']
        except Exception as e:  # noqa: BLE001
            out[q] = [f'failed: {e}']; continue
        out[q] = [f"{i['full_name']} | {i['pushed_at'][:10]} | {i['stargazers_count']}* | {(i['description'] or '')[:90]}" for i in items]
        time.sleep(7)
    return out


def main():
    (ROOT / 'third_party').mkdir(parents=True, exist_ok=True); (ROOT / 'papers').mkdir(parents=True, exist_ok=True)
    if '--search' in sys.argv:
        print(json.dumps(search(), indent=1)); return
    for repo in REPOS:
        print('repo ', repo, clone(repo), flush=True)
    for k, w in PAPERS.items():
        print('paper', k, paper(k, w), flush=True)
    (ROOT / 'SOURCES.json').write_text(json.dumps(dict(repos=REPOS, papers=PAPERS), indent=1))


if __name__ == '__main__':
    main()
