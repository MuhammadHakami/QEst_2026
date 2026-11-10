# 2026 quantum-for-energy competitions and conferences: code, papers, what we took

Source list: our doc "Quantum for Energy: 2026 State of the Art" (tables "Challenges and competitions" and "Big
conferences and journals"). Everything here is re-fetchable: `.venv/bin/python research/fetch.py` (clones into
`third_party/`, PDFs into `papers/`, `--search` queries GitHub for unlinked entries). Checked 7 Nov 2026.

## Challenges and competitions

| Competition (2026) | Result | Method | Local code / paper | What we took |
|---|---|---|---|---|
| DOE Quantum Grid Planning (Global Industry Challenge, Connected DMV) | winner Qtyche, runner-up BosonQ Psi | Qtyche: QAOA + Benders decomposition, storage siting under contingencies (no public code) | `third_party/HQCMCBD-API` (open hybrid quantum-classical multi-cut Benders, D-Wave / OpenJij master) | `qest/qbenders.py`: annealing master (OpenJij SQA) + HiGHS LP subproblem on our day MILP |
| same challenge, other entries | eQoSystem; team kazi | PCE + transformer circuit synthesis, N-1 storage siting; QAOA siting on IBM ibm_kingston, matches brute force | `third_party/qGridX` (+ `eQoSystem__Phase3_Version1.pdf`), `third_party/Q-Site` (+ `kazi_Challenge_Phase3.pdf`) | PCE already in `qest/qsota.py`; Q-Site confirms QAOA = brute force at this size |
| QCi microgrid challenge (GIC) | winners not published | dispatch with cubic costs on IBM, D-Wave, QCi, IonQ via qBraid | GIC forecasting entry `third_party/resque-qrc` (QRC, gate model + QuEra Aquila analog) | `qest/qrc.py` (quantum reservoir correction of Chronos-2) |
| E.ON grid expansion (Global Quantum + AI Challenge) | phase I open | build-or-no-build with network physics | none public | not applicable to trading |
| Tennessee Quantum Hackathon, TVA tracks | event 13-15 Nov | hour-by-hour QAOA unit commitment with slack qubits, IonQ Forte, <= few dozen 2-qubit gates | `third_party/tn-quantum-hackathon` | the gate budget: our NAS reports CZ counts, PostNAS cuts them |
| Q-Volution, Rigetti grid track | prizes paid, names unpublished; eQoSystem entry | PCE weighted Max-Cut on Ankaa-3 | `third_party/Rigetti_Quantum_Grid_Optimization_eQoSystem` | PCE (P3: 30 bits on 6 qubits) |
| Interstate Quantum Hackathon, energy track | XY + Dicke QAOA EV charging: 100 % valid, optimum 6.9 % on ibm_fez, 8.4 % on ibm_kingston | constraint-preserving XY mixer | `third_party/quant5-fisk-IQ-hackathon` | the XY one-hot mixer our NAS picked independently (`qest/qnas.py`) |
| Quantum grid intelligence (entry) | MA-QAOA p=1, r 0.987 on Quantinuum Nexus | multi-angle warm-started QAOA | `third_party/quantum-grid-intelligence` | warm start (NAS: it does most of the work, we say so) |
| XPRIZE Quantum Applications | finals 2027 | materials, not grids | none | none |

## Conferences and journals

| Venue | Result | Local code / paper | What we took |
|---|---|---|---|
| IEEE QCE26 (Toronto): PNNL, quantum Hamiltonian descent | 5-unit dispatch cost -3.97 % vs GAMS, simulated, ~2 min per case | `third_party/QHDOPT`, `papers/2303.01471.pdf`, `papers/2508.02969.pdf` | not used: our dispatch is a convex LP, where QHD has nothing to gain |
| IEEE QCE26: Los Alamos, HHL for DC-OPF | condition number grows ~ buses^1.81 on PGLib up to 78,000 buses: no speedup | `papers/2607.19263.pdf` (same verdict, Khanpour & Talkington) | quantum linear solvers left out |
| IEEE QCE26: unit commitment | D-Wave picks the units, dispatch stays classical | (method family) `third_party/HQCMCBD-API` | the binaries-to-quantum split of `qest/qday.py` and `qest/qbenders.py` |
| IEEE PES GM 2026 | quantum task-force session | `papers/PES_GM_2026_programme.pdf` | none |
| IEEE TII 2026, quantum-ready stochastic UC | one pipeline: classical, quantum-inspired, quantum | `papers/2611.05030.pdf` | scenario framing for the whole-day problem (open item) |
| Benchmarking hybrid algorithms (AC-OPF UC) | qubit-efficient hybrid does not beat random sampling | `third_party/PowerGridQuantumOpt` (812 MB), `papers/2607.15543.pdf` | every quantum run here has a random / classical control |
| REGRID-QAOA | resource-efficient hybrid QAOA for islanding | `papers/2606.15083.pdf` | none |
| Smart-grid review; WEF utilities report | operational optimisation is the nearest value | `papers/2609.18847.pdf`, `papers/WEF_Quantum_for_Energy_and_Utilities_2026.pdf` | framing |

Method papers behind our own modules: BF-DCQO (`2409.04477`, `2505.08663`), QMCMC (`2602.06171`), QRC (`2606.12806`),
VQC RL (`2601.18811`), PCE unit commitment (`2605.17145`), copula QAOA (`2603.00260`); ADMM-DQAOA code in
`third_party/DQAOA`, FALQON unit commitment in `third_party/Unit-Commitment-Quantum-Opt`.
