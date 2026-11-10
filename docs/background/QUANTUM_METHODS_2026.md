# Quantum and quantum-inspired optimisation for the portfolio (research, 5 Nov 2026)

User (5 Nov): research the latest quantum algorithms, annealing and others, especially energy-related, and add them to the
notebook's benchmarks (notebooks/KratTrade_Makkah_1450.ipynb, code in notebooks/makkah_qopt.py).

## What was added (15 methods, four families)
| Family | Method | Source | Why it fits the plant's problem | Code |
|---|---|---|---|---|
| Classical | Tabu search | dwave-samplers TabuSampler | the strong local-search baseline every annealing benchmark uses | `tabu` |
| Quantum-inspired | Simulated quantum annealing (path-integral Monte Carlo) | D-Wave dwave-samplers 1.8 `PathIntegralAnnealingSampler` | the physics a D-Wave QPU anneals, on a CPU (no Leap token here); unit commitment on D-Wave: Sci. Rep. 2025 (s41598-025-96220-2), Teng 2025 (arXiv 2502.15917) | `sqa` |
| Quantum-inspired | Simulated bifurcation | Goto et al., Sci. Adv. 5, eaav2372 (2019); Sci. Adv. 7, eabe7953 (2021); Toshiba SQBM+ (3rd generation, Apr 2026); open code: simulated-bifurcation 2.0 | Ising machine that scales to 100k variables; quadratic only, so the 19-qubit hour goes through dimod.make_quadratic | `sbm` |
| Quantum | LR-QAOA (linear ramp, fixed angles) | Montañez-Barrera, Michielsen, Bernal Neira, npj Quantum Inf. 11, 131 (2025); QPU benchmark arXiv 2502.06471 | no classical loop: one circuit a day; the ramp Δ = 1.0 chosen once on 4 Mar (outside the study days) | `lr_qaoa` |
| Hybrid | XY-mixer QAOA | Hadfield et al. 2019; Wang, Rubin, Dominy, Rieffel, PRA 101, 012320 (2020) | one battery plan and one commitment are one-hot: the XY mixer keeps every sample valid (as unit commitment's one state per unit and hour) | `xy_qaoa` |
| Hybrid | Warm-start QAOA | Egger, Mareček, Woerner, Quantum 5, 479 (2021) | starts from the continuous relaxation (cited by the power-grid QAOA benchmarks, arXiv 2607.15543) | `ws_qaoa` |
| Hybrid | Grover adaptive search | Gilliam, Woerner, Gonciulea, Quantum 5, 428 (2021); EV charging: Federer et al. 2023 (GI) | quadratic speed-up in oracle calls in principle; the oracle's cost decides it | `gas` |
| Hybrid | Quantum-enhanced MCMC | Layden et al., Nature 619, 282 (2023) | quantum proposals, classical Metropolis: robust to noise in the paper; cooled here for optimisation | `qemcmc` |

Already in: exact enumeration, simulated annealing, QAOA p = 1, 2 (+ COBYLA), CVaR-QAOA (Barkoutsos et al. 2020), BF-DCQO
(Cadavid et al., PRR 2025; Kipu's Iskay optimiser on IBM Heron, arXiv 2409.04477), DCQO one round.

## Checked and left out
- D-Wave Advantage2 QPU, Kipu Iskay (IBM Qiskit Functions), Toshiba SQBM+: paid cloud services, no account here. Their
  algorithms stand in: SQA for the annealer, BF-DCQO for Iskay, open simulated bifurcation for SQBM+.
- ADAPT-QAOA, multi-angle QAOA, VQE: more classical tuning on a problem exact enumeration solves in milliseconds; no energy
  result that beats the methods above.
- Constrained counterdiabatic QAOA for portfolios (arXiv 2605.06858): finance, not yet with open code.

## Honest limits
- The 6-qubit day and the 19-qubit hour are enumerable: exact wins on time; the benchmark measures how often each method
  samples the optimum and at what shot cost, not a speed-up.
- Annealing reviews (arXiv 2602.03101) record unit commitment as "classical wins"; LR-QAOA's scaling claim is on weighted
  MaxCut, not on energy scheduling.
- QeMCMC and GAS are simulated exactly on the state vector; GAS's oracle cost is a generic diagonal gate (Gilliam's
  quantum dictionary is cheaper); QeMCMC runs at 6 qubits only (a 2^19 x 2^19 propagator is not practical).
- dimod.make_quadratic: penalty = max|c| (2 Σ|c| left SQA 23 % short on the 19-qubit hour, max|c| a few %); Tabu stays 23 % short at every penalty tried: 4-body terms cost the quadratic solvers. GAS stops after 2n misses in a row (10 left it 28 % short at 19 qubits).
