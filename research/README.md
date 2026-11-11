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

## Added 10 Nov 2026 (second research pass: how 2026 entries made their point, and the SOTA we build on)

| Source | What it shows | What we take |
|---|---|---|
| QEst Hack 2025 (Metrosert, Dec 2025): winner Enefit, EV charging on IBM / IQM clouds; prize = a 3-month pilot with Université de Sherbrooke | the jury rewarded a real company problem, a working hybrid pipeline and a statement of savings, not a speedup claim; the organisers' own write-up quotes "real cost savings" without a published benchmark | our case is built the same way but with the numbers shown: walk-forward revenue, both settlements, paired intervals, every quantum run with a random / classical control |
| Global Industry Challenge 2026 winner Qtyche (QAOA + Benders, storage siting) | "matches the exact classical optimum on IEEE 14-bus": the point is correctness at small scale plus a decomposition that scales | our partitioned solver and `qbenders` follow the same decomposition logic; we also report where it loses |
| IonQ, arXiv:2611.08129 (Provazza et al., shadow-trained NQS for AFQMC) | the QPU is sampled once, a small autoregressive model is distilled from the samples and used in production: 13,200 GPU h -> 4 GPU h per structure | `qest.qdistill`: the season runs on the distilled model, zero QPU calls at decision time (research/distilled_quantum.md) |
| IBM / Hartree, arXiv:2602.06171 (Marshall, Egger et al., Feb-Mar 2026): quantum-enhanced MCMC + warm start + parallel tempering for combinatorial optimisation, IBM Heron r3, median 114-151 iterations to the optimum on MIS; "early empirical evidence" of a scaling advantage | the quantum proposal is a sampler, the value is in mixing, and the honest claim is scaling, not a wall-clock win | `qest.qscen_q`: the same proposal measured against the exact chain (TV, acceptance, autocorrelation), now extended with tempering and used as the day's joint-scenario sampler (section below) |
| Orfi & Sels, arXiv:2403.03087: an upper bound on the Markov gap rules out a speedup for any unital quantum proposal | the sampler's advantage, if any, is polynomial and target-dependent | we quote mixing per step and per second, never "exponential" |
| TISI, arXiv:2607.24722 (PCE on power-demand portfolios, m = 18 .. 10,296 variables, cost gap ~1e-4, a trapped-ion run) and arXiv:2607.11637 (247 method-instance runs on IBM Heron: deeply compiled QAOA-style methods "predominantly random-baseline-limited"; PCE and QRAO judged on objective quality, eligibility, circuit size, robustness) | qubit-efficient encodings are the commercially credible 2026 route (one qubit per ~5 bits); shallow circuits survive, deep QAOA does not | PCE already in `qest.qsota` / `qdecide` (4 qubits per 12-bit window); our scoring matches 2607.11637's protocol (random control, hit rate, circuits) |
| Sample-based training of quantum generative models (arXiv:2511.11802); Quantum Fourier generative models trainable at scale (arXiv:2606.28483) | generative models trained from samples scale when the trainable part is classical | supports the distil-once design: the QPU is a data source |
| WEF "Quantum for Energy and Utilities" 2026; S&P "Energy, compute and the quantum era" 2026; boardroom guide (Fifth Row) | as of 2026 no energy pilot reports a quantified commercial outcome; every project is at demonstrator stage | our deck says so and prices the QPU per decision across four vendors |

## Added 11 Nov 2026 (forecasting SOTA pass, requested by the user)

| Source | What it shows | What we take |
|---|---|---|
| Mistral Timestral (Oct 2026, arXiv:2610.37585; 5M-209M, #1 zero-shot on GIFT-Eval / TIME / fev-bench at launch; code Apache 2.0, weights Modified MIT) | patch transformer with past-only and known-future covariates, 21 quantiles, 32,768-step context, CPU-fast | `qest.fc_tsfm`: drop-in beside Chronos-2 (same groups, covariates, layout); Q3 balancing targets -3 to -7 % MAE, Q4 +3 %; `QEST_FC=timestral` runs the whole plan family on it; `tsens` averages the two models |
| t0 (The Forecasting Company, arXiv:2609.24559; 102M / 256M, covariates, Apache 2.0, `tfc-t0`) | third on GIFT-Eval, context-conditioned | installed (`t0.T0Forecaster`), not yet run: Timestral first |
| Kim, Baglio et al., arXiv:2609.05408 (Sep 2026): quantum fine-tuning head on frozen Chronos embeddings, PSML-5 grid events; 12-qubit core + "wings"; beats a 1.2-3 M MLP by 1.7-2.0 pp with 3,240 parameters | the bottleneck is data intake, not expressiveness: more qubits help only with new inputs | `qest.qhead`: the head on our activation drivers; beats the MLP control on every target, ties climatology / LightGBM on mFRR calibration, trails LightGBM on ranking and aFRR shares (quantum_gain_2026.md s6a) |
| Chakraborty & Heintz, arXiv:2607.16358 (QRC-F / VQF-F vs DLinear / PatchTST / iTransformer); Vercellino et al., arXiv:2606.04686 (4-qubit digital QRC with measure-and-reset on IQM Spark, ATM cash demand: loses to Prophet on MAE, competitive on DTW) | compact quantum forecasters match classical baselines on benchmarks, do not beat strong ones on real series | consistent with our `qrc` result (ties Chronos-2 blended) |
| Quantum generative time series: QGAN / QCBM on S&P 500 (IOP MLST 2026, arXiv:2507.22035), MPS generators (2402.17148), QDiffusion-TS on IQM (2606.27561) | sampled paths with stylised facts on windows of 5-20 points; no energy application | `qest.qgen`: conditional Born machine over 16-qubit activation windows as the scenario generator (s8) |
| QRC long-horizon behaviour: 2509.12071 (chaotic maps), 2604.03469 (memory saturates), 2608.00139 (neural forecasting) | closed-loop quantum reservoirs accumulate error with horizon like classical ones | the "no collapse to the mean" premise is ours to test, not established (s7) |
| IonQ 2026 resource papers (Walking Cat qLDPC 2604.19481; single-CPU real-time decoder; ECDLP blueprint) and Harvard / QuEra AI-designed circuits (2610.22367) | fault-tolerant overhead falling 10x; LLM-found circuits beat hand designs on 7 of 8 tasks | context for the cost slide; nothing to run on NISQ-scale bidding |

Method papers behind our own modules: BF-DCQO (`2409.04477`, `2505.08663`), QMCMC (`2602.06171`), QRC (`2606.12806`),
VQC RL (`2601.18811`), PCE unit commitment (`2605.17145`), copula QAOA (`2603.00260`); ADMM-DQAOA code in
`third_party/DQAOA`, FALQON unit commitment in `third_party/Unit-Commitment-Quantum-Opt`.
