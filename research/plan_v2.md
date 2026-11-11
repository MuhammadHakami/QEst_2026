# Plan of action v2 (11 Nov 2026): evidence, then what to do with the IQM / IBM keys

Supersedes the plan in research/quantum_gain_2026.md s4. Every line cites a measurement in this folder or a 2026 paper.
Aim stated by the user: revenue as close to hindsight as possible over the year, with quantum earning its place.

## 1. What the evidence says (walk-forward, 291 days, battery + hybrid)

| Finding | Number | Source |
|---|---|---|
| Baseline, Chronos-2 + MILP | EUR 37.48 M (30.28 M at the 25 MW cap) | results/days_*_actfc_wf*.parquet |
| Hindsight | EUR 105.2 M; 82 % of the gap is knowing the TSO's activations | research/gap_to_hindsight.md |
| Activation model + ranking gate | +3.2 % (+0.3 % capped) | research/activation_model.md s6 |
| Two-stage scenario plan, faithful recourse | +4.0 to +6.7 % more; capped +5.4 % (every month positive) | research/quantum_gain_2026.md s1 |
| Price forecaster swapped to Timestral-209M | baseline plan +6.0 % Q3, -6.2 % Q4; plan of record -0.7 % / +1.8 % (noise) | s6c below |
| Classical medians keep 30-57 % of the realised spread | all three forecasters | results/forecast_collapse.csv |
| Quantum deciders on the best inputs | +-0.7 % of the MILP; QUBO optimum hit 94-98 % | s3 |
| Quantum-proposal MCMC scenarios | plan +7.3 % over baseline; classical samplers +7.4 to +10.1 % | s1d |
| Quantum fine-tuning head (12 qubits) | beats the MLP control on all targets; ties / trails LightGBM | s6a |
| Conditional Born machine, 16 qubits, 4 layers | held-out log-lik -0.69 to -0.82 per MTU vs chain -0.47 to -0.55; UP runs 2.5 vs 6.9 MTUs | results/qgen_bench_*.csv |
| P3 puzzle (30 qubits) | 99.92 % of optimum on 15 / 10 qubits | s5 |

Lean 4 (proofs/Qproofs/Revenue.lean): hindsight is an information bound; the scenario plan is never worse than a point
plan in expectation; a sampler with L1 error d costs at most 2 B d; hence a quantum sampler can add at most
2 B ||p - q_classical||_1 over a classical one. On a target the classical sampler draws exactly, that bound is zero.

## 2. What the 2026 literature adds (second pass: NeurIPS 2026, QCE26, arXiv)

- NeurIPS 2026 position paper "Stop quantum machine learning; do AI-for-quantum instead": QML on classical data needs a
  stricter evidentiary standard. We adopt it: every quantum number here has a matched classical twin and an interval.
- "When forecast accuracy fails: rank correlation and decision quality in multi-market battery storage" (arXiv:2604.12082):
  decision quality follows rank correlation, not MAE. Matches our Timestral result (worse MAE on Q4, more euros) and our
  ranking gate.
- Diff2SP (arXiv:2606.05649) and decision-focused scenario generation for dispatch (arXiv:2607.05830): train the scenario
  generator through the stochastic program, on decision loss. Our finding "4 scenarios beat 16, the worst sampler earns
  most" is the symptom these papers treat.
- Stochastic Mean-CVaR BESS bidding with mFRR activation as tail risk (arXiv:2609.18349, IECON26): the model family of
  qest.stoch; they find honest recourse makes storage withdraw from reserves, as our guard model did.
- IQP / spectral Born machines trained classically and sampled on hardware (Bowles et al.; arXiv:2603.14576, 2607.06675,
  2609.28802): sampling hardness is proven for the circuit family, advantage for *trained* models is still conjectural.
  This is the only 2026 quantum-generative design that scales past statevector training.
- QCE26: IonQ best papers on distributed optimisation and quantum fine-tuning; PNNL QHD dispatch -3.97 % vs GAMS in
  classical simulation. No energy-market hardware result with a classical control.

## 3. The plan

Ordered by expected euros per day of work; each step has its acceptance test.

1. **Decision-focused scenario selection (classical, highest expected value).** Draw 64 chain scenarios, keep the 4-8
   whose plan earns most on the *previous* 30 days' realised outcomes (walk-forward, D-2 lag). Test: beats `milp_scenf4`
   over the year in both settlements, paired interval excluding zero. Rationale: the plan cannot tell samplers apart by
   fidelity, so select by decision loss (Diff2SP).
2. **Extend the scenario object beyond the chain** (aFRR share level, not only the mFRR bits; both borders; the Finnish
   price). This is where q_classical stops being exact and the Lean bound stops being zero. Test: held-out joint
   log-likelihood above the chain's; plan euros above step 1.
3. **Quantum sampler on that joint target, on hardware.** IQP-type Born machine, trained classically (MMD on the
   characteristic function), sampled on the QPU: 20 qubits on IQM Garnet (10 MTUs), 100+ on IBM Heron (a half day).
   Budget: 300 windows x 1,000 shots, about $1k on IQM, $3-10k on IBM. Controls: tempered MCMC and the chain at equal
   wall time. Test: (a) samples pass the held-out likelihood proxy (MMD) within the simulator's value, (b) plan euros
   on QPU scenarios within the interval of the simulated ones, (c) distil once (qdistill) and re-run the season at zero
   QPU calls. Claim allowed if passed: "a QPU supplies the plan's scenarios at parity"; a euro advantage only if (b)
   beats the classical samplers with an interval excluding zero.
4. **Run the P3 blocks and the day QUBO on IQM / IBM** (15 and 10 qubits, LR-QAOA p = 1-3): the organisers' quantum
   track on real hardware, with the random control. Budget: under $500. Test: optimum hit under noise against 0.99 ideal.
5. **Quantum head as a committee member** for the mFRR probabilities (seed ensemble of 5, wings with the border flows as
   new inputs). Test: ranking gate admits it more often than LightGBM alone and the gated plan earns more over the year.
6. **Not pursued:** quantum reservoirs as stand-alone forecasters (tie at best, literature agrees), deeper QAOA (random-
   limited on Heron, arXiv:2607.11637), the Born machine in its present form (loses to the chain by 0.25 nats per MTU).

## 4. On "near-perfect hindsight"

Theorem 1 is the constraint: at gate A the day's activations are not known, and no computation recovers information that is
not in the inputs. The route toward hindsight is therefore (i) more information (the TSO's activated volumes, intraday
re-planning at gate C, which the organisers' rules allow only through bid prices) and (ii) using the uncertainty that
remains optimally, which is what the scenario plan does. Modelling noise does (ii), not (i). The realistic target for
2026 is the reachable optimum with perfect prices and forecast activations, EUR 46.5 M; the plan of record is at 41.3 M.
