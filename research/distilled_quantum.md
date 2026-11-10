# Shadow-trained NQS (IonQ, arXiv:2611.08129) -> a distilled QPU for the partitioned grid solver

Paper: Provazza, Knitter, Goings, Epifanovsky, Chen, Roetteler, Zhao, *Accelerating Quantum-Classical Auxiliary-Field
Quantum Monte Carlo with Shadow-Trained Neural Quantum States* (IonQ, 5 Nov 2026). Code: `qest/qdistill.py`, hooked into
`qest.qpart` (window sources `nqs-<variant>`, `nqaoa`, `uniform`) and `qest.qdecide` (`p_nqs`, `p_nqs-ideal`,
`p_nqs-twin`, `p_nqaoa`); `p_nqs@chronos` is replayed over 2026 in the twin. Updated 9 Nov 2026 (learned calibration,
committee, cost accounting); only the method was read, none of the chemistry.

## Mapping

| Paper | Ours |
|---|---|
| Quantum trial state guiding classical AFQMC walkers | LR-QAOA on each 12-qubit window, guiding the classical block-descent loop over windows (qest.qpart) |
| ~60k matchgate shadows, summed at every walker step | 1,000 QAOA shots per window, needed at every sweep of every day |
| Training inputs = determinants from a preliminary classical run; targets = shadow overlaps | training windows = the ones the classical loop visits on Q3 days; targets = the QAOA samples of each |
| Distil once into an autoregressive transformer NQS (1 causal block, emb 16, 2 heads, FF 64, ~13k params) | the same block, conditioned on the window's QUBO (3,953 params), trained once on 1,517 Q3 windows |
| One NQS per molecular structure | one model for every window, day and unit (amortised: windows are new every day) |
| Symmetry mask on the conditionals | not needed: MW / SoC / reserves stay in the LP, every bit string is valid; padding beyond the window is masked |
| Noise-aware: device state = a x ideal + uniform response; a LEARNED with the weights under a ridge toward 1 (Eq. 15), no weight decay on a | the same: measured = a p_model + (1 - a) 2^-n, a = sigmoid(alpha) learned, ridge lam (a - 1)^2, lam scanned (0, .3, 3, 30) |
| Committees per setting, the member with the lowest Rayleigh energy kept | 3 seeds per setting, lowest held-out normalised sample energy kept |
| VNA entropy annealing early in training | skipped: maximum likelihood on fixed samples does not collapse (their risk is energy minimisation) |
| Shadows used only for training; production no longer scales with shots | the loop samples the model: zero QPU calls at decision time |
| 13,200 GPU h -> 4 GPU h per structure | see Cost below: our classical cost is cents either way, the saving is QPU time |

## Results

Held-out Q3 windows (374, days from 12 Sep; normalised mean sample energy, 0 = window optimum, 1 = uniform):
ideal QAOA 0.051, noisy QAOA 0.534, classical Boltzmann twin 0.051. Models (best of 3 seeds): ideal 0.173, twin 0.174,
noisy noise-unaware 0.584, noise-aware 0.540 with learned a = 0.90 (lam = 0; lam .3 -> a .92, 0.548; lam 3 -> a .95,
0.558; lam 30 -> a .95, 0.559). 21 models trained in 1.0 min on one GPU.

Honest reading: the learned calibration closes about 8 % of the noisy-to-ideal model gap (0.584 -> 0.540 of 0.173). The
paper's denoising leverage comes from a symmetry sector the chemistry state must lie in; our bit strings have no such
sector, and gate noise in a 12-qubit QAOA is not a uniform response, so the scalar has little to grip. The earlier version
of this note reported "92 % recovered" with a fixed a = 0.05: that was a sharpening knob, not a calibration, and is
withdrawn.

Q4 day QUBOs (50 day-units, exact optimum known), partitioned loop, hit = optimum reached:

| Window source | Hit | QPU circuits / day-unit | Time |
|---|---:|---:|---:|
| Live LR-QAOA | 0.76 | 10.1 | 0.13 s |
| Live noisy LR-QAOA | 0.76 | 10.2 | 0.55 s |
| Distilled, ideal data | 0.82 | 0 | 0.03 s |
| Distilled, noisy data, noise-aware (a = 0.90) | 0.74 | 0 | 0.03 s |
| Distilled, noisy data, unaware | 0.76 | 0 | 0.03 s |
| Distilled, classical twin data | 0.80 | 0 | 0.03 s |
| Uniform random | 0.44 | 0 | 0.01 s |

Q4 euros (battery + hybrid, official settlement; Chronos-2 + MILP EUR 1,812,548): live partitioned QAOA and live noisy
QAOA EUR 1,800,367 (-0.67 %, identical: the top-8 exact-LP check absorbs the noise), twin-distilled 1,800,368,
noise-aware distilled 1,802,092 (-0.58 %), ideal-distilled 1,796,677 (-0.88 %). Whole 2026 season (291 days, both units):
noise-aware distilled EUR 37,409,048 vs MILP 37,602,146 (-0.51 %; Q1-Q2 -0.45 %, Q3 -0.70 %, Q4 -0.58 %), with zero QPU
calls after the one-off Q3 samples. It is the fourth strategy in the twin (`run.sh twin`).

"Guide, not replace": in the paper the quantum state improves an approximate classical solver. Our MILP is exact on its
forecast objective, so a plan with the sampled pattern fixed can never beat the free MILP on that objective. A rule that
takes the quantum pattern only when it wins on the planned objective picked the free MILP on all 50 Q4 day-units and
reproduced the baseline to the euro. Removed. The pattern can only differ in realised euros, which the replay measures.

Under the realistic settlement (activation volume cap 25 MW, planner volume-aware, `QEST_ACT_CAP=25`, Q4): MILP EUR 1,328,436,
distilled 1,315,734 (-0.96 %), live LR-QAOA 1,313,247 (-1.14 %). The distilled route holds its ground against the live QPU
route when the market is made faithful, with zero QPU calls (results/days_test_qdecide_chronos_cap25.parquet).

## Cost (`qdistill --cost`, results/qdistill_cost.csv)

A 2026 season = 582 day-units (291 traded days x 2 units). List prices, Nov 2026: Braket $0.30 a task plus per shot
(IonQ Forte $0.08, IQM Emerald $0.0016, IQM Garnet $0.00145, Rigetti Cepheus $0.000425); IBM Quantum pay-as-you-go
$96 a QPU minute, with 1-4 s of QPU time per 1,000-shot 12-qubit job assumed (not measured).

| Route | Circuits | IonQ Forte | IQM Garnet | Rigetti | IBM (1-4 s) |
|---|---:|---:|---:|---:|---:|
| Live partitioned, 1,000 shots a window, every day | 5,878 | $472 k | $10.3 k | $4.3 k | $9.4-37.6 k |
| Live partitioned, 200 shots a window (untested) | 5,878 | $96 k | $3.5 k | $2.3 k | $9.4-37.6 k |
| Live monolithic 24 qubits, 2,000 shots | 582 | $93 k | $1.9 k | $0.7 k | $0.9-3.7 k |
| Distilled: Q3 training windows, once | 1,891 | $152 k | $3.3 k | $1.4 k | $3.0-12.1 k |
| Distilled: the season on the model | 0 | $0 | $0 | $0 | $0 |

Per decision (one day-unit, live partitioned): $811 on IonQ Forte, $18 on IQM, $7 on Rigetti, $16-65 on IBM. Classical
compute: $0.15 a season live (one H100), $0.001 distilled, $0.12 to train once. So the IonQ-style "99 % classical cut"
does not transfer (12-qubit windows simulate in 0.1 s); what the distillation removes is the recurring QPU bill, which is
six digits a season only on IonQ Forte and four digits on superconducting hardware. Hardware noise at 0.3 % per CNOT
leaves the euros unchanged (the LP check absorbs it), so the cheap vendors are the relevant ones.

The paper's own sentence, for our season (`--cost` prints it): one decision a day-unit, live simulation 0.02 GPU h ->
0.017 GPU h of training once + 0.05 CPU core h; a re-decision every MTU (96 a day), 2.08 GPU h -> 0.017 GPU h once +
5.3 CPU core h, and 564 M live QPU shots -> 1.9 M once. The classical side is hours, not thousands of hours, because the
windows are 12 qubits; the shot count is where the 300x sits.

## Verdict

The paper's mechanism carries over: one distillation of the QPU's samples replaces every later QPU call (about 5,900
circuits a season -> 1,891 once), at equal decision quality and a quarter of the time, and it makes a re-decision every
15 minutes affordable where a live QPU is not. Three honest limits: the classical twin, trained on classical samples
of the same windows, is as good as the quantum-trained model, so the value lies in amortising the sampler, not in
quantum data; the learned noise calibration recovers little here; and none of these deciders beats the MILP in euros.
