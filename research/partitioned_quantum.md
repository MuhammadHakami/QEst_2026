# Daedalus → partitioned quantum optimisation on a small QPU

Source: Mousavi, Lingsch, Mishra, De Lorenzis, *Self-Supervised Learning of Universal Latent Spaces for Operator
Learning*, arXiv:2611.05735 (ETH Zurich, Nov 2026), site daedalus.ethz.ch. Code: `qest/qpart.py`, wired into
`qest/qdecide.py` as solvers `p_qaoa`, `p_qaoa_add`, `p_wsqaoa`, `p_pce`, `p_sqa`.

## What the paper is (and is not)

A classical neural-operator paper. One functional autoencoder is pre-trained on synthetic analytic functions only,
frozen, and reused for every task; fields are cut into partitions, each normalised into a canonical frame and encoded
by the same encoder; a processor (transformer) is the only part that couples partitions. Its headline numbers are GPU
memory and speed: 64x larger problems, **92x larger models (101 M vs GAOT's largest)** on one 24 GB RTX 4090, 13.6x /
4.9x faster training, 10x faster latent rollouts, 537 M points by chunked inference. Nothing in it is quantum, and the
92x is parameters, not qubits. What transfers is the *structure*, which is the domain-decomposition structure that
quantum optimisation on small devices already uses (qbsolv, DQAOA, Benders; research/methods.csv O1, O4, S4).

## Mechanism by mechanism

| Paper (section) | What it does | Quantum counterpart in `qest.qpart` |
|---|---|---|
| Partitioner (§2, App. A.1) | k-means seeded along a Morton space-filling curve after a random rotation; P chosen per input | hours cut into windows of B qubits; on a 1-D time axis k-means = contiguous windows |
| Overlap (App. A.1) | partitions extended across boundaries, "the overlap between subdomains in Schwarz-type methods" | windows overlap by B/4 bits |
| Segment mean (App. E, decoder) | a point covered by several partitions gets the mean of their outputs | `mode='add'`: all windows solved against the same pinned solution, overlap bits by majority (additive Schwarz) |
| Processor couples partitions (§2, App. E) | transformer / GNN over partition embeddings, the only coupling | classical loop: each window's QUBO absorbs its pinned neighbours (exact conditioning), swept until no window improves (`mode='mult'`, multiplicative Schwarz) |
| PartNorm static frame (§2, App. A.2-A.3) | coordinates and values normalised per partition; shift / scale kept as static features, replayed by the decoder: exact equivariance | each window's conditioned QUBO is divided by its rms before the circuit; the constant and scale are kept for scoring |
| Synthetic pre-training, frozen backbone (§1-2) | trained on analytic functions only, never on the task | one LR-QAOA schedule (p, Δ) picked on random synthetic QUBOs of size B, frozen for every window, day and site: zero angle training on the task (B = 8, 12: p 12, Δ 0.9; B = 16: p 12, Δ 0.7; B = 24 transfers B = 16's) |
| Effective spatial frequency (§3, App. A.4) | more partitions pull every partition into the training band; resolution becomes an inference-time choice | window size B is chosen per problem without retraining; the synthetic score worsens with B (0.028 at 8, 0.038 at 12, 0.057 at 16): smaller windows sit deeper in the schedule's band, at the cost of more coupling |
| Random rotation / windowed attention (App. A.1, E) | random rotation before partitioning; random windows | `shift=True`: window boundaries move by a random offset every sweep, so no cut is permanent |
| Cached embeddings (§4) | encode once, train the processor on cached latents | the day QUBO is built once (384 LP checks) and shared by every solver; reservoir features computed once for 11 targets x 5 ridge strengths |
| Chunked inference (§4, App. D.4) | encode / decode one chunk at a time: 537 M points on one GPU | windows run one after another on the same 24 qubits (`mult`), or floor(device / B) windows per call (`add`) |
| Latent rollout (§4) | step in latent space, decode once | the carried solution is the state; warm-start QAOA starts each window from it; only windows whose neighbours changed are re-sent |
| Limitation (§5) | a partition above the training band is lost | windows whose coupling to the rest is strong lose information at the cut; overlap + shifting + sweeps are the remedy, not a guarantee |

## Reach: one 24-qubit register, 92 windows

N binaries on a B-qubit register with overlap o need ceil((N - o) / (B - o)) windows per sweep: 2,208 binaries
(= 92 x 24) take 122 windows of 24 qubits (o = 6). That is the sense in which a small device "acts like" a 92x
larger one: it covers the problem, sweep by sweep. It does not reproduce a 2,208-qubit state; cross-window
correlations are carried only by the loop. Measured below against simulated annealing and greedy on the whole
problem, and on the real day QUBOs against the exact optimum.

## Lean: the waste removed in our structure

1. **No angle training** (over-processing): the frozen schedule replaces per-instance optimisation (PCE used
   4 restarts x 300 COBYLA steps per day; partitioned PCE uses 1 per window, the sweeps are the restarts).
2. **No re-sends** (motion): a window goes back to the QPU only if a neighbour changed.
3. **No re-compilation** (waiting): CUDA-Q kernels compiled once per process (was ~6 s per QAOA call).
4. **Build once, solve many** (inventory): one QUBO per day and unit, shared by all solvers and window sizes.
5. **Right-sized hardware**: B is a knob (8-24 qubits; PCE needs 3-5), so the same code runs on 24-qubit and
   100-qubit devices (`add` packs 4 windows of 24 into one 100-qubit call).
6. **Constraints stay classical** (defects): MW, SoC and reserves live in the LP, so no penalty terms inflate the
   QUBO or produce infeasible bids.

## Results

**Reach** (`results/qpart_reach.json`): a time-coupled synthetic QUBO of 2,208 binaries (couplings at lags 1, 2, 4),
24-qubit windows, overlap 6, frozen schedule (B = 16's, transferred).

| Solver | Energy | Circuits | Sweeps | Wall time |
|---|---:|---:|---:|---:|
| Partitioned LR-QAOA, multiplicative | -1251.02 | 370 | 3 | 192 s |
| Partitioned LR-QAOA, additive | -1251.02 | 493 | 4 | 245 s |
| Partitioned warm-start QAOA | -1251.02 | 370 | 3 | 140 s |
| Simulated annealing, whole problem | -1250.45 | | | 1.2 s |
| Greedy, whole problem | -1223.78 | | | |

A 24-qubit register covers a 92x larger problem and lands 0.05 % below SA's energy. The true optimum is unknown, and
SA is about 100x faster on a GPU-simulated QPU. This is coverage, not a 2,208-qubit quantum state.

**Real day QUBOs** (`results/qpart_days_test.csv`): Q4, 25 days x 2 units, exact optimum known by enumeration. Hit =
the best sample is the optimum.

| Solver | Qubits | Hit | Circuits |
|---|---:|---:|---:|
| Monolithic LR-QAOA | 24-25 | 0.84 | 1 |
| Partitioned LR-QAOA / warm-start QAOA, B = 12, mult | 12 | 0.76 | 10.1 |
| Partitioned PCE, B = 12 | 4 | 0.74 | 10.5 |
| Partitioned SQA, B = 12 | 12 | 0.70 | 10.2 |
| Partitioned LR-QAOA, B = 12, add | 12 | 0.58 | 11.4 (6.9 QPU calls) |
| Partitioned, B = 8 (qaoa / wsqaoa / pce / sqa / add) | 3-8 | 0.68 / 0.68 / 0.72 / 0.58 / 0.50 | 15-16 |

Halving the register costs 8 points of hit rate; additive (parallel) windows are cheaper in calls but worse.

**Forecaster** (`qforecast --part`, `results/qforecast_skill_part.csv`): 32 gate-A inputs (own 10, every other
target's last value and week lag, wind and consumption) in 4 overlapping windows through the same frozen 10-qubit
reservoir, one ridge on top. Q4 MAE: better than the plain reservoir on 3 of 11 targets (day-ahead 47.3 -> 44.4,
Chronos-2 36.0), better than the classical twin on the same inputs on 8 of 11, better than Chronos-2 on 1 (mFRR, by
0.03). Blended with Chronos-2 it ties the plain blend. Ridge strength hit the grid's top (100) on every target.

**Euros** (`results/days_test_qdecide_chronos_part.parquet`): 12-qubit windows in the full plan -> MILP -> official
settlement pipeline, Q4, battery + hybrid. Baseline Chronos-2 + MILP EUR 1,812,548.

| Decider | Q4 EUR | vs MILP |
|---|---:|---:|
| Monolithic LR-QAOA (24-25 qubits) | 1,811,924 | -0.03 % |
| Partitioned SQA | 1,801,416 | -0.61 % |
| Partitioned LR-QAOA, additive | 1,800,625 | -0.66 % |
| Partitioned LR-QAOA / warm-start QAOA | 1,800,367 | -0.67 % |
| Partitioned PCE (4 qubits) | 1,799,013 | -0.75 % |

Partitioned vs monolithic LR-QAOA: EUR -11,557 +/- 9,434 (1.2 s.e., 12 wins / 13 losses vs MILP). Not significant, but
every variant lands on the same side. Verdict: partitioning works as designed (a 12-qubit or even 4-qubit register
runs the day problem, a 24-qubit one runs a 2,208-variable one), and it costs a little quality, as the paper's own
limitation section predicts. It is how our method fits today's devices, not a source of extra euros.
