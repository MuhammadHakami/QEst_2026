# Where the qubits can still earn: the uncertainty line, measured (10-11 Nov 2026)

**Verdict in four lines.** (1) The highest quantum gain measurable on the Estonia 2026 data is on the organisers' own
quantum metric, the day QUBO: LR-QAOA on 24-25 qubits returns the optimum on 94-98 % of day-units, PCE on 4 qubits per
12-bit window on 74 %, and a model distilled once from 1,891 QPU windows on 74-82 % with zero QPU calls afterwards
(research/partitioned_quantum.md, research/distilled_quantum.md). (2) In euros, no qubit moved the result: every decider
ties the MILP within +-0.7 % on every input set (section 3), and the quantum-enhanced sampler's per-step mixing edge does
not reach the plan (sections 1-2). (3) The euros came from classical work on the uncertainty: a driver-based activation
model (+3.2 % a year) and a two-stage plan with a settlement-faithful recourse (+5.4 % more under the realistic 25 MW cap,
+4 to +7 % under the organisers' rules, every month positive). (4) The commercially efficient quantum design that follows,
and that runs end to end here, is: sample the QPU once, distil, decide classically for the season ($1.4-152k once across
vendors against $4-472k a season live), and bring a QPU back as a scenario source only for a joint target that cannot be
sampled exactly, benchmarked against tempering. This is stated as a measured negative on quantum euros, not dressed up.

Goal set by the user: the highest quantum gain the Estonia 2026 data allows, a commercially efficient use of the qubits,
grounded in the 2026 state of the art (research/README.md, "Added 10 Nov 2026"). This note records what was built and
measured today, in the order it happened, with the losses. Walk-forward mode throughout (`QEST_WALK=1`,
research/leakage_audit.md); the deterministic reference is `milp_act_rank` (research/activation_model.md s6).

## 1. The only place a sampler matters: the two-stage scenario plan

The plan of record is deterministic (expected activation energy per MW, gate-C guard in settlement). A sampler of joint
activation scenarios (`qest.qscen` exact chain, `qest.qscen_q` quantum-enhanced MCMC) only earns if a two-stage plan
over scenarios beats the deterministic plan. Until today the scenario plans lost 16-32 % and the recourse model was the
suspect. Today's steps, seven Q3 days (3, 17 Jul; 5, 14, 27 Aug; 9, 22 Sep), battery + hybrid, organisers' settlement:

| Plan | 7-day EUR | vs `milp_act_rank` | What changed |
|---|---:|---:|---|
| `milp_act_rank` (deterministic, rank-gated marginals) | 1,064,491 | - | reference |
| `milp_chronos` (deterministic, climatology) | 962,966 | -9.5 % | |
| `milp_scen` (8 exact chain scenarios, free recourse) | 736,856 | -30.8 % | as before: delivered <= activated, the plan prices out at will |
| `milp_scenf` (faithful recourse) | 944,137 | -11.3 % | delivered = activated, shortfall at >= the activation price (single imbalance pricing) |
| `milp_scenf_cvar` (CVaR 0.3) | 903,660 | -15.1 % | |
| `milp_scenf_q` (quantile scenarios, faithful) | 853,984 | -19.8 % | |
| `milp_sceng` (faithful + settlement's SoC guard, fixed point) | 723,019 | -32.1 % | priced out exactly when the scenario's SoC drifts > tol from the expected SoC |
| `milp_sceng4` / `milp_sceng16` | 755,074 / 683,305 | -29 % / -36 % | |

The faithful recourse recovered half the loss. The guard made it worse, and the diagnostic says why: valuing the
*deterministic* plan inside the model (first stage pinned, qest.stoch `fixed`) gives 25,853 / 87,617 / 13,200 EUR on
14 Aug / 17 Jul / 9 Sep against 83,154 / 157,575 / 141,934 paid by settlement (priced-out share in the model 0.47 per
direction-MTU, in settlement 0.43-0.44: the guard count is right, the value is not). The model is too pessimistic, not
the optimiser. One cause found: the chain's bits mark "mFRR price crosses DA +- 5" (the actfc target) while our bids sit at
DA + margin, so a bit treated as a certain full activation over-drains the battery in every scenario; `day_scenarios` now
rescales the bits so the scenario mean equals the plan's own probability at the margin.

### 1b. Mean-consistent scenarios: the first scenario plan that beats the deterministic one

Same seven days, scenarios rescaled to the plan's activation probability at the margin:

| Plan | 7-day EUR | vs `milp_act_rank` |
|---|---:|---:|
| `milp_act_rank` | 1,064,491 | - |
| `milp_mk` (chain marginals, deterministic) | 956,837 | -10.1 % |
| **`milp_scenf`** (8 exact chain scenarios, faithful recourse) | **1,101,880** | **+3.5 %** |
| `milp_sceng` (+ guard) | 820,450 | -22.9 % |
| `milp_sceng_cvar` | 819,052 | -23.1 % |

The faithful plan buys more day-ahead energy (-1.17 M vs -0.86 M) and sells more of it into both products (mFRR 979k vs
908k, aFRR 915k vs 668k), with a larger positive imbalance component (238k vs 208k): it takes the same one-sided position
as the deterministic plan, further. The guard model stays too pessimistic and is dropped from the line. Seven days are a
probe; the walk-forward season is section 1c.

### 1c. Walk-forward, Q3 and Q4, organisers' settlement (paired 95 % against `milp_act_rank`)

| Plan (faithful recourse, rank-gated marginals) | Q3, 92 days | vs best deterministic | Q4, 25 days | vs best deterministic |
|---|---:|---:|---:|---:|
| `milp_act_rank` (deterministic reference) | 8,943,649 | - | 1,878,086 | - |
| `milp_scenf4` (4 exact chain scenarios) | 9,324,980 | +4.3 % (+381k +- 425k, 55 / 92) | 1,948,190 | +3.7 % (+70k +- 169k, 16 / 25) |
| `milp_scenf` (8 exact) | 9,219,319 | +3.1 % (+276k +- 470k, 52 / 92) | 1,901,646 | +1.3 % (+24k +- 147k) |
| `milp_scenf16` (16 exact) | 9,342,563 | +4.5 % (+399k +- 422k, 60 / 92) | 1,961,887 | +4.5 % (+84k +- 166k, 16 / 25) |
| `milp_scenf_ls` (8 from the single-flip MCMC day sampler) | 9,623,423 | **+7.6 % (+680k +- 537k, 56 / 92)** | 1,962,701 | +4.5 % (+85k +- 194k) |
| `milp_scenf_qs` (8 from the quantum-enhanced MCMC day sampler) | 9,253,975 | +3.5 % (+310k +- 506k, 61 / 92) | 1,958,515 | +4.3 % (+80k +- 136k, 15 / 25) |

Reading: the two-stage plan with faithful recourse beats the best deterministic plan in both quarters, +3 to +8 % on Q3
and +1 to +4.5 % on Q4; 16 exact scenarios is significant on Q3 at the 95 % level (+399k +- 422k is not, +680k +- 537k is).
This is the first plan that improves on `milp_act_rank`, and it is the uncertainty object (joint activation scenarios)
doing it. What it is *not*: a reward for sampler fidelity. The single-flip MCMC sampler, the worst of the three on the
day benchmark (section 2: TV 0.086, UP runs 6.0 against 6.8), earns the most; the quantum sampler sits between it and the
exact chain. Scenario sets with shorter activation runs let the plan buy more day-ahead energy (-12.5 M against -11.4 M
on Q3) and sell it into the activations, the same one-sided position as before, taken further. The differences between
scenario sources are inside each other's intervals (+-0.4-0.5 M), so the honest statement is: the plan pays for
scenarios, not for their accuracy; between 4 and 16 scenarios and between samplers it cannot tell. Whole year and the
25 MW cap: section 1d.

### 1d. The whole year, walk-forward (291 days, 8 Jan - 25 Oct), organisers' settlement

Chain for January and February fitted to 1 March (no actfc-covered history before; in-sample for those two months,
printed; research/leakage_audit.md s2), everything else from earlier months only. Paired 95 % against `milp_act_rank`
(38,688,597); the Chronos-2 baseline is 37,480,487.

| Plan | Year EUR | vs `milp_act_rank` | vs Chronos-2 + MILP | Days won |
|---|---:|---:|---:|---:|
| `milp_scenf4` (4 exact chain scenarios) | **41,280,072** | **+6.7 % (+2.59 M +- 0.90 M)** | +10.1 % | 193 / 291 |
| `milp_scenf16` (16 exact) | 40,740,475 | +5.3 % (+2.05 M +- 0.81 M) | +8.7 % | 192 / 291 |
| `milp_scenf_ls` (8, single-flip MCMC) | 40,538,346 | +4.8 % (+1.85 M +- 0.85 M) | +8.2 % | 167 / 291 |
| `milp_scenf` (8 exact) | 40,273,730 | +4.1 % (+1.59 M +- 0.88 M) | +7.4 % | 174 / 291 |
| `milp_scenf_qs` (8, quantum-enhanced MCMC) | 40,236,167 | +4.0 % (+1.55 M +- 0.82 M) | +7.3 % | 194 / 291 |

Every variant is significant over the year and positive in 8-10 of 10 months (the exceptions: -2.4 % and -1.3 % in April
for the 8 / 16 exact sets, -0.7 % in August for 8 exact, -1.5 % in January for the local sampler). Two-stage planning on
joint activation scenarios with a settlement-faithful recourse is now the plan of record under the organisers' rules:
+7 to +10 % over the Chronos-2 + MILP baseline, of which about a third is the activation model (+3.2 %) and two thirds the
scenarios. The sampler ranking is again not by fidelity: 4 exact scenarios beat 16, the single-flip MCMC beats the quantum
one, and all five sit within each other's intervals.

Mar-Oct only (239 days, clean of the two in-sample chain months; `milp_act_rank` 27,971,742): `milp_scenf4` +6.1 %
(+1.71 M +- 0.77 M, 153 / 239), `milp_scenf16` +4.7 % (+1.31 M +- 0.68 M), `milp_scenf_ls` +6.4 % (+1.79 M +- 0.79 M),
`milp_scenf` +2.8 % (+0.77 M +- 0.68 M), `milp_scenf_qs` +4.1 % (+1.15 M +- 0.76 M). The in-sample months do not carry
the result.

Where the money comes from (year, `milp_scenf4` against `milp_act_rank`): day-ahead purchases -50.5 M against -44.7 M,
aFRR energy 29.8 M against 21.0 M (+42 %), mFRR 36.8 M against 36.5 M, capacity 18.2 M against 18.9 M, imbalance 8.3 M
against 8.5 M, UP activation 418 GWh against 361 GWh. The scenario plan buys more energy and sells it into aFRR
activations; the deviation volume is unchanged (232 against 227 GWh) and the guard fires as often (52.8k against 50.8k
priced-out direction-MTUs). It is the same one-sided position with the aFRR leg sized by scenarios instead of a mean.

### 1e. The realistic settlement: 25 MW activation cap (`QEST_ACT_CAP=25`), cap-aware scenario plan

The TSO activates at most the volume it needs; until the Baltic dashboard volumes are in, the 25 MW cap is the realistic
bracket (research/environment_audit.md). The stochastic MILP now carries the cap exactly: per scenario, MTU and
direction the activated MW g_k <= offer_k, sum g <= vol, forced to min(sum offer, vol) by one binary, delivered = share x g
(qest.stoch; with one scenario it reproduces the volume-aware deterministic planner within 0.7 %). The first version that
put the cap on the expected energy instead of the MW lost 6-9 %; corrected:

| Plan, 25 MW cap | Q3, 92 days | vs `milp_act_rank` (6,036,478) | Q4, 25 days | vs `milp_act_rank` (1,327,164) |
|---|---:|---:|---:|---:|
| `milp_scenf` (8 exact scenarios) | 6,537,381 | **+8.3 % (+501k +- 183k, 65 / 92)** | 1,435,138 | **+8.1 % (+108k +- 72k, 17 / 25)** |
| `milp_scenf16` (16 exact) | 6,554,144 | **+8.6 % (+518k +- 174k, 69 / 92)** | 1,440,006 | **+8.5 % (+113k +- 90k, 18 / 25)** |
| `milp_scenf_ls` (8, single-flip MCMC day sampler) | 6,617,897 | +9.6 % (+581k +- 174k, 72 / 92) | 1,462,542 | +10.2 % (+135k +- 83k, 18 / 25) |
| `milp_scenf_qs` (8, quantum-enhanced MCMC day sampler) | 6,567,639 | +8.8 % (+531k +- 179k, 67 / 92) | 1,460,775 | +10.1 % (+134k +- 78k, 19 / 25) |

Under the cap the deterministic plan hardly trades energy (DA -10k on Q3) because the capped activation energy cannot pay
for it; the scenario plan buys 1.36 M of day-ahead energy and more than doubles the aFRR energy income (3.1 M against
1.4 M) at unchanged mFRR, capacity and imbalance. This is the largest gain under the realistic settlement found in the
project, significant in both quarters, and it comes from the uncertainty object, not from a forecaster or a solver.

### 1f. The whole year under the 25 MW cap: the number of record for the realistic settlement

291 days, walk-forward, cap-aware scenario plan against the volume-aware deterministic plans (`milp_act_rank` 30,375,085;
Chronos-2 + MILP 30,277,111):

| Plan, 25 MW cap, whole year | Revenue | vs `milp_act_rank` | vs Chronos-2 + MILP | Days won | Mar-Oct only (239 d) |
|---|---:|---:|---:|---:|---:|
| `milp_scenf` (8 exact scenarios) | **32,023,349** | **+5.4 % (+1.65 M +- 0.53 M)** | +5.8 % | 204 / 291 | +7.1 % (+1.45 M +- 0.32 M, 176 / 239) |
| `milp_scenf16` (16 exact) | 31,927,225 | +5.1 % (+1.55 M +- 0.60 M) | +5.5 % | 204 / 291 | +7.0 % (+1.44 M +- 0.40 M, 178 / 239) |

By month (8 exact): Jan +1.2, Feb +2.7, Mar +3.1, Apr +7.6, May +11.4, Jun +5.0, Jul +9.2, Aug +6.3, Sep +9.3, Oct +8.1 %:
positive in every month, growing with the activation model's history. (`milp_scenf4` was run under the cap on Jan-Jun only,
+1 to +13 % by month; not quoted for the year.) Under the organisers' rules the same plan family is +4 to +7 %
(section 1d). The two settlements now agree on the direction and roughly the size, which the deterministic activation
model never achieved (+3.2 % against +0.3 %): sizing the aFRR leg by scenarios is worth more when the TSO caps the volume,
because the deterministic plan then stops trading energy altogether.

Settlement itself shows what the deterministic plan does: priced out on 45 % of direction-MTUs (87 of 192 per unit-day),
626 MWh of UP activation a day, a positive imbalance component (+208k on the 7 days): a one-sided position (buy DA, hold a
high SoC, offer UP) whose downside the gate-C guard removes and whose deviations the official rules settle at the
imbalance price without penalty. A two-stage plan that prices the tails honestly declines that position, and loses.

## 2. The sampler, measured on the day target

`qest.qscen_q` now samples whole days (12 windows of 8 MTUs = 16 qubits each, every chain conditioned on its own previous
state, one batched call per window) and carries parallel tempering (Marshall & Egger, arXiv:2602.06171: replicas at
T = 1, 1.6, 2.5, 4, neighbour swaps every step). Six held-out Q3 days, 64 scenarios a day, 300 steps a window, against
2,000 exact forward draws (results/qscen_q_day.csv):

| Day sampler | TV per MTU | UP rate (exact 0.416) | UP run length (exact 6.8) | s / day |
|---|---:|---:|---:|---:|
| exact chain, 64 draws (finite-sample floor) | 0.072 | 0.441 | 7.2 | 0.002 |
| single-bit flips | 0.086 | 0.454 | 6.0 | 0.4 |
| single-bit flips + tempering | 0.066 | 0.421 | 6.7 | 1.2 |
| quantum proposal | 0.085 | 0.417 | 5.2 | 30 |
| quantum proposal + tempering | 0.064 | 0.416 | 6.2 | 120 |

Reading: on this target the quantum proposal's per-step edge from the window benchmark (TV 0.071 vs 0.088, autocorrelation
28 vs 52 steps, results/qscen_q_window_v1.csv) does not survive at the day level at a fixed step budget: marginals equal to
the local chain, persistence worse, 80x the wall time on a GPU statevector (a QPU would be slower still). Tempering helps
both samplers equally and is the cheaper upgrade. Because the target is a chain, the exact forward sampler beats every
MCMC at no cost; the MCMC samplers are the stand-in for the non-chain target (aFRR, both borders, the Finnish price).
### 2b. Window benchmark with tempering (12 windows of 16 qubits, 300 steps x 256 chains, results/qscen_q.csv)

| Window sampler | TV to exact marginals | acceptance | autocorrelation (steps) | s / window |
|---|---:|---:|---:|---:|
| single-bit flips | 0.109 | 0.161 | 34.8 | 0.1 |
| single-bit flips + tempering | **0.037** | 0.140 | 14.9 | 0.2 |
| quantum proposal | 0.096 | 0.101 | 21.8 | 50 |
| quantum proposal + tempering | 0.058 | 0.084 | **12.2** | 198 |

The quantum proposal decorrelates fastest per step (12-22 steps against 15-35), which is the Layden / Marshall-Egger
claim and it holds here; but a classical tempering ladder buys a larger improvement in the marginals (TV 0.109 -> 0.037)
at a quarter of a second, and the combination is not better than tempering alone on TV. Per step the quantum proposal
wins; per second, per euro and per circuit it does not, on a 16-qubit window whose exact answer is a 65,536-row table.

## 3. The deciders at the best information level

`qdecide` forecaster 'actrank': the quantum deciders on the rank-gated activation marginals, walk-forward, baseline
`milp@actrank` = `milp_act_rank` (the free MILP on the same inputs). Battery + hybrid, organisers' settlement, paired 95 %:

| Decider on the day QUBO | Q3 (92 d) | vs MILP | Q4 (25 d) | vs MILP | QUBO optimum hit |
|---|---:|---:|---:|---:|---:|
| MILP (HiGHS) | 8,943,649 | - | 1,878,086 | - | - |
| LR-QAOA, 24-25 qubits (CUDA-Q statevector) | 8,879,055 | -0.72 % (-65k +- 87k, 43 / 92) | 1,882,673 | +0.24 % (+5k +- 19k, 15 / 25) | 0.94 / 0.98 |
| distilled QPU, partitioned 12-qubit windows (`p_nqs`) | 8,882,992 | -0.68 % (-61k +- 97k) | 1,872,369 | -0.30 % (-6k +- 15k) | 0.63 / 0.74 |
| sampling control (best of the QUBO's training samples) | 8,915,771 | -0.31 % (-28k +- 71k) | 1,877,109 | -0.05 % (-1k +- 26k) | 0.00 |

Reading: with the best information the deciders stay within +-0.7 % of the MILP and inside noise, exactly as on the
Chronos-2 inputs (research/partitioned_quantum.md). LR-QAOA finds the QUBO optimum on 94-98 % of day-units and still
earns no more than the MILP: the ceiling is the QUBO surrogate (r2 0.89-0.92 against the LP value), not the solver. No
solver, qubit count or distillation can lift this line above the MILP; its value is cost (the distilled model runs the
season for $0 of QPU time) and the demonstration that the day's charge / discharge pattern is a 24-bit problem a 2026
device solves. The euros are in the inputs (section 1), where the sampler lives.

## 4. Qubit budget and commercial efficiency: what a QPU should be asked to do here

The 2026 entries that convinced juries (research/README.md) did three things: a real problem with company data, a hybrid
pipeline that runs end to end, and a stated commercial effect. The ones that convinced reviewers added a random or
classical control and a cost per decision. This project has all five; the numbers below are the honest version of
"optimal utilisation of the qubits" for this problem.

| Where a QPU could sit | Qubits | Calls a season | What it earns here | Verdict |
|---|---:|---:|---|---|
| Deciding the day's charge / discharge pattern (LR-QAOA, monolithic) | 24-25 | 582 | +-0.7 % of the MILP, inside noise, on every input set tried; QUBO optimum hit 94-98 % | solved; the surrogate is the ceiling, not the solver |
| Same, partitioned 12-qubit windows, live | 12 | 5,878 | same euros, $4-472k a season across vendors | 5 % of the qubits of the monolithic job, same answer |
| Same, distilled once (IonQ's route, `qdistill`) | 12 (training only) | 1,891 once, then 0 | same euros, $1.4-152k once, $0 after | the only live-QPU design that is affordable per decision |
| PCE on the windows (`qsota`) | 4 | 5,878 | 0.74 hit against 0.76 for QAOA | the 2026 qubit-efficient route, 3x fewer qubits again |
| Sampling joint activation scenarios (quantum-enhanced MCMC, `qscen_q`) | 16 per window, 12 windows a day | 3,492 windows a season x 300 steps | plan euros equal to the exact chain and to single-bit flips; per-step mixing 1.5-2x faster, wall time 80-250x | not needed while the target is a chain; the candidate job when it is not |
| Forecasting (quantum reservoir, `qforecast`, `qrc`) | 8-12 | daily | loses to Chronos-2 alone, trims balancing errors blended | a feature, not a forecaster |

What moved the euros (whole year, walk-forward, organisers' rules; 25 MW cap in brackets): Chronos-2 + MILP 37.48 M ->
activation model with ranking gate +3.2 % (+0.3 %) -> two-stage scenario plan +6.7 % more (+8 % on Q3-Q4). Both steps
are classical and cost cents of compute. The qubits did not move euros on this data, and the design that keeps them in the
pipeline at an affordable price is: sample once, distil, decide classically; bring the QPU back as a scenario source only
when the joint target (aFRR + both borders + the Finnish price) stops being exactly samplable, and then benchmark it
against tempering, which is the cheaper classical upgrade this note found.

Costs quoted at Nov 2026 list prices (Braket $0.30 a task + per shot: IonQ Forte $0.08, IQM Garnet $0.00145, Rigetti
Cepheus $0.000425; IBM $96 a QPU-minute); the GPU statevector time for the sampler is on an H100-class card.

## 5. The highest quantum gain on the organisers' own metric: the P3 puzzle solved at half the qubits (11 Nov)

The quantum track (deck slide 9 / 11) scores the portfolio puzzle retrospectively on every Q4 hour: P1 (9 qubits), P2 (15)
and P3 (30: two hours linked by the batteries' SoC). P1 and P2 were already solved by every quantum method; P3 was the
weak point: PCE on 6 qubits reached 84.5 % of the exact optimum and returned it on 1.7 % of instances, and no statevector
solver fits 30 qubits. Two 2026-style changes (`qest.qubo.partitioned`, `qest.qsota.pce(groups=)`):

| P3, 300 instances | Qubits | % of optimum | Optimum hit |
|---|---:|---:|---:|
| PCE (before) | 6 | 84.5 | 0.017 |
| PCE, problem-aware one-hot decoding (arXiv:2605.20163) | 6 | 84.5 | 0.017 |
| PCE, 8 qubits, 48 restarts | 8 | 86.6 | 0.000 |
| Partitioned by hour (LR-QAOA p=1, BF-DCQO, FALQON, SQA: identical) | 15 | 92.1 | 0.730 |
| Alternating partitions: hours, then sites across both hours (SoC link inside) | 15 / 10 | 99.7 | 0.977 |
| **Alternating partitions, 4 random restarts** | **15 / 10** | **99.92** | **0.993** |
| Tabu search (classical reference, 30 bits) | - | 99.92 | 0.907 |
| Random 4,096 shots + polish (control) | - | 30.8 | 0.010 |

Reading: the gain is the decomposition, not the solver: every gate-model solver returns the same plans, because a 10-15
qubit block with its neighbours pinned is solved to its optimum by all of them, and the alternating partition carries the
SoC coupling that the hour partition cuts (the Daedalus recipe of research/partitioned_quantum.md, applied to the
organisers' puzzle). PCE's problem-aware decoding changed nothing here: its plans were already valid; its limit at 6-8
qubits is the optimisation, not the decode. On the competition's quantum metric this is the highest result the project has:
the P3 optimum on 99.3 % of hours with a register a 2026 device has (15 qubits), equal in value to the best classical
heuristic and above it on hit rate, with a random control at 31 %. It is a quantum-track result, not a euro result: the
puzzle values prices that are known.

## 6. Forecasting SOTA pass (11 Nov): a quantum fine-tuning head and Mistral's Timestral, on the lever that pays

The user's list (IonQ resource papers, QTSF, the quantum fine-tuning "wing" head, 4-qubit digital reservoirs, t0,
Timestral) was screened for what is testable here with a classical control on the activation targets, the input whose
quality moves euros. Two were built and measured; the rest are in research/README.md.

### 6a. Quantum fine-tuning head on the activation drivers (`qest.qhead`, after Kim, Baglio et al., arXiv:2609.05408)

The paper's core: a 12-qubit data re-uploading circuit (R = 2, Z readout on 5 wires, linear head) on frozen Chronos
embeddings beats a 1.2-3 M-parameter MLP on power-grid event classification with 3,240 quantum parameters. Here the head
sits on the gate-A driver features of qest.actfc and predicts m_up / m_dn / aFRR shares, walk-forward (monthly refits,
D-2 lag), AdamW with weight decay (both heads over-fitted without it). Controls on identical inputs and training loop: an
MLP head (2 x 64); and the LightGBM forecaster of record. Two seeds, the mean of the two reported (results/qhead_*.csv):

| Q3 (92 d) | quantum head, 12 qubits (~3k params) | MLP head (same inputs) | LightGBM (actfc) | climatology |
|---|---:|---:|---:|---:|
| mFRR UP: AUC / Brier / log-loss | 0.585 / 0.235 / 0.663 | 0.572 / 0.260 / 0.735 | 0.571 / 0.243 / 0.679 | 0.564 / 0.236 / 0.665 |
| mFRR UP: within-day Spearman | 0.128 | 0.160 | 0.163 | 0.113 |
| mFRR DOWN: AUC / Brier | 0.572 / 0.246 | 0.563 / 0.273 | 0.548 / 0.255 | 0.557 / 0.247 |
| aFRR UP share: Brier | 0.164 | 0.167 | 0.163 | 0.172 |
| aFRR DOWN share: Brier | 0.160 | 0.160 | 0.155 | 0.162 |

| Q4 (25 d) | quantum head | MLP head | LightGBM | climatology |
|---|---:|---:|---:|---:|
| mFRR UP: AUC / Brier | 0.528 / 0.249 (seeds 0.49 / 0.57) | 0.501 / 0.313 | 0.566 / 0.244 | 0.483 / 0.250 |
| aFRR UP / DOWN share: Brier | 0.159 / 0.152 | 0.175 / 0.169 | 0.154 / 0.148 | 0.163 / 0.156 |

Reading: the paper's claim replicates in its own terms: the quantum head beats the classical MLP head on identical inputs
on every target and both quarters, with a thousandth of the parameters, and it is the best-calibrated mFRR model on Q3
(log-loss 0.663 against 0.665 climatology, 0.679 LightGBM). Against the stronger classical control it does not win: LightGBM
ranks hours within the day better (Spearman 0.16 against 0.13), carries the aFRR shares, and is stable on Q4 where the
quantum head is seed-sensitive. Training cost: 12 s a month on one GPU. Placement: a committee member for the mFRR
probabilities, not a replacement; its Q4 instability must be fixed (seed ensembles, the paper's wings carrying new inputs)
before it enters the plan.

### 6b. Mistral Timestral-209M as the price forecaster (`qest.fc_tsfm`)

Drop-in beside Chronos-2: same two information groups, same known covariates, 56-day context, zero-shot. MAE by target,
Timestral against Chronos-2 (results/skill_timestral_{val,test}.csv):

| Target | Q3: Timestral / Chronos-2 / LightGBM | Q4: Timestral / Chronos-2 / LightGBM |
|---|---:|---:|
| day-ahead price | 32.7 / 32.7 / 40.3 | 35.7 / 36.0 / 42.5 |
| mFRR price | **63.1** / 68.1 / 71.9 (-7.2 %) | 69.2 / 67.4 / 66.0 (+2.7 %) |
| imbalance price | **67.5** / 72.6 / 75.6 (-7.0 %) | 67.4 / 64.6 / 62.9 (+4.3 %) |
| aFRR UP share | **0.355** / 0.368 / 0.374 (-3.5 %) | 0.351 / 0.340 / 0.343 (+3.3 %) |
| aFRR DOWN share | **0.338** / 0.350 / 0.370 (-3.5 %) | 0.333 / 0.322 / 0.334 (+3.6 %) |
| capacity prices (4) | -0.7 to -5.9 % | -1.5 to +0.2 % |

Reading: on Q3 Timestral is the best forecaster of the balancing targets by a clear margin and ties on the day-ahead
price; on the 25 October days it gives part of that back. Inference: 18 s for 92 days on one GPU (Chronos-2: minutes).
Euros follow in 6c: the whole plan family (`QEST_FC=timestral`: prices, aFRR shares and the scenario chain from Timestral;
`tsens`: the two foundation models averaged quantile by quantile) on Q3 and Q4, both settlements.

## 7. "Classical forecasts collapse to the average over the horizon": measured, and what follows for the qubits (11 Nov)

The user's hypothesis: autoregressive classical forecasters flatten toward the mean as the horizon grows, a quantum model
keeps the dispersion and is noise-aware, so it should earn more. Tested on our forecasters first (results/forecast_collapse.csv;
Q3 + Q4, median forecast against the realised series; the D-2 group is a 24-48 h horizon, the day-ahead price 9-39 h):

| Target, hours of the delivery day | spread kept, Chronos-2 | Timestral | LightGBM | within-day spread kept (C-2 / T / LGBM) |
|---|---:|---:|---:|---|
| mFRR price, 0-7 / 8-15 / 16-23 | 0.52 / 0.40 / 0.56 | 0.47 / 0.38 / 0.53 | 0.39 / 0.35 / 0.43 | 0.29-0.53 / 0.31-0.60 / 0.26-0.45 |
| imbalance price | 0.48 / 0.36 / 0.48 | 0.42 / 0.32 / 0.45 | 0.34 / 0.31 / 0.41 | 0.17-0.36 / 0.23-0.45 / 0.20-0.42 |
| aFRR UP share | 0.52 / 0.45 / 0.57 | 0.46 / 0.36 / 0.52 | 0.49 / 0.39 / 0.50 | 0.10-0.31 / 0.16-0.38 / 0.23-0.34 |
| day-ahead price | 0.87 / 0.81 / 0.79 | 0.80 / 0.74 / 0.71 | 0.50 / 0.58 / 0.53 | 0.58-0.64 / 0.63-0.68 / 0.33-0.58 |

(spread kept = std of the median forecast / std of the outcome; 1 = no collapse.) The premise holds: the balancing medians
carry 30-57 % of the realised spread and 10-60 % of the within-day spread, for every classical model, and it is not an
autoregression artefact (Timestral and Chronos-2 forecast the whole horizon in one pass): a conditional median *should*
shrink toward the mean when skill is low. That is exactly why the deterministic plan on medians left money on the table
and the scenario plan (section 1) recovered 5-8 %: the plan needs the dispersion, not a sharper point.

What follows for the qubits is a precise job, not a slogan: the quantum model that can "keep the dispersion" is a
generative one, sampled, conditioned on the drivers, scored against the classical generator of record (the logistic
Markov chain, qest.qscen) on held-out log-likelihood and on euros through the same plan. That object is built in section 8
(`qest.qgen`, a conditional quantum circuit Born machine over 16-qubit activation windows, exact-likelihood training on
the statevector, shot sampling on hardware). Literature check (research/README.md): quantum generative time-series models
exist for finance (QGAN / QCBM on S&P 500, MPS generators, QDiffusion-TS on IQM, 2606.27561) with windows of 5-20 points;
none for energy prices or activations, and no paper shows a quantum sampler avoiding mean reversion over a long horizon.
"Noise-aware" in the literature means conformal error bars that absorb hardware noise (Park & Simeone), which our pipeline
already does classically (split-conformal widening of the 10-90 % band). On the horizon claim itself the 2025-26 quantum
reservoir literature says the opposite of the premise: closed-loop QRC forecasts degrade with horizon by error accumulation
(chaotic maps, arXiv:2509.12071; memory saturating over a limited horizon, 2604.03469), and no paper found compares a quantum
sampler against a classical one on mean reversion. So the claim is treated as ours to test (section 8), not as established.

### 6c. Euros on Timestral (`QEST_FC=timestral`) and on the two-model average (`tsens`), walk-forward, organisers' rules

| Plan | Q3 on Chronos-2 | on Timestral | on the average | Q4 on Chronos-2 | on Timestral | on the average |
|---|---:|---:|---:|---:|---:|---:|
| Baseline MILP | 8,112,684 | +6.0 % (+484k +- 292k) | +2.3 % (+189k +- 168k) | 1,821,939 | -6.2 % (-113k +- 184k) | -4.5 % (-82k +- 78k) |
| + activation model, ranking gate | 8,943,649 | -0.3 % (+- 236k) | +0.1 % | 1,878,086 | **+3.8 % (+71k +- 62k)** | +3.5 % (+66k +- 58k) |
| + scenario plan (8) | 9,219,319 | -0.7 % (+- 201k) | -0.4 % | 1,901,646 | +1.8 % (+34k +- 109k) | +0.8 % |

Q4 at the 25 MW cap: activation plan +3.2 % (+43k +- 44k) on Timestral, scenario plan +1.3 % (+- 46k). Reading: swapping
the price forecaster moves the *baseline* by +-6 % with opposite signs in the two quarters, and moves the plan of record
by less than its interval. Once the TSO's activations are modelled and planned on scenarios, the price forecaster is no
longer the lever (and the first `tsens` run, which silently fell back to the LightGBM knobs, lost EUR 0.5 M on Q4: the
knobs matter more than the forecaster). Timestral stays as an alternative input; no claim is made for it.

## 8. A quantum generative model for the scenarios (`qest.qgen`): measured, negative so far

Conditional quantum circuit Born machine: 16 qubits = 8 MTUs x (up, down), 4 layers of RY(condition) RZ RY + CZ ring,
conditioned on the previous state, the hour block and the driver summary, exact-likelihood training on the statevector,
refitted before each month. Held-out windows, per MTU (results/qgen_bench_{val,test}.csv):

| Month | Born machine log-lik | classical chain log-lik | UP rate data / Born / chain | UP run length data / Born / chain |
|---|---:|---:|---|---|
| Jul | -0.820 | -0.545 | 0.37 / 0.37 / 0.51 | 5.9 / 2.5 / 7.8 |
| Aug | -0.802 | -0.524 | 0.42 / 0.39 / 0.34 | 6.8 / 2.9 / 5.0 |
| Sep | -0.743 | -0.474 | 0.40 / 0.36 / 0.40 | 7.3 / 2.3 / 6.5 |
| Oct | -0.689 | -0.469 | 0.43 / 0.41 / 0.46 | 7.9 / 2.7 / 7.5 |

The circuit gets the activation *rate* right (better than the chain in July and August) and misses the *persistence*: runs
of 2.5 quarter-hours against 7 in the data, 0.22-0.28 nats per MTU behind the chain. A shallow hardware-efficient ansatz
on a ring does not express the 4-state Markov structure that a 16-parameter logistic transition captures. This is the
result the Lean bound predicts in sign: with a classical sampler that is exact on its target, a quantum sampler has
nothing to add (`quantum_gain_bound`), and a worse generative fit can only cost. Not carried into the plan. Next variant if
pursued: an IQP / MPS-structured circuit with nearest-neighbour-in-time entanglers (research/plan_v2.md step 3).

## 9. Lean 4 proofs (`proofs/Qproofs/Revenue.lean`, Mathlib, builds clean, axioms: propext, Classical.choice, Quot.sound)

`ev_le_hindsight`, `scenario_ge_any`, `ev_diff_le`, `sampler_regret`, `quantum_gain_bound`: statements in
research/plan_v2.md s1. `cd proofs && lake build`. The bounds are over the day's joint scenario law; in euros they are
loose (B is a day's revenue scale, the joint L1 distance is not small), so they are used for direction, not magnitude: they
say where a quantum sampler can and cannot earn.

## 10. Optuna, NAS and PostNAS on the winning strategy, leakage-free (11 Nov)

**Optuna on the plan** (`scripts/tune_scen.py`, TPE, 64 trials): knobs (scenario count, aFRR / mFRR bid margins, activation
haircuts, gate-C drift tolerance) tuned on 46 Q2 days only (every 2nd day of Apr-Jun, walk-forward calibration); Q3 and Q4
are never seen. Q2: EUR 8.04 M against 7.83 M with the inherited knobs (+2.7 %); the change that matters is the drift
tolerance (0.016 -> 0.19 of the battery's energy: the guard prices out later). Applied unchanged, out of sample:

| Q3 + Q4, 117 days | Organisers' rules | vs baseline | 25 MW cap | vs baseline |
|---|---:|---:|---:|---:|
| Chronos-2 + MILP | 9,934,623 | - | 7,292,317 | - |
| activation plan (rank gate) | 10,821,735 | +8.9 % | 7,363,642 | +1.0 % |
| scenario plan, 4 exact, inherited knobs | 11,273,170 | +13.5 % | 8,063,440 | +10.6 % |
| **scenario plan, 4 exact, Q2-tuned** | **11,640,522** | **+17.2 %** (+367k +- 191k over untuned, 93 / 117) | **8,280,196** | **+13.5 %** (+217k +- 112k, 83 / 117) |
| Q2-tuned, single-flip MCMC scenarios | 11,565,679 | +16.4 % | 8,244,529 | +13.1 % |
| Q2-tuned, quantum-MCMC scenarios (default proposal) | 11,416,524 | +14.9 % | 8,203,925 | +12.5 % |

By quarter the quantum-sampled plan is the best plan on Q4 in both settlements (2,082,236, +6.9 % over untuned, +- 112k;
capped 1,531,286, +4.8 % +- 45k) and the weakest of the three on Q3 (9,334,288 against 9,603,878 exact); over the 117
days the three scenario sources are inside each other's intervals, as before.

**NAS + PostNAS on the sampler circuit** (`scripts/nas_sampler.py`): Optuna over the quantum proposal (mixer weight range,
evolution-time range, Trotter depth), objective = autocorrelation x depth cost subject to TV, on June windows; scored on 16
held-out Q3 windows (results/nas_sampler.csv):

| Proposal | TV to exact | autocorrelation (steps) | acceptance | Trotter steps |
|---|---:|---:|---:|---:|
| default (Layden ranges: mixer 0.25-0.60, t 2-20) | 0.103 | 18.5 | 0.114 | 8 |
| NAS (mixer 0.18-0.38, t 4.4-9.6) | **0.079** | **11.7** | 0.172 | 8 |
| PostNAS (fewest Trotter steps within 10 %) | = NAS | = NAS | | 8 (7 already loses > 10 %) |
| single-bit flips | 0.113 | 27.3 | 0.171 | - |

The searched proposal mixes 1.6x faster than the published ranges and 2.3x faster than single flips per step, with the
acceptance of the local chain. PostNAS found no depth to cut. Euros with it: section 10b (running).

### 10b. The quantum-sampled plan with the NAS proposal (Q2-tuned knobs, out of sample)

| 117 days, Q3 + Q4 | exact chain | quantum MCMC, default proposal | quantum MCMC, NAS proposal | NAS - default | NAS - exact |
|---|---:|---:|---:|---:|---:|
| Organisers' rules | 11,640,522 | 11,416,524 | 11,586,530 (+16.6 % vs baseline) | +170k +- 450k | -54k +- 308k |
| 25 MW cap | 8,280,196 | 8,203,925 | 8,223,139 (+12.8 %) | +19k +- 106k | -57k +- 80k |

Q4 alone: NAS-quantum 2,110,259 (+15.8 % over baseline, +74k +- 93k over exact), capped 1,525,987 (+15.0 %). The NAS
proposal moves the quantum-sampled plan from the weakest to level with the exact chain; no difference between scenario
sources is significant. Whole year under the sim-to-real preset: research/sim2real.md s4 (28.70 M -> 30.68 M, +6.9 %).
