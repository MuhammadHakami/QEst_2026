/-
Optimality guarantees of the hybrid quantum-classical strategies used in the KratTrade pipeline (QEst 2026).
x : the discrete decisions handed to the quantum solver (charge / discharge pattern, integer MW offers, puzzle bits).
y : the continuous decisions left to the classical LP (day-ahead MW, state of charge).   f x y : revenue.

1. `decomposition_exact`   quantum-master / classical-subproblem split loses nothing (qday, qdecide, qbenders).
2. `benders_stop`          a Benders loop that stops when the cut model meets the true value has the optimum.
   `benders_gap`           before that, the cut model bounds the remaining gap.
3. `block_step`            a partitioned block update never lowers the objective (qpart, qubo.partitioned).
   `block_fixed_point`     and a point no block can improve is optimal over every block neighbourhood.
4. `hybrid_ge_classical`   best-of-samples with the classical incumbent in the pool is never worse than classical.
5. `penalty_feasible`      a QUBO penalty larger than the revenue range makes every maximiser feasible,
   `penalty_optimal`       and that maximiser is optimal among feasible plans (qubo.build's lambda).
-/
import Mathlib

variable {X Y : Type*}

/-- 1. If the classical solver returns a best response `ystar x` for every discrete choice `x`, and the quantum
solver maximises the value function `V x = f x (ystar x)`, the pair is a global optimum of the joint problem. -/
theorem decomposition_exact (f : X → Y → ℝ) (V : X → ℝ) (ystar : X → Y)
    (hV : ∀ x, V x = f x (ystar x)) (hy : ∀ x y, f x y ≤ f x (ystar x))
    (xs : X) (hx : ∀ x, V x ≤ V xs) : ∀ x y, f x y ≤ f xs (ystar xs) := by
  intro x y
  calc f x y ≤ f x (ystar x) := hy x y
    _ = V x := (hV x).symm
    _ ≤ V xs := hx x
    _ = f xs (ystar xs) := hV xs

/-- 2. Benders: `U` is the master's model built from valid cuts (`V ≤ U`). If the master's maximiser `xk` closes
the gap (`U xk = V xk`), it maximises the true value. -/
theorem benders_stop (V U : X → ℝ) (hU : ∀ x, V x ≤ U x) (xk : X)
    (hmax : ∀ x, U x ≤ U xk) (hgap : U xk = V xk) : ∀ x, V x ≤ V xk := by
  intro x
  linarith [hU x, hmax x]

/-- 2b. At any iteration the optimality gap of the incumbent is at most the master's bound minus its value. -/
theorem benders_gap (V U : X → ℝ) (hU : ∀ x, V x ≤ U x) (xk : X)
    (hmax : ∀ x, U x ≤ U xk) : ∀ x, V x - V xk ≤ U xk - V xk := by
  intro x
  linarith [hU x, hmax x]

/-- 3. Partitioned solver: replacing `x` by the best point of a neighbourhood that contains `x` (one block
re-solved with the others pinned) never lowers the objective. -/
theorem block_step (f : X → ℝ) (N : X → Set X) (x x' : X) (hx : x ∈ N x)
    (hbest : ∀ z ∈ N x, f z ≤ f x') : f x ≤ f x' := hbest x hx

/-- 3b. When no block improves `x`, `x` is optimal over the union of all block neighbourhoods. -/
theorem block_fixed_point {ι : Type*} (f : X → ℝ) (N : ι → X → Set X) (x : X)
    (hfix : ∀ b, ∀ z ∈ N b x, f z ≤ f x) : ∀ z ∈ ⋃ b, N b x, f z ≤ f x := by
  intro z hz
  obtain ⟨b, hb⟩ := Set.mem_iUnion.mp hz
  exact hfix b z hb

/-- 4. Sample-then-verify with the classical plan `x0` in the candidate pool: the selected plan `xb` is at least
as good as the classical one, whatever the quantum samples `S` are (noise included). -/
theorem hybrid_ge_classical (f : X → ℝ) [DecidableEq X] (S : Finset X) (x0 xb : X)
    (hb : ∀ z ∈ insert x0 S, f z ≤ f xb) : f x0 ≤ f xb :=
  hb x0 (Finset.mem_insert_self x0 S)

/-- 5. QUBO penalty: `v x = 0` iff the plan is valid, violations count at least 1, and `lam` exceeds the revenue
any plan can gain over a valid one. Then every maximiser of `f - lam * v` is valid. -/
theorem penalty_feasible (f v : X → ℝ) (lam : ℝ) (x0 : X) (h0 : v x0 = 0)
    (hv : ∀ x, v x = 0 ∨ 1 ≤ v x) (hlam : ∀ x, f x - f x0 < lam) (xs : X)
    (hmax : ∀ x, f x - lam * v x ≤ f xs - lam * v xs) : v xs = 0 := by
  rcases hv xs with h | h
  · exact h
  · exfalso
    have h1 := hmax x0
    rw [h0] at h1
    have h2 := hlam xs
    have hl : 0 < lam := by
      have := hlam x0
      simpa using this
    have h3 : lam * 1 ≤ lam * v xs := mul_le_mul_of_nonneg_left h hl.le
    linarith

/-- 5b. ... and it is optimal among the valid plans. -/
theorem penalty_optimal (f v : X → ℝ) (lam : ℝ) (xs : X) (hs : v xs = 0)
    (hmax : ∀ x, f x - lam * v x ≤ f xs - lam * v xs) : ∀ x, v x = 0 → f x ≤ f xs := by
  intro x hx
  have h := hmax x
  rw [hx, hs] at h
  linarith
