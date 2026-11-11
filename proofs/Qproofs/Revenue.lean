/-
Revenue bounds for a gate-A bidding plan over finitely many scenarios (QEst 2026, team KratTrade).

X : plans (day-ahead position, capacity offers) committed before the scenario is known.
S : scenarios (the day's joint activation / price outcome).   f x s : settled revenue of plan x in scenario s.
p : the true scenario law;  q : the law a sampler (classical chain, quantum Born machine, QPU shots) hands the planner.

1. `ev_le_hindsight`      no plan, classical or quantum, beats hindsight in expectation (the EVPI gap is information).
2. `scenario_ge_any`      the plan optimised over the scenarios is at least as good as any fixed plan (VSS >= 0).
3. `ev_diff_le`           a sampler's error moves any plan's expected revenue by at most B * ||p - q||_1.
4. `sampler_regret`       planning on q instead of p loses at most 2 B ||p - q||_1.
5. `quantum_gain_bound`   so the most a better (quantum) sampler can add over a classical one is 2 B ||p - q_classical||_1.
-/
import Mathlib

open Finset BigOperators

variable {X S : Type*} [Fintype S]

/-- Expected revenue of plan `x` under scenario weights `w`. -/
def ev (w : S → ℝ) (f : X → S → ℝ) (x : X) : ℝ := ∑ s, w s * f x s

/-- 1. Hindsight bound. -/
theorem ev_le_hindsight [Fintype X] [Nonempty X] (w : S → ℝ) (hw : ∀ s, 0 ≤ w s)
    (f : X → S → ℝ) (x : X) :
    ev w f x ≤ ∑ s, w s * (Finset.univ.sup' Finset.univ_nonempty (fun x' => f x' s)) := by
  unfold ev
  apply Finset.sum_le_sum
  intro s _
  apply mul_le_mul_of_nonneg_left _ (hw s)
  exact Finset.le_sup' (fun x' => f x' s) (Finset.mem_univ x)

/-- 2. Value of the stochastic solution is non-negative. -/
theorem scenario_ge_any (p : S → ℝ) (f : X → S → ℝ) (xs xd : X)
    (h : ∀ x, ev p f x ≤ ev p f xs) : ev p f xd ≤ ev p f xs := h xd

/-- 3. A sampler's L1 error bounds the error on any plan's expected revenue. -/
theorem ev_diff_le (p q : S → ℝ) (f : X → S → ℝ) (B : ℝ) (hB : ∀ x s, |f x s| ≤ B) (x : X) :
    |ev p f x - ev q f x| ≤ B * ∑ s, |p s - q s| := by
  unfold ev
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]
  calc |∑ s, (p s * f x s - q s * f x s)|
      ≤ ∑ s, |p s * f x s - q s * f x s| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ s, B * |p s - q s| := by
        apply Finset.sum_le_sum
        intro s _
        rw [← sub_mul, abs_mul, mul_comm]
        exact mul_le_mul_of_nonneg_right (hB x s) (abs_nonneg _)

/-- 4. Planning on the sampler's law `q` instead of the truth `p` costs at most `2 B ||p - q||_1`. -/
theorem sampler_regret (p q : S → ℝ) (f : X → S → ℝ) (B : ℝ) (hB : ∀ x s, |f x s| ≤ B)
    (xp xq : X) (hq : ∀ x, ev q f x ≤ ev q f xq) :
    ev p f xp - ev p f xq ≤ 2 * B * ∑ s, |p s - q s| := by
  have h1 := abs_le.mp (ev_diff_le p q f B hB xp)
  have h2 := abs_le.mp (ev_diff_le p q f B hB xq)
  have h3 := hq xp
  linarith [h1.1, h1.2, h2.1, h2.2]

/-- 5. The revenue a quantum sampler can add over a classical one is bounded by the classical sampler's error. -/
theorem quantum_gain_bound (p qC : S → ℝ) (f : X → S → ℝ) (B : ℝ) (hB : ∀ x s, |f x s| ≤ B)
    (xp xC xQ : X) (hp : ∀ x, ev p f x ≤ ev p f xp) (hC : ∀ x, ev qC f x ≤ ev qC f xC) :
    ev p f xQ - ev p f xC ≤ 2 * B * ∑ s, |p s - qC s| := by
  have h := sampler_regret p qC f B hB xp xC hC
  have h' := hp xQ
  linarith
