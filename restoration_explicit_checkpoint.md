# Explicit Elastic Restoration Checkpoint

This document is the math checkpoint for the **currently implemented**
explicit elastic restoration path. It is meant to be stricter than
[`restoration.md`](/home/harper/Documents/moto/restoration.md): the goal here is
to record only the pieces that are easier to lose while iterating on the
implementation:

- the exact local linearized system,
- the exact condensation formulas used by the code,
- the exact local step-recovery formulas,
- and the current implementation boundary versus Ipopt.

For the restoration NLP, stagewise Lagrangian, and globalization semantics,
see [`restoration.md`](/home/harper/Documents/moto/restoration.md).

The notation used below is the same as in the main note:

$$
w := (x,u,y), \qquad
A := F_w, \qquad
C := c_w,
$$

with local elastic variables and duals

$$
p,\ n,\ \nu_p,\ \nu_n,\ \lambda_c,
$$

and diagonal matrices

$$
T_p := \operatorname{diag}(p), \qquad
T_n := \operatorname{diag}(n), \qquad
N_p := \operatorname{diag}(\nu_p), \qquad
N_n := \operatorname{diag}(\nu_n) .
$$

The normal `__ineq_x / __ineq_xu` constraints are outside the current
restoration NLP.

The restoration-local residuals are

$$
r_c := c(w)-p+n ,
$$

$$
r_p := \varrho \mathbf{1} - \lambda_c - \nu_p ,
\qquad
r_n := \varrho \mathbf{1} + \lambda_c - \nu_n ,
$$

$$
r_{s,p} := \nu_p \odot p - \bar\mu \mathbf{1} ,
\qquad
r_{s,n} := \nu_n \odot n - \bar\mu \mathbf{1} .
$$

## Implemented Local Linearized System

The code does **not** solve the exact NLP KKT block directly. Instead, for
`lambda_reg > 0`, it uses the following Hippo-style regularized local
linearization:

$$
C\,\delta w - \delta p + \delta n = -r_c ,
$$

$$
\delta p - \delta \lambda_c - \rho_\lambda \delta \nu_p = -r_p ,
$$

$$
\delta n + \delta \lambda_c - \rho_\lambda \delta \nu_n = -r_n ,
$$

$$
N_p \,\delta p + T_p \,\delta \nu_p = -r_{s,p} ,
$$

$$
N_n \,\delta n + T_n \,\delta \nu_n = -r_{s,n} .
$$

This is the system that must be used when checking the current implementation.
It is the source of every formula below.

When `lambda_reg == 0`, the implementation falls back to the unregularized
branch:

$$
-\delta \lambda_c - \delta \nu_p = -r_p ,
\qquad
\delta \lambda_c - \delta \nu_n = -r_n ,
$$

with the same first, fourth, and fifth equations.

## Exact Condensation For `lambda_reg > 0`

This section derives the formulas used by the implemented regularized branch.

### Step 1: Eliminate `\delta \nu_p` and `\delta \nu_n`

From the regularized stationarity rows,

$$
\delta \nu_p = \frac{\delta p + r_p - \delta \lambda_c}{\rho_\lambda},
\qquad
\delta \nu_n = \frac{\delta n + r_n + \delta \lambda_c}{\rho_\lambda}.
$$

Substitute into the complementarity rows:

$$
N_p\,\delta p + T_p\,\delta \nu_p = -r_{s,p},
$$

$$
N_n\,\delta n + T_n\,\delta \nu_n = -r_{s,n}.
$$

This gives

$$
(T_p + \rho_\lambda N_p)\,\delta p - T_p\,\delta \lambda_c
=
-(\rho_\lambda r_{s,p} + T_p r_p),
$$

$$
(T_n + \rho_\lambda N_n)\,\delta n + T_n\,\delta \lambda_c
=
-(\rho_\lambda r_{s,n} + T_n r_n).
$$

Hence

$$
\delta p
=
(T_p + \rho_\lambda N_p)^{-1} T_p\,\delta \lambda_c
-
(T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p} + T_p r_p),
$$

$$
\delta n
=
-(T_n + \rho_\lambda N_n)^{-1} T_n\,\delta \lambda_c
-
(T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n} + T_n r_n).
$$

### Step 2: Eliminate `\delta p` and `\delta n`

Substitute the expressions above into

$$
C\,\delta w - \delta p + \delta n = -r_c .
$$

After collecting the `\delta \lambda_c` terms, one obtains

$$
M_\rho \,\delta \lambda_c = C\,\delta w + b_c ,
$$

where

$$
M_\rho :=
T_p (T_p + \rho_\lambda N_p)^{-1}
+
T_n (T_n + \rho_\lambda N_n)^{-1},
$$

and

$$
b_c :=
r_c
+
(T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p} + T_p r_p)
-
(T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n} + T_n r_n).
$$

Therefore

$$
\delta \lambda_c = M_\rho^{-1}(C\,\delta w + b_c).
$$

### Step 3: Condensed Global `w`-System

The global `w`-gradient already contains the base term

$$
C^T \lambda_c
$$

through the normal `lag_jac_` assembly. Therefore the explicit elastic
condensation contributes **only**

$$
C^T M_\rho^{-1} b_c
$$

to the stage gradient.

Define

$$
\widehat \eta_c := M_\rho^{-1} b_c .
$$

Then the implemented condensed contributions are

$$
\Delta \tilde Q_R{}_{(\cdot)} = \widehat \eta_c^T c_{(\cdot)},
$$

$$
\Delta \tilde Q_R{}_{(\cdot,\cdot)} = c_{(\cdot)}^T M_\rho^{-1} c_{(\cdot)} .
$$

Equivalently, if the base stage stationarity is

$$
r_w = \nabla_w \phi_R(w) + A^T \lambda_f + C^T \lambda_c ,
$$

then the condensed global system is

$$
\left(K_R + C^T M_\rho^{-1} C\right)\delta w + A^T \delta \lambda_f
=
-
\left(r_w + C^T M_\rho^{-1} b_c\right),
$$

with hard dynamics still given by

$$
A \delta w = -r_f .
$$

Equivalently, the reduced stagewise stationarity solved by Riccati is

$$
g_{R,k}^{\mathrm{base}}
+ \Delta g_{R,k}^{\mathrm{cond}}
+ H_{R,k}\,\delta w_k
+ A_k^T \delta \lambda_{f,k}
+ C_k^T \delta \lambda_{c,k}
= 0.
$$

Any residual check of this equation must start from
$g_{R,k}^{\mathrm{base}}$, not from an already-corrected active stage
gradient. That is why the exact restoration diagnostic uses
`base_lag_grad_backup`.

## Step Recovery For `lambda_reg > 0`

After the global solve returns $\delta w$, define

$$
\delta c := C\,\delta w .
$$

Then recover

$$
\delta \lambda_c = M_\rho^{-1}(\delta c + b_c).
$$

Following the paper's ordering, recover the bound-dual steps first:

$$
\delta \nu_p
=
(T_p + \rho_\lambda N_p)^{-1}\!\left(N_p (r_p - \delta \lambda_c) - r_{s,p}\right),
$$

$$
\delta \nu_n
=
(T_n + \rho_\lambda N_n)^{-1}\!\left(N_n (r_n + \delta \lambda_c) - r_{s,n}\right).
$$

Then recover the primal slack steps from the regularized stationarity rows:

$$
\delta p = \delta \lambda_c + \rho_\lambda \delta \nu_p - r_p,
\qquad
\delta n = -\delta \lambda_c + \rho_\lambda \delta \nu_n - r_n.
$$

The implementation now follows this `\delta \nu`-first ordering directly,
matching the paper and reducing cancellation when `\delta p` and
`\delta \lambda_c` are close.

Presolve and rollout must use the **same** formulas for

- $M_\rho$,
- $b_c$,
- $\delta \lambda_c$,
- $\delta p,\delta n$,
- $\delta \nu_p,\delta \nu_n$,

otherwise the condensed system and the recovered local step will drift apart.

## Unregularized Branch (`lambda_reg == 0`)

For completeness, the code also supports the unregularized branch.

The local equations are then

$$
C\,\delta w - \delta p + \delta n = -r_c ,
$$

$$
-\delta \lambda_c - \delta \nu_p = -r_p ,
\qquad
\delta \lambda_c - \delta \nu_n = -r_n ,
$$

$$
N_p \,\delta p + T_p \,\delta \nu_p = -r_{s,p},
\qquad
N_n \,\delta n + T_n \,\delta \nu_n = -r_{s,n}.
$$

This yields

$$
M_0 := T_p N_p^{-1} + T_n N_n^{-1},
$$

$$
b_{c,0} := r_c + N_p^{-1}(r_{s,p}+T_p r_p) - N_n^{-1}(r_{s,n}+T_n r_n),
$$

$$
\delta \lambda_c = M_0^{-1}(C\,\delta w + b_{c,0}).
$$

The paper-style recovery order is then

$$
\delta \nu_p = r_p - \delta \lambda_c,
\qquad
\delta \nu_n = r_n + \delta \lambda_c,
$$

followed by

$$
\delta p = -N_p^{-1}(r_{s,p}+T_p \delta \nu_p),
\qquad
\delta n = -N_n^{-1}(r_{s,n}+T_n \delta \nu_n).
$$

This is exactly the branch implemented in
[`resto_local_kkt.hpp`](/home/harper/Documents/moto/include/moto/solver/restoration/resto_local_kkt.hpp)
when `lambda_reg == 0`.

## IPOPT-Style Initialization

At restoration entry,

$$
\bar\mu_0 = \max(\mu_j,\ \|c(w_k)\|_\infty) .
$$

For each scalar component $c_i(w_k)$, the initialization solves the
barrier-smoothed one-dimensional elastic subproblem and sets

$$
n_i =
\frac{\bar\mu_0 - \varrho c_i
+ \sqrt{(\bar\mu_0 - \varrho c_i)^2 + 2 \varrho \bar\mu_0 c_i}}
{2 \varrho},
$$

$$
p_i = c_i + n_i,
$$

$$
\nu_{p,i} = \frac{\bar\mu_0}{p_i},
\qquad
\nu_{n,i} = \frac{\bar\mu_0}{n_i},
$$

$$
\lambda_{c,i} = \varrho - \nu_{p,i} = \nu_{n,i} - \varrho .
$$

The normal equality multipliers are reset when entering restoration.

## Mapping To The Code

Current implementation files tied to this checkpoint:

- [resto_local_kkt.hpp](/home/harper/Documents/moto/include/moto/solver/restoration/resto_local_kkt.hpp):
  shared helper implementing the initialization, exact condensation, and
  local-step recovery formulas above.
- [resto_elastic_constr.hpp](/home/harper/Documents/moto/include/moto/solver/restoration/resto_elastic_constr.hpp):
  stage-local storage for
  $(p,n,\nu_p,\nu_n,r_*,M_\rho^{-1},M_\rho^{-1}b_c,\delta\lambda_c,\dots)$.
- [resto_runtime.cpp](/home/harper/Documents/moto/src/solver/restoration/resto_runtime.cpp):
  lifecycle integration, initialization, lambda scatter/gather, and local-step
  recovery hooks.
- [presolve.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp):
  writes the condensed
  $C^T M_\rho^{-1} b_c$
  and
  $C^T M_\rho^{-1} C$
  terms into the global stage system.
- [rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp):
  leaves local elastic step recovery to the restoration runtime.
- [ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp):
  hooks restoration backup/restore, line-search bounds, and affine-step updates
  into the SQP loop.

## Current Boundary

This checkpoint reflects the explicit local-KKT formulation used by the
current code path. The remaining open question is solver behavior on difficult
examples such as `arm`, not the intended math represented here.

## Ipopt Comparison Snapshot

Compared with the behavior summarized in
[`RESTORATION_LINESEARCH.md`](/home/harper/Downloads/Ipopt-stable-3.14/RESTORATION_LINESEARCH.md),
the current implementation matches Ipopt in these high-level steps:

- restoration is entered from the globalization-failure path,
- the outer filter is augmented at restoration entry with a relaxed point,
- restoration uses its own inner acceptor and restoration-specific metrics,
- restoration exits only after an accepted inner step is also acceptable to
  the outer globalization logic and the original infeasibility improved enough.

The main differences are:

- Ipopt runs a **nested restoration NLP solve** with its own solver stack;
  the current implementation stays inside the same `ns_sqp` machinery and
  switches to an explicit elastic condensed subproblem.
- Ipopt's restoration problem can cover more than the current implementation;
  here restoration is limited to hard dynamics and the stacked
  `__eq_x/__eq_xu` equalities through `c(w)-p+n=0`, while normal
  `__ineq_x/__ineq_xu` are excluded.
- Ipopt performs explicit multiplier cleanup on successful return; the current
  implementation does not yet reset bound multipliers or recompute equality
  multipliers in an Ipopt-style postprocessing step.
