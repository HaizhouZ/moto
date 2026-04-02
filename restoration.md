# Restoration Problem

This note describes the restoration phase that is **currently implemented** in
[`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp),
[`src/solver/nsp_impl/presolve.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp),
[`src/solver/nsp_impl/rollout.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp),
and the restoration runtime helpers under
[`include/moto/solver/restoration/`](/home/harper/Documents/moto/include/moto/solver/restoration).

The current implementation is an **explicit elastic restoration condensed onto
the global stage variables** $w=(x,u,y)$. The elastic variables
$(p,n,\nu_p,\nu_n,\lambda_c)$ are maintained only in a stage-local
restoration runtime and are eliminated analytically when assembling the global
Riccati system.

## Entry Condition

The outer SQP loop enters restoration when the normal globalization returns a
tiny-step failure while the primal infeasibility is still too large:

$$
\mathrm{action}_k = \mathrm{failure},
\qquad
\mathrm{failure\_reason}_k = \mathrm{tiny\_step},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

Equivalently,

$$
\alpha_k \le \alpha_k^{\min},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

The current implementation uses the IPOPT-style adaptive threshold
$\alpha_k^{\min}$ based on:

$$
\theta(x_k)=\mathrm{prim\_res\_l1},
\qquad
\theta_{\min}=\mathrm{ls.constr\_vio\_min},
$$

$$
\gamma_\theta=\mathrm{ls.primal\_gamma},
\qquad
\gamma_\phi=\mathrm{ls.dual\_gamma},
\qquad
\alpha_{\min}^{\mathrm{fac}}
=
\mathrm{settings.restoration.alpha\_min\_factor}.
$$

## Restoration Initialization

At restoration entry, the barrier parameter is reset to

$$
\bar\mu_0 = \max\!\bigl(\mu_j,\ \|c(w_k)\|_\infty\bigr).
$$

During restoration, `mu_bar` is updated with the **same method and the same
settings** as the normal IPM loop, but with restoration-specific drivers:

- if `settings.ipm.mu_method == monotonic_decrease`, the monotone test uses
  the restoration metrics `(primal_res_R, dual_R, comp_R)` instead of the normal
  NLP metrics
- if `settings.ipm.mu_method == mehrotra_predictor_corrector`, the adaptive
  update uses only the restoration local complementarity statistics from
  `(p,n,nu_p,nu_n)`

In both cases the state is separate: restoration evolves its own `mu_bar`,
while the outer `settings.ipm.mu` is restored when the restoration phase
returns.

The restoration reference point is simply the current iterate:

$$
w_R = w_k.
$$

The normal equality multipliers are reset:

$$
\lambda^{\mathrm{dyn}}_0 = 0,
\qquad
\lambda^{\mathrm{eq\_x}}_0 = 0,
\qquad
\lambda^{\mathrm{eq\_xu}}_0 = 0.
$$

The `u` and `y` references are snapshotted for proximal anchoring, and
componentwise proximal scaling still uses

$$
\sigma(\xi) = \frac{1}{\max(|\xi|,1)}.
$$

For the restoration-owned inequality block, the current implementation
initializes the positivity pair on its own interior point instead of copying
the outer normal-IPM slack state:

$$
t_0 = \max(-g(w_k), 1),
\qquad
\nu_{t,0} = \bar\mu_0 / t_0.
$$

Then the elastic inequality pair is initialized from the current violation

$$
v_{d,0} := g(w_k) + t_0
$$

using the same scalar elastic initializer as the equality block, producing
strictly positive $(p_{d,0}, n_{d,0}, \nu_{p_d,0}, \nu_{n_d,0}, \lambda_{d,0})$.

The restoration loop itself is capped by both the global SQP iteration budget
and the restoration-local limit:

$$
N_{\mathrm{resto}}
\le
\min\bigl(\texttt{settings.restoration.max\_iter},\,
\texttt{settings.max\_iter} - k_{\mathrm{entry}}\bigr).
$$

## Restoration NLP

Let

$$
w := (x,u,y),
\qquad
F(w)=0
$$

be the hard dynamics constraints, and let

$$
c(w)=0
$$

be the stacked equalities formed from `__eq_x` and `__eq_xu`, and let

$$
g(w) \le 0
$$

be the stacked inequalities formed from `__ineq_x` and `__ineq_xu`.

The restoration subproblem is

$$
\begin{aligned}
\min_{w,t,p_c,n_c,p_d,n_d}\quad &
\operatorname{obj}_R(w)
 + \varrho \mathbf{1}^T (p_c+n_c+p_d+n_d) \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p_c+n_c=0, \\
&
g(w)+t-p_d+n_d=0, \\
&
t \succeq 0,\quad
p_c \succeq 0,\quad n_c \succeq 0,\quad
p_d \succeq 0,\quad n_d \succeq 0 .
\end{aligned}
$$

Here:

- `F(w)=0` stays hard throughout restoration.
- `c(w)` is restored elastically through the `p_c/n_c` block.
- `g(w)` is restored through a restoration-owned inequality block
  `g(w)+t-p_d+n_d=0`.
- `\varrho` is the elastic exact-penalty weight, currently stored in
  `settings.restoration.rho_eq`.
- `\rho_\lambda` is the Hippo-style local elastic regularization, currently stored
  in `settings.restoration.lambda_reg`.
- `\operatorname{obj}_R(w)` contains the restoration-only terms on `w`, currently the
  proximal anchoring on `u/y`.
- These proximal terms are assembled only into the restoration base
  Lagrangian state (`lag_ / lag_jac_ / lag_hess_`); they do not modify the
  outer-NLP `cost_ / cost_jac_` bookkeeping used by the normal filter/objective.

Important semantic split:

- the **restoration original problem** is the elastic NLP with exact-penalty
  terms and positivity inequalities
  `p_c > 0`, `n_c > 0`, `t > 0`, `p_d > 0`, `n_d > 0`
- the logarithmic barrier with parameter `\bar\mu` is **not** part of that
  original restoration NLP
- instead, `\bar\mu` belongs to the IPM/Newton search model used to compute
  interior steps for those positivity inequalities
- therefore restoration public objective/KKT summaries should stay on the
  original restoration NLP, while line search and IPM-style step updates may still use the barrier-regularized
  search model

## Barrier-Regularized Search Lagrangian

To compute an interior Newton step for the positivity inequalities above, the
solver introduces the barrier-regularized search Lagrangian

$$
\mathcal{L}_{R,k}
=
\operatorname{obj}_{R,k}(w_k)
+ \varrho \mathbf{1}^T (p_{c,k}+n_{c,k}+p_{d,k}+n_{d,k})
- \bar\mu \sum_i \ln t_{k,i}
- \bar\mu \sum_i \ln p_{c,k,i}
- \bar\mu \sum_i \ln n_{c,k,i}
- \bar\mu \sum_i \ln p_{d,k,i}
- \bar\mu \sum_i \ln n_{d,k,i}
+ \lambda_{f,k}^T F_k(w_k)
+ \lambda_{c,k}^T \bigl(c_k(w_k)-p_{c,k}+n_{c,k}\bigr)
+ \lambda_{d,k}^T \bigl(g_k(w_k)+t_k-p_{d,k}+n_{d,k}\bigr).
$$

This search Lagrangian is an internal IPM model for the restoration step. It is
not the same object as the original restoration NLP above.

Its base gradient with respect to the global stage variables is

$$
g_{R,k}^{\mathrm{base}}
:=
\nabla_{w_k}\operatorname{obj}_{R,k}(w_k)
+ A_k^T \lambda_{f,k}
+ C_k^T \lambda_{c,k}
+ G_k^T \lambda_{d,k}.
$$

This is the quantity that belongs to the base stage gradient state seen by the
linear solve. In code it is what `activate_lag_jac_corr()` snapshots into
`base_lag_grad_backup` before any reduced correction is activated.

The barrier terms should not be conflated with the restoration
original objective that is reported publicly. They are search-only
regularization for the positivity inequalities.

The condensed elastic terms

$$
\Delta g_{R,k}^{\mathrm{cond,eq}}
:=
C_k^T M_{\rho,k}^{-1} b_{c,k},
\qquad
\Delta g_{R,k}^{\mathrm{cond,ineq}}
:=
G_k^T M_{d,k}^{-1} b_{d,k}
$$

are **not** part of the original restoration Lagrangian. They belong only to
the reduced system created after eliminating the restoration-local elastic
blocks, and so they must stay in `lag_jac_corr_` until activation.

## Search-Model KKT Residuals

Using Hippo-style notation, define

$$
A := F_w,
\qquad
C := c_w,
\qquad
G := g_w.
$$

The barrier-regularized search-model residuals are

$$
r_w := \nabla_w \operatorname{obj}_R(w) + A^T \lambda_f + C^T \lambda_c + G^T \lambda_d ,
$$

$$
r_f := F(w) ,
$$

$$
r_c := c(w)-p+n ,
$$

$$
r_d := g(w)+t-p_d+n_d ,
$$

$$
r_t := \lambda_d - \nu_t ,
$$

$$
r_p := \varrho \mathbf{1} - \lambda_c - \nu_p ,
$$

$$
r_n := \varrho \mathbf{1} + \lambda_c - \nu_n ,
$$

$$
r_{p_d} := \varrho \mathbf{1} - \lambda_d - \nu_p ,
$$

$$
r_{n_d} := \varrho \mathbf{1} + \lambda_d - \nu_n ,
$$

$$
r_{s,p} := \nu_p \odot p - \bar\mu \mathbf{1} ,
$$

$$
r_{s,n} := \nu_n \odot n - \bar\mu \mathbf{1} ,
$$

$$
r_{s,t} := \nu_t \odot t - \bar\mu \mathbf{1} ,
$$

$$
r_{s,p_d} := \nu_p \odot p_d - \bar\mu \mathbf{1} ,
$$

$$
r_{s,n_d} := \nu_n \odot n_d - \bar\mu \mathbf{1} .
$$

These are search-model residuals for the equality-elastic and restoration-owned
inequality blocks. The complementarity terms are barrier residuals from the IPM
search model; they are not part of the original restoration objective.

Introduce

$$
T_p := \operatorname{diag}(p),
\qquad
T_n := \operatorname{diag}(n),
\qquad
N_p := \operatorname{diag}(\nu_p),
\qquad
N_n := \operatorname{diag}(\nu_n).
$$

## Implemented Local Linearized System

For `lambda_reg > 0`, the current implementation uses the regularized local
linearization

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
N_p\,\delta p + T_p\,\delta \nu_p = -r_{s,p},
$$

$$
N_n\,\delta n + T_n\,\delta \nu_n = -r_{s,n}.
$$

This is the system actually condensed by the code. The bound blocks do not
couple directly to $w$; all coupling comes from

$$
c(w)-p+n=0.
$$

## Condensation Onto The Global Stage System

The elastic variables are **not** part of the global Riccati state. Instead,
the stage-local block is condensed exactly.

From the regularized stationarity rows,

$$
\delta \nu_p = \frac{\delta p + r_p - \delta \lambda_c}{\rho_\lambda},
\qquad
\delta \nu_n = \frac{\delta n + r_n + \delta \lambda_c}{\rho_\lambda}.
$$

Substituting into the complementarity rows gives

$$
\delta p
=
(T_p + \rho_\lambda N_p)^{-1} T_p\,\delta \lambda_c
-
(T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p}+T_p r_p),
$$

$$
\delta n
=
-(T_n + \rho_\lambda N_n)^{-1} T_n\,\delta \lambda_c
-
(T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n}+T_n r_n).
$$

Substituting these into

$$
C\,\delta w - \delta p + \delta n = -r_c
$$

produces

$$
M_\rho \,\delta \lambda_c = C\,\delta w + b_c,
$$

with

$$
M_\rho := T_p (T_p + \rho_\lambda N_p)^{-1} + T_n (T_n + \rho_\lambda N_n)^{-1},
$$

and

$$
b_c :=
r_c
+ (T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p}+T_p r_p)
- (T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n}+T_n r_n).
$$

Therefore

$$
\delta \lambda_c = M_\rho^{-1}(C\,\delta w + b_c).
$$

So the reduced stagewise stationarity solved by Riccati is

$$
g_{R,k}^{\mathrm{base}}
+ \Delta g_{R,k}^{\mathrm{cond}}
+ H_{R,k}\,\delta w_k
+ A_k^T \delta \lambda_{f,k}
+ C_k^T \delta \lambda_{c,k}
= 0.
$$

For residual checking, the correct starting point is therefore
$g_{R,k}^{\mathrm{base}}$, i.e. `base_lag_grad_backup`, not the already
corrected active `Q_(·)`.

The base stage gradient already contains

$$
C^T \lambda_c
$$

through the normal `lag_jac_` assembly. So the explicit elastic condensation
adds only

$$
C^T M_\rho^{-1} b_c
$$

to the global stage gradient. Define

$$
\hat\eta_c := M_\rho^{-1} b_c.
$$

Then the contribution written into the global stage system is

$$
\Delta \tilde Q_R{}_{(\cdot)} = \hat\eta_c^T c_{(\cdot)} ,
$$

$$
\Delta \tilde Q_R{}_{(\cdot,\cdot)} = c_{(\cdot)}^T M_\rho^{-1} c_{(\cdot)} .
$$

So the explicit elastic restoration is solved as:

- a **global** Riccati system only in $w=(x,u,y)$,
- plus a **local** restoration KKT block for
  $(p,n,\nu_p,\nu_n,\lambda_c)$,
- connected by exact Schur-complement condensation.

### Stage-Level Mapping Used By The Code

At the nullspace/Riccati stage, the stacked restoration equality Jacobians are
stored as

$$
C_u \;\widehat{=}\; \texttt{nsp.s\_c\_stacked},
\qquad
C_x \;\widehat{=}\; \texttt{nsp.s\_c\_stacked\_0\_K},
\qquad
c_0 \;\widehat{=}\; \texttt{nsp.s\_c\_stacked\_0\_k}.
$$

The helper computes

$$
\hat\eta_c = M_\rho^{-1} b_c,
\qquad
W = M_\rho^{-1}.
$$

Then presolve writes the condensed elastic terms into the stage system as

$$
\Delta g_{R,u} := \hat\eta_c^T C_u,
\qquad
\Delta g_{R,x} := \hat\eta_c^T C_x,
$$

and

$$
Q_{zz} \mathrel{+}= C_u^T W C_u,
\qquad
u_{0,p,K} \mathrel{+}= C_u^T W C_x,
\qquad
V_{xx} \mathrel{+}= C_x^T W C_x .
$$

The first-order condensed terms are not written directly into the active stage
gradient. They are first accumulated into `lag_jac_corr_`, and
`data_base::activate_lag_jac_corr()` is then responsible for folding them into
the active `Q_x/Q_u/Q_y` seen by the next Riccati solve, together with the
other pending reduced-system corrections.

So the global Riccati solve only sees a correction on the existing
$(x,u,y)$-blocks; it never sees explicit stage variables for $p$ or $n$.

## Local Step Recovery

After rollout provides $\delta w$, the local restoration steps are recovered
with the same regularized condensed coefficients:

$$
\delta \lambda_c = M_\rho^{-1}(\delta c + b_c), \qquad \delta c := C \delta w .
$$

$$
\delta \nu_p =
(T_p + \rho_\lambda N_p)^{-1}\!\left(N_p (r_p - \delta \lambda_c) - r_{s,p}\right) ,
$$

$$
\delta \nu_n =
(T_n + \rho_\lambda N_n)^{-1}\!\left(N_n (r_n + \delta \lambda_c) - r_{s,n}\right) ,
$$

$$
\delta p = \delta \lambda_c + \rho_\lambda \delta \nu_p - r_p,
\qquad
\delta n = -\delta \lambda_c + \rho_\lambda \delta \nu_n - r_n .
$$

In the implementation, the local residual increment is formed from the rolled
out primal step as

$$
\delta c = C_u \,\delta u + C_x \,\delta x .
$$

The recovered $\delta \lambda_c$ is then scattered back into the existing
equality-dual storage for `__eq_x` and `__eq_xu`, while
$\delta p,\delta n,\delta \nu_p,\delta \nu_n$ stay inside the restoration
runtime.

The implementation keeps a single shared helper for both presolve condensation
and rollout step recovery so that the sign convention cannot drift between the
two phases. The code now follows the same `\delta \nu`-first recovery order as
the paper, which avoids an extra cancellation-prone subtraction when recovering
the local dual steps.

### Hippo-Style Multiplier Regularization

Following Hippo and the implementation pattern in
[`ipm_constr.cpp`](/home/harper/Documents/moto/src/solver/ipm_impl/ipm_constr.cpp),
the local elastic block regularizes the explicit elastic bound rows in the
same `t + \rho \nu` style. The regularized local bound equations are

$$
\delta p - \delta \lambda_c - \rho_\lambda \delta \nu_p = -r_p,
\qquad
\delta n + \delta \lambda_c - \rho_\lambda \delta \nu_n = -r_n.
$$

This yields the same denominator pattern as Hippo's
$T_\rho = T + \rho N$. In our explicit elastic condensation it produces

$$
M_\rho = T_p (T_p + \rho_\lambda N_p)^{-1} + T_n (T_n + \rho_\lambda N_n)^{-1},
$$

so the effective denominators are of the form
$p + \rho_\lambda \nu_p$ and $n + \rho_\lambda \nu_n$. When
`lambda_reg == 0`, the implementation falls back to the unregularized explicit
elastic local-KKT formulas. This regularized branch keeps the condensed Hessian
modification bounded when $p,n$ or their duals approach the boundary.

## First-Order Correction Lifecycle

Only reduced-system restoration first-order terms follow the normal correction
buffer lifecycle:

- the explicit elastic condensed gradient correction
  $$
  \Delta g_R = C^T M_\rho^{-1} b_c
  $$
  is written into `lag_jac_corr_[__u]` and `lag_jac_corr_[__x]`
- `activate_lag_jac_corr()` then activates these pending reduced-system
  first-order corrections before the stage factorization

The restoration proximal terms on `u` and `y` are part of the restoration NLP
objective on $w=(x,u,y)$ itself. They are therefore added directly to the base
stage Lagrangian gradient `lag_jac_[__u]` and `lag_jac_[__y]` after each
restoration derivative evaluation, instead of being routed through
`lag_jac_corr_`.

So restoration no longer special-cases its first-order condensed term by
writing directly into the active gradient outside the normal correction path.

## What Is Kept And What Is Removed

During restoration:

- the original user cost is still **evaluated**
- the original filter/objective bookkeeping is still based on the normal pair
  `(prim_res_l1, barrier objective)`
- the original cost directional derivative is still available for globalization
- the restoration-local elastic variables $p,n$ do **not** contribute to the
  outer barrier objective or `log_slack_sum`
- the restoration-local barrier terms do **not** change the restoration
  original objective; they only affect the restoration search model

But the original user cost does **not** drive the restoration search direction.
The restoration direction is driven by:

- the explicit elastic local-KKT condensation for `__eq_x` and `__eq_xu`
- the proximal anchoring on `u` and `y`
- the hard dynamics constraints

## Dynamic Activeness

The implementation does **not** rebuild the problem graph and does **not** use
`update_active_status()` to enter restoration.

Instead, restoration is controlled by solver-phase activeness:

- restoration runtime state is attached to each stage
- when `settings.in_restoration == false`, the restoration hooks are dormant
- when `settings.in_restoration == true`, the restoration hooks are active

At the same time, the original `__eq_x` and `__eq_xu`:

- continue to be evaluated so that `c(w)` and `C=c_w` are available
- continue to contribute to primal infeasibility and the outer filter
- but no longer enter the restoration phase as hard equalities

## Globalization Semantics

The current implementation now uses **two acceptor layers**:

- the **outer** acceptor is still the original filter globalization for the
  original NLP
- the **inner restoration** acceptor uses restoration-only quantities
  `(primal_res_R, obj_R)`

At restoration entry, the outer filter is augmented once with the IPOPT-style
relaxed point derived from the entry iterate:

$$
\theta_{\mathrm{add}}=(1-\gamma_\theta)\theta_{\mathrm{ref}},
\qquad
\phi_{\mathrm{add}}=\phi_{\mathrm{ref}}-\gamma_\phi\theta_{\mathrm{ref}}.
$$

This relaxed point is inserted into the **outer** filter only once, right when
restoration starts.

Relative to Ipopt's full restoration phase, the current implementation should
be viewed as a lighter explicit elastic variant:

- it keeps the outer relaxed-filter augmentation at restoration entry,
- it uses a restoration-specific inner acceptor and restoration-specific
  metrics,
- but it does **not** build a separate nested restoration solver stack,
- and it still keeps the explicit elastic subproblem condensed onto the stage
  variables `w=(x,u,y)`.

Inside restoration iterations, line-search acceptance is driven by the
restoration acceptor, not by the outer filter. The restoration acceptor uses:

$$
\operatorname{primal\_res}_R
=
\max\Bigl(
\|F(w)\|_\infty,\,
\|c(w)-p+n\|_\infty
\Bigr),
$$

and

$$
\operatorname{obj}_R
=
\operatorname{obj}_{\mathrm{prox}}(w)
\;+\;
\varrho\,\mathbf 1^T(p+n)
\;-\;
\bar\mu \sum_i \ln p_i
\;-\;
\bar\mu \sum_i \ln n_i.
$$

It also tracks restoration-local dual and complementarity diagnostics:

$$ 
\mathrm{dual}_R
=
\max\Bigl(
\|r_w^{\mathrm{red}}\|_\infty,
\|r_{p_c}\|_\infty,
\|r_{n_c}\|_\infty,
\|r_t\|_\infty,
\|r_{p_d}\|_\infty,
\|r_{n_d}\|_\infty
\Bigr),
$$

$$
\mathrm{comp}_R
=
\max\Bigl(
\|r_{s,p_c}\|_\infty,
\|r_{s,n_c}\|_\infty,
\|r_{s,t}\|_\infty,
\|r_{s,p_d}\|_\infty,
\|r_{s,n_d}\|_\infty
\Bigr).
$$

These restoration quantities are evaluated from the **current trial stage
values**:

- `F(w)` comes from the trial `eval_val` update
- `c(w)` comes from the current trial values of `__eq_x` / `__eq_xu`
- `g(w)` comes from the current trial values of `__ineq_x` / `__ineq_xu`
- `\lambda_c` is read from the current scattered equality dual state
- `p_c,n_c,\nu_{p_c},\nu_{n_c}` are read from the equality elastic runtime
- `t,p_d,n_d,\nu_t,\nu_{p_d},\nu_{n_d},\lambda_d` are read from the
  restoration-owned inequality runtime after its affine step

Here $r_w^{\mathrm{red}}$ denotes the reduced restoration stationarity on
$w=(x,u,y)$ as seen by the condensed Riccati system:

- the stage-wise `u` component comes directly from `lag_jac_[__u]`
- the cross-stage component is the same `y/x` costate cancellation used by the
  Riccati rollout,
  $$
  r_{w,\mathrm{cross},k}^{\mathrm{red}}
  =
  \operatorname{lag\_jac}_{y,k}
  +
  \operatorname{lag\_jac}_{x,k+1} P_k
  $$

Once restoration is active, `dual_R` intentionally stops reusing the original
NLP dual residual as a whole. Instead it uses the restoration-phase
stationarity summary on $w$ together with the explicit local stationarity
residuals of both the equality and inequality restoration blocks.

So the restoration acceptor does **not** rely on cached local residuals from
the previous factorization.

Accepted restoration iterates update only the **restoration** filter history.
They do **not** add new points to the outer filter.

The outer barrier objective is still

$$
\phi(x)=f(x)-\mu\sum_i \log s_i,
$$

where the sum only runs over the normal active `ipm_constr` inequalities. The
restoration-local quantities $p,n,\nu_p,\nu_n$ are stored outside that IPM
traversal, so they do not pollute:

- `kkt.log_slack_sum`
- `kkt.barrier_dir_deriv`
- `kkt.objective`
- the outer complementarity residual

This is why the original cost still has to be evaluated during restoration,
even though it no longer defines the search direction.

At the same time, the **restoration** line search is allowed to use its own
barrier-regularized search objective built from:

- restoration original objective (`prox + exact penalty`)
- restoration barrier value from the positivity-IPM model

That search objective is internal to the restoration phase and is not the same
as the restoration original objective reported in public stats.

The outer filter is consulted again only when restoration wants to **exit**:
an accepted restoration step is mapped back to the original NLP and must also
be acceptable to the outer filter logic.

## Successful Return Mapping

When restoration exits successfully, the primal iterate is kept, but the
multiplier state is mapped back to the **original** NLP explicitly:

- the restoration-owned inequality state `(t,\nu_t)` is copied back to the
  outer normal inequality storage
- if the copied-back normal inequality multipliers exceed
  `settings.restoration.bound_mult_reset_threshold`, they are all reset to `1`
- equality multipliers for `__dyn / __eq_x / __eq_xu` are then handled by a
  separate cleanup step:
  - if `constr_mult_reset_threshold <= 0`, they are reset to zero
  - otherwise a least-squares estimate is formed from the original-NLP
    stationarity equation
    $$
    \min_\lambda \| g_{\mathrm{cost}} + J_{\mathrm{eq}}^T \lambda \|_2
    $$
    and accepted only if its maximum magnitude stays below
    `constr_mult_reset_threshold`
  - if the equality system is square, the implementation follows the Ipopt
    boundary behavior and resets these multipliers to zero instead of using the
    least-squares estimate

After this return mapping, the solver recomputes the **outer** KKT quantities
from the original NLP semantics.

## Success / Failure

The restoration phase is considered successful only when an accepted
restoration step satisfies both:

- the step is acceptable to the **outer** globalization logic
- it produces sufficient improvement relative to the entry infeasibility

That is, success requires

$$
\mathrm{outer\_acceptable}(x_{\mathrm{resto}})
$$

and

$$
\|r_{\mathrm{prim}}(x)\|_\infty
<
\eta_R \,
\|r_{\mathrm{prim}}(x_{\mathrm{entry}})\|_\infty.
$$

It is classified as `infeasible_stationary` only when:

- the restoration-local complementarity residual is already small,
- the **outer/original** primal infeasibility is still above tolerance,
- the **outer/original** dual residual is already below tolerance,
- and restoration has stalled or its line search has repeatedly failed.

This choice matches the current explicit elastic design: the restoration
subproblem itself can reach
$$
\operatorname{primal\_res}_R \approx 0
$$
because the elastic variables satisfy `c(w)-p+n=0`, while the original NLP can
still remain visibly infeasible.

Otherwise, if the restoration phase does not recover feasibility before the
global iteration budget is exhausted, the solver returns

$$
\mathrm{iter\_result}=\mathrm{restoration\_failed}
$$

or eventually `exceed_max_iter`.

## Iterative Refinement Status

After restoring the normal `lag_jac_corr_ -> activate_lag_jac_corr()`
lifecycle for the explicit elastic first-order term, the ordinary SQP
iterations again show normal iterative-refinement behavior: the stationarity
residuals are typically driven down to near machine precision within a few
correction sweeps.

The first restoration factorization can still show a very large *generic*
iterative-refinement residual. At the moment this should not be interpreted as
proof that the explicit elastic local-KKT recovery is wrong. The generic
`compute_kkt_residual()` checker is still derived from the standard stage-level
stationarity layout, while restoration is solving a reduced QP that already
contains the explicit local elastic Schur-complement terms.

So the current interpretation is:

- ordinary iterative refinement is healthy again after the correction-buffer
  lifecycle fix
- a large residual on the first restoration factorization indicates a mismatch
  between the generic residual checker and the restoration-condensed QP, not by
  itself a bug in the local elastic algebra

## Current Practical Status

The explicit local-KKT implementation has replaced the older condensed-only
prototype. Current behavior observed on `example/arm/run.py` is:

- restoration is triggered consistently for `--n-job 1` and `--n-job 4`
- the implementation is thread-consistent
- the solver still stalls on the terminal `arm_ee_constr` bottleneck
- current explicit restoration does not yet fully rescue the `arm` example

So the remaining limitation is no longer the old condensed analytic shortcut;
it is now the behavior of the explicit elastic restoration on difficult
terminal equalities.
