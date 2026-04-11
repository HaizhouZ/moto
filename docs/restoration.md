# Restoration Math Note

This note records the restoration overlay formulation used by the elastic
restoration wrappers in:

- `resto_prox_cost`
- `resto_eq_elastic_constr`
- `resto_ineq_elastic_ipm_constr`

The goal here is narrow:

- state the restoration NLP
- derive the local KKT systems
- show the condensation used by the wrappers
- show how the eliminated local Newton steps are recovered

There is no extra `lambda_reg` term or separate regularization multiplier in
this derivation.

## 1. Restoration NLP

Let `w` denote the stage primal variables seen by the restoration overlay. In
practice this is the stage `x/u/y` primal block used by the normal solver.

Let

$$
F(w)=0
$$

denote the hard dynamics constraints,

$$
c(w)=0
$$

the hard equality constraints that are made elastic in restoration, and

the inequality constraints that are made elastic in restoration.

For ordinary one-sided inequalities, the source residual is simply

$$
g(w)\le 0.
$$

For boxed inequalities,

$$
lb \le g(w) \le ub,
$$

the OCP still stores only the base residual `g(w)`. The bound sides are tracked
through runtime metadata:

$$
r_{ub}(w)=g(w)-ub,
\qquad
r_{lb}(w)=lb-g(w).
$$

The restoration problem is

$$
\begin{aligned}
\min_{w,p_c,n_c,\{t_s,p_s,n_s\}_{s \in S}}\quad &
\phi_R(w)
\;+\;
\rho_c \mathbf{1}^T(p_c+n_c)
\;+\;
\rho_d \sum_{s \in S}\mathbf{1}^T(p_s+n_s) \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p_c+n_c=0, \\
&
r_s(w)+t_s-p_s+n_s=0,\qquad \forall s \in S, \\
 &
t_s \succ 0,\quad p_s \succ 0,\quad n_s \succ 0,\qquad \forall s \in S, \\
&
p_c \succ 0,\quad n_c \succ 0.
\end{aligned}
$$

Here:

- `\phi_R(w)` is the proximal restoration cost
- `\rho_c > 0` is the equality elastic weight
- `\rho_d > 0` is the inequality elastic weight
- `S` is the set of present inequality sides
- each present side gets its own local positive slack `t_s`

The side residuals are

$$
r_{ub}(w)=g(w)-ub,\qquad r_{lb}(w)=lb-g(w).
$$

For a one-sided inequality, only the `ub` side is present. For a boxed
inequality, both `ub` and `lb` sides may be present row-wise. The restoration
wrapper still evaluates only one base residual `g(w)` and one base Jacobian
`J_g(w)`, then reuses them with signs `+J_g` for `ub` and `-J_g` for `lb`.

The restoration wrappers are local. They do not alter the hard dynamics block;
they only modify how `c(w)` and `g(w)` enter the stage QP.

## 2. Equality Elastic Block

For one equality block, introduce local variables

$$
p \succ 0,\qquad n \succ 0,
$$

and the equality multiplier

$$
\lambda.
$$

The local equality relation is

$$
c(w)-p+n=0.
$$

The positivity duals for `p,n` are

$$
z_p \succ 0,\qquad z_n \succ 0.
$$

The local primal-dual residuals are

$$
\begin{aligned}
r_c   &= c(w)-p+n, \\
r_p   &= \rho_c-\lambda-z_p, \\
r_n   &= \rho_c+\lambda-z_n, \\
r_{s,p} &= z_p \circ p - \mu \mathbf{1}, \\
r_{s,n} &= z_n \circ n - \mu \mathbf{1}.
\end{aligned}
$$

These are exactly the residuals computed by the local model:

- `r_c` is the elastic equality residual
- `r_p,r_n` are the local stationarity rows
- `r_{s,p},r_{s,n}` are the local complementarity rows

### 2.1 Linearized KKT System

Let

$$
\delta c = J_c \delta w,
\qquad
J_c := \frac{\partial c}{\partial w}.
$$

The local Newton system is

$$
\begin{aligned}
\delta c - \delta p + \delta n &= -r_c, \\
-\delta\lambda - \delta z_p &= -r_p, \\
\delta\lambda - \delta z_n &= -r_n, \\
z_p \circ \delta p + p \circ \delta z_p &= -r_{s,p}, \\
z_n \circ \delta n + n \circ \delta z_n &= -r_{s,n}.
\end{aligned}
$$

The stage solver only needs the condensed relation between `\delta c` and
`\delta\lambda`. The local elastic variables are eliminated inside the wrapper.

### 2.2 Condensation

Introduce the diagonal matrices

$$
P := \operatorname{diag}(p),
\qquad
N := \operatorname{diag}(n),
\qquad
Z_p := \operatorname{diag}(z_p),
\qquad
Z_n := \operatorname{diag}(z_n).
$$

From the stationarity rows,

$$
\delta z_p = r_p - \delta\lambda,
\qquad
\delta z_n = r_n + \delta\lambda.
$$

Substitute these into the complementarity rows:

$$
\begin{aligned}
\delta p
&=
P Z_p^{-1}\delta\lambda
- P Z_p^{-1} r_p
- Z_p^{-1} r_{s,p}, \\
\delta n
&=
-N Z_n^{-1}\delta\lambda
- N Z_n^{-1} r_n
- Z_n^{-1} r_{s,n}.
\end{aligned}
$$

Now substitute `\delta p,\delta n` into

$$
\delta c-\delta p+\delta n=-r_c.
$$

This gives the condensed equality block

$$
-M_c \,\delta\lambda + \delta c = -\hat r_c,
$$

with

$$
M_c := P Z_p^{-1} + N Z_n^{-1},
$$

and

$$
\hat r_c
:=
r_c
+
P Z_p^{-1} r_p
+
Z_p^{-1} r_{s,p}
-
N Z_n^{-1} r_n
-
Z_n^{-1} r_{s,n}.
$$

Equivalently,

$$
\delta\lambda = M_c^{-1}(\delta c + \hat r_c).
$$

This is exactly the scalar/vector formula implemented in the wrapper through
the cached diagonal inverse

$$
M_c^{-1}
=
\left(\operatorname{diag}\!\left(\frac{p}{z_p}+\frac{n}{z_n}\right)\right)^{-1}.
$$

### 2.3 Newton-Step Recovery

Once the stage solve provides `\delta w`, the wrapper forms

$$
\delta c = J_c \delta w
$$

and recovers the eliminated local steps by back-substitution:

$$
\begin{aligned}
\delta\lambda &= M_c^{-1}(\delta c + \hat r_c), \\
\delta p &=
P Z_p^{-1}\delta\lambda
- P Z_p^{-1} r_p
- Z_p^{-1} r_{s,p}, \\
\delta n &=
-N Z_n^{-1}\delta\lambda
- N Z_n^{-1} r_n
- Z_n^{-1} r_{s,n}, \\
\delta z_p &= r_p - \delta\lambda, \\
\delta z_n &= r_n + \delta\lambda .
\end{aligned}
$$

The code stores:

- `d_lambda = \delta\lambda`
- `d_value[p] = \delta p`
- `d_value[n] = \delta n`
- `d_dual[p] = \delta z_p`
- `d_dual[n] = \delta z_n`

### 2.4 Contribution To The Stage QP

Because `\delta\lambda = M_c^{-1}(J_c \delta w + \hat r_c)`, the eliminated
equality block contributes

$$
J_c^T M_c^{-1}\hat r_c
$$

to the first-order correction and

$$
J_c^T M_c^{-1} J_c
$$

to the Hessian modification.

That is exactly why the wrapper propagates:

- `lag_jac_corr += J_c^T M_c^{-1}\hat r_c`
- `hessian_modification += J_c^T M_c^{-1}J_c`

## 3. Inequality Elastic Block

The implementation does **not** build one monolithic boxed inequality block.
Instead it applies the same local elastic algebra independently to each present
bound side and then aggregates the condensed contributions.

Let `s \in S` denote a present side:

- `s = ub` with residual `r_{ub}(w)=g(w)-ub`
- `s = lb` with residual `r_{lb}(w)=lb-g(w)`

Both sides reuse the same base residual `g(w)` and base Jacobian `J_g(w)`.
Their Jacobians differ only by sign:

$$
J_s = \sigma_s J_g,
\qquad
\sigma_{ub}=+1,
\qquad
\sigma_{lb}=-1.
$$

For one present side block, restoration introduces a positive slack `t_s` and
the elastic pair `p_s,n_s`:

$$
r_s(w)+t_s-p_s+n_s=0,
\qquad
t_s \succ 0,\quad p_s \succ 0,\quad n_s \succ 0.
$$

The local duals are

$$
\nu_{t,s} \succ 0,\qquad \nu_{p,s} \succ 0,\qquad \nu_{n,s} \succ 0.
$$

The local residuals are

$$
\begin{aligned}
r_{d,s}   &= r_s(w)+t_s-p_s+n_s, \\
r_{p,s}   &= \rho_d-\nu_{t,s}-\nu_{p,s}, \\
r_{n,s}   &= \rho_d+\nu_{t,s}-\nu_{n,s}, \\
r_{s,t,s} &= \nu_{t,s} \circ t_s - \mu \mathbf{1}, \\
r_{s,p,s} &= \nu_{p,s} \circ p_s - \mu \mathbf{1}, \\
r_{s,n,s} &= \nu_{n,s} \circ n_s - \mu \mathbf{1}.
\end{aligned}
$$

### 3.1 Linearized KKT System

Let

$$
\delta r_s = J_s \delta w.
$$

Then the local Newton system is

$$
\begin{aligned}
\delta r_s + \delta t_s - \delta p_s + \delta n_s &= -r_{d,s}, \\
-\delta\nu_{t,s} - \delta\nu_{p,s} &= -r_{p,s}, \\
\delta\nu_{t,s} - \delta\nu_{n,s} &= -r_{n,s}, \\
\nu_{t,s} \circ \delta t_s + t_s \circ \delta\nu_{t,s} &= -r_{s,t,s}, \\
\nu_{p,s} \circ \delta p_s + p_s \circ \delta\nu_{p,s} &= -r_{s,p,s}, \\
\nu_{n,s} \circ \delta n_s + n_s \circ \delta\nu_{n,s} &= -r_{s,n,s}.
\end{aligned}
$$

### 3.2 Condensation

Introduce

$$
T_s := \operatorname{diag}(t_s),
\qquad
P_s := \operatorname{diag}(p_s),
\qquad
N_s := \operatorname{diag}(n_s),
$$

$$
Z_{t,s} := \operatorname{diag}(\nu_{t,s}),
\qquad
Z_{p,s} := \operatorname{diag}(\nu_{p,s}),
\qquad
Z_{n,s} := \operatorname{diag}(\nu_{n,s}).
$$

From the local stationarity rows,

$$
\delta\nu_{p,s} = r_{p,s} - \delta\nu_{t,s},
\qquad
\delta\nu_{n,s} = r_{n,s} + \delta\nu_{t,s}.
$$

Substitute into the complementarity rows:

$$
\begin{aligned}
\delta t_s
&=
-T_s Z_{t,s}^{-1}\delta\nu_{t,s}
- Z_{t,s}^{-1} r_{s,t,s}, \\
\delta p_s
&=
P_s Z_{p,s}^{-1}\delta\nu_{t,s}
- P_s Z_{p,s}^{-1} r_{p,s}
- Z_{p,s}^{-1} r_{s,p,s}, \\
\delta n_s
&=
-N_s Z_{n,s}^{-1}\delta\nu_{t,s}
- N_s Z_{n,s}^{-1} r_{n,s}
- Z_{n,s}^{-1} r_{s,n,s}.
\end{aligned}
$$

Substitute these into

$$
\delta r_s+\delta t_s-\delta p_s+\delta n_s=-r_{d,s}.
$$

This gives the condensed side block

$$
-M_s\,\delta\nu_{t,s} + \delta r_s = -\hat r_s,
$$

with

$$
M_s := T_s Z_{t,s}^{-1} + P_s Z_{p,s}^{-1} + N_s Z_{n,s}^{-1},
$$

and

$$
\hat r_s
:=
r_{d,s}
- Z_{t,s}^{-1} r_{s,t,s}
+
P_s Z_{p,s}^{-1} r_{p,s}
+
Z_{p,s}^{-1} r_{s,p,s}
-
N_s Z_{n,s}^{-1} r_{n,s}
-
Z_{n,s}^{-1} r_{s,n,s}.
$$

Equivalently,

$$
\delta\nu_{t,s} = M_s^{-1}(\delta r_s + \hat r_s).
$$

So the implementation works with the diagonal inverse

$$
\operatorname{diag}\!\left(\frac{t_s}{\nu_{t,s}}+\frac{p_s}{\nu_{p,s}}+\frac{n_s}{\nu_{n,s}}\right)^{-1}.
$$

### 3.3 Newton-Step Recovery

Once the stage solve provides `\delta w`, the wrapper forms

$$
\delta r_s = J_s \delta w
$$

and recovers the eliminated local steps as

$$
\begin{aligned}
\delta\nu_{t,s} &= M_s^{-1}(\delta r_s + \hat r_s), \\
\delta t_s &=
-T_s Z_{t,s}^{-1}\delta\nu_{t,s}
- Z_{t,s}^{-1} r_{s,t,s}, \\
\delta p_s &=
P_s Z_{p,s}^{-1}\delta\nu_{t,s}
- P_s Z_{p,s}^{-1} r_{p,s}
- Z_{p,s}^{-1} r_{s,p,s}, \\
\delta n_s &=
-N_s Z_{n,s}^{-1}\delta\nu_{t,s}
- N_s Z_{n,s}^{-1} r_{n,s}
- Z_{n,s}^{-1} r_{s,n,s}, \\
\delta\nu_{p,s} &= r_{p,s} - \delta\nu_{t,s}, \\
\delta\nu_{n,s} &= r_{n,s} + \delta\nu_{t,s}.
\end{aligned}
$$

The code stores these in the side-local state:

- `side[s].d_dual[t] = \delta\nu_{t,s}`
- `side[s].d_value[t] = \delta t_s`
- `side[s].d_value[p] = \delta p_s`
- `side[s].d_value[n] = \delta n_s`
- `side[s].d_dual[p] = \delta\nu_{p,s}`
- `side[s].d_dual[n] = \delta\nu_{n,s}`

The outer multiplier exposed to the stage QP is rebuilt from the side
multipliers as

$$
\nu = \nu_{t,ub} - \nu_{t,lb},
\qquad
d\nu = \delta\nu_{t,ub} - \delta\nu_{t,lb},
$$

with missing sides treated as zero.

### 3.4 Contribution To The Stage QP

Because

$$
\delta\nu_{t,s} = M_s^{-1}(J_s \delta w + \hat r_s),
$$

one side block contributes

$$
J_s^T M_s^{-1}\hat r_s
$$

to the first-order correction and

$$
J_s^T M_s^{-1}J_s
$$

to the Hessian modification.

The wrapper then sums these condensed contributions over all present sides
`s \in S`. This is why a boxed inequality still evaluates `g` and `J_g` only
once even though it owns two local elastic side blocks.

## 4. Initialization Used By The Wrappers

The wrapper initialization matters because the condensed local solve is only
well behaved when the local elastic state starts reasonably centered.

### 4.1 Equality Elastic Initialization

For one scalar equality residual `c`, the equality wrapper initializes `p,n`
from the centered local system

$$
p-n=c,
\qquad
\frac{\mu}{p}+\frac{\mu}{n}=2\rho_c.
$$

Eliminating `p=c+n` gives the scalar quadratic solved in the code:

$$
\rho_c n^2 + \rho_c c\, n - \mu n - \frac{\mu c}{2} = 0,
$$

which is written in the implementation as

$$
n=\frac{\mu-\rho_c c+\sqrt{(\mu-\rho_c c)^2+2\rho_c\mu c}}{2\rho_c},
\qquad
p=c+n.
$$

Then

$$
z_p=\frac{\mu}{p},
\qquad
z_n=\frac{\mu}{n},
\qquad
\lambda=\rho_c-z_p.
$$

So the initializer satisfies

$$
p-n=c,
\qquad
z_p p = \mu,
\qquad
z_n n = \mu,
\qquad
z_p+z_n=2\rho_c.
$$

It does **not** use an `O(1)` lower bound on `p,n`; the code only applies a
tiny numerical floor.

### 4.2 Inequality Elastic Initialization

For one scalar side residual `r_s`, restoration reuses the outer boxed-IPM side
state as the initial

$$
t_s = t_{s,0},
\qquad
\nu_{t,s} = \nu_{t,s,0},
$$

and only initializes the extra elastic pair from

$$
c_s := r_s + t_{s,0},
\qquad
p_s - n_s = c_s.
$$

The implementation then uses the same centered elastic-pair formula as the
equality wrapper:

$$
n_s=\frac{\mu-\rho_d c_s+\sqrt{(\mu-\rho_d c_s)^2+2\rho_d\mu c_s}}{2\rho_d},
\qquad
p_s=c_s+n_s,
$$

and finally sets

$$
\nu_{p,s} = \frac{\mu}{p_s},
\qquad
\nu_{n,s} = \frac{\mu}{n_s}.
$$

So the initialized side block satisfies

$$
r_s+t_s-p_s+n_s=0,
\qquad
\nu_{p,s} p_s = \mu,
\qquad
\nu_{n,s} n_s = \mu.
$$

In addition, because `p_s,n_s` are initialized from the centered elastic-pair
formula, they satisfy

$$
\nu_{p,s} + \nu_{n,s} = 2\rho_d.
$$

What is **not** centered here is the full side stationarity with the inherited
outer multiplier `\nu_{t,s,0}`. In other words, the initializer does not impose

$$
\rho_d-\nu_{t,s}-\nu_{p,s}=0,
\qquad
\rho_d+\nu_{t,s}-\nu_{n,s}=0.
$$

The pair `t_s,\nu_{t,s}` is inherited from the outer IPM side rather than
re-centered from scratch, while only the added elastic pair `p_s,n_s` is
centered locally.

## 5. Search Objective

The restoration line search distinguishes:

- the original restoration objective
- the barrier-augmented search objective

The original restoration objective is

$$
\phi_R(w)
\;+\;
\rho_c \mathbf{1}^T(p_c+n_c)
\;+\;
\rho_d \sum_{s \in S}\mathbf{1}^T(p_s+n_s).
$$

The positivity barriers are not part of the original NLP. They appear only in
the primal-dual search model through the complementarity residuals and the
barrier parameter `\mu`.

## 6. Wrapper Lifecycle, Metrics, And Exit Check

### 6.1 Pre-Initialization Value Pass

On restoration entry, the overlay graph is evaluated once in value mode before
the soft-constraint initializer sizes and seeds the local elastic state.

So during that first value-only pass, the elastic wrappers simply forward the
source residuals:

- equality overlay: `v = c(w)`
- inequality overlay: `v = g(w)`

and only after `solver::ineq_soft::initialize(...)` do they switch to the full
elastic residuals `c-p+n` and the side-aware inequality elastic model.

### 6.2 Restoration Iteration Metric

During restoration iterations, the printed

$$
r(\mathrm{prim})
$$

is still the infinity norm of the **active restoration-overlay residual
vectors**. In particular it includes:

- hard residuals kept active in the overlay, such as dynamics
- equality elastic residuals `c-p+n`
- inequality elastic residual view built row-wise from the present side blocks

So the per-iteration restoration log is not the original-problem infeasibility
metric.

### 6.3 Outer Trial Check

The actual restoration success test is stricter:

1. accept a restoration trial on the overlay graph,
2. sync the trial primal state and hard duals back to the normal graph,
3. commit the restoration inequality bound state back to the outer IPM blocks,
3. evaluate the normal graph,
4. require both

$$
\text{outer\_filter\_accept}
$$

and

$$
\|r_{\mathrm{prim}}^{\mathrm{outer,trial}}\|_1
<
\|r_{\mathrm{prim}}^{\mathrm{outer,before}}\|_1,
$$

so restoration exits only after improving the **original** normal problem's
`L^1` primal residual, not merely after reducing the overlay metric.

For boxed outer inequalities, the extra bound-state commit in the outer trial
check matters. The outer normal problem must be evaluated with the restoration
side slacks and side multipliers copied back to the corresponding `ub/lb` IPM
side state, then the outer net multiplier and lifted residual compatibility
views rebuilt from those side states.

### 6.4 Successful Exit Cleanup

On successful exit, the implementation also commits restoration inequality
state back to the normal IPM constraints:

- one-sided inequalities copy the single side block `t / \nu_t` back directly
- boxed inequalities copy each present side block `t_s / \nu_{t,s}` back to the
  matching outer `ub/lb` side state and then rebuild the outer net-dual and
  lifted-residual compatibility views

The normal problem is then re-evaluated in value/derivative mode. Equality-side
dual blocks are reset afterward according to the configured threshold rule.

### 6.5 Return Semantics

If restoration exhausts its budget without satisfying the exit test, the solver
returns

$$
\texttt{restoration\_reached\_max\_iter}.
$$

If restoration succeeds, it does **not** mark the whole SQP solve as
converged. Instead it returns control to the normal phase with the updated
outer state, and the main SQP loop continues from the post-restoration
iteration count.

## 7. Summary

The restoration wrappers use a fully local elimination:

- the stage solver computes `\delta w`
- the local wrapper computes `\delta c` or `\delta g`
- the condensed diagonal system recovers the local multiplier step
- the remaining elastic primal/dual steps are recovered by back-substitution

The mathematically relevant condensed operators are:

$$
M_c = \operatorname{diag}\!\left(\frac{p}{z_p}+\frac{n}{z_n}\right),
\qquad
M_s = \operatorname{diag}\!\left(\frac{t_s}{\nu_{t,s}}+\frac{p_s}{\nu_{p,s}}+\frac{n_s}{\nu_{n,s}}\right).
$$

Those are the exact local Schur complements used by the restoration overlay,
with one `M_s` per present inequality side.
