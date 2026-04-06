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

$$
g(w)\le 0
$$

the inequality constraints that are made elastic in restoration.

The restoration problem is

$$
\begin{aligned}
\min_{w,p_c,n_c,p_d,n_d,t}\quad &
\phi_R(w)
\;+\;
\rho_c \mathbf{1}^T(p_c+n_c)
\;+\;
\rho_d \mathbf{1}^T(p_d+n_d) \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p_c+n_c=0, \\
&
g(w)+t-p_d+n_d=0, \\
&
t \succ 0,\quad p_c \succ 0,\quad n_c \succ 0,\quad p_d \succ 0,\quad n_d \succ 0.
\end{aligned}
$$

Here:

- `\phi_R(w)` is the proximal restoration cost
- `\rho_c > 0` is the equality elastic weight
- `\rho_d > 0` is the inequality elastic weight
- `t` is the explicit positive slack for the inequality block

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

For one inequality block, restoration introduces a positive slack `t` and the
elastic pair `p,n`:

$$
g(w)+t-p+n=0,
\qquad
t \succ 0,\quad p \succ 0,\quad n \succ 0.
$$

The local duals are

$$
\nu_t \succ 0,\qquad \nu_p \succ 0,\qquad \nu_n \succ 0.
$$

The reduced inequality block keeps `\nu_t` as the active multiplier carried by
the stage solver.

The local residuals are

$$
\begin{aligned}
r_d   &= g(w)+t-p+n, \\
r_p   &= \rho_d-\nu_t-\nu_p, \\
r_n   &= \rho_d+\nu_t-\nu_n, \\
r_{s,t} &= \nu_t \circ t - \mu \mathbf{1}, \\
r_{s,p} &= \nu_p \circ p - \mu \mathbf{1}, \\
r_{s,n} &= \nu_n \circ n - \mu \mathbf{1}.
\end{aligned}
$$

There is no extra local `\lambda_d` here. The remaining local multiplier is
`\nu_t`.

### 3.1 Linearized KKT System

Let

$$
\delta g = J_d \delta w,
\qquad
J_d := \frac{\partial g}{\partial w}.
$$

The local Newton system is

$$
\begin{aligned}
\delta g + \delta t - \delta p + \delta n &= -r_d, \\
-\delta\nu_t - \delta\nu_p &= -r_p, \\
\delta\nu_t - \delta\nu_n &= -r_n, \\
\nu_t \circ \delta t + t \circ \delta\nu_t &= -r_{s,t}, \\
\nu_p \circ \delta p + p \circ \delta\nu_p &= -r_{s,p}, \\
\nu_n \circ \delta n + n \circ \delta\nu_n &= -r_{s,n}.
\end{aligned}
$$

### 3.2 Condensation

Introduce

$$
T := \operatorname{diag}(t),
\qquad
P := \operatorname{diag}(p),
\qquad
N := \operatorname{diag}(n),
$$

$$
Z_t := \operatorname{diag}(\nu_t),
\qquad
Z_p := \operatorname{diag}(\nu_p),
\qquad
Z_n := \operatorname{diag}(\nu_n).
$$

From the local stationarity rows,

$$
\delta\nu_p = r_p - \delta\nu_t,
\qquad
\delta\nu_n = r_n + \delta\nu_t.
$$

Substitute into the complementarity rows:

$$
\begin{aligned}
\delta t
&=
-T Z_t^{-1}\delta\nu_t
- Z_t^{-1} r_{s,t}, \\
\delta p
&=
P Z_p^{-1}\delta\nu_t
- P Z_p^{-1} r_p
- Z_p^{-1} r_{s,p}, \\
\delta n
&=
-N Z_n^{-1}\delta\nu_t
- N Z_n^{-1} r_n
- Z_n^{-1} r_{s,n}.
\end{aligned}
$$

Substitute these into

$$
\delta g+\delta t-\delta p+\delta n=-r_d.
$$

This gives the condensed inequality block

$$
-M_d\,\delta\nu_t + \delta g = -\hat r_d,
$$

with

$$
M_d := T Z_t^{-1} + P Z_p^{-1} + N Z_n^{-1},
$$

and

$$
\hat r_d
:=
r_d
- Z_t^{-1} r_{s,t}
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
\delta\nu_t = M_d^{-1}(\delta g + \hat r_d).
$$

Again, this matches the implementation exactly: the wrapper computes the
diagonal inverse of

$$
\operatorname{diag}\!\left(\frac{t}{\nu_t}+\frac{p}{\nu_p}+\frac{n}{\nu_n}\right).
$$

### 3.3 Newton-Step Recovery

Once the stage solve provides `\delta w`, the wrapper forms

$$
\delta g = J_d \delta w
$$

and recovers the eliminated local steps as

$$
\begin{aligned}
\delta\nu_t &= M_d^{-1}(\delta g + \hat r_d), \\
\delta t &=
-T Z_t^{-1}\delta\nu_t
- Z_t^{-1} r_{s,t}, \\
\delta p &=
P Z_p^{-1}\delta\nu_t
- P Z_p^{-1} r_p
- Z_p^{-1} r_{s,p}, \\
\delta n &=
-N Z_n^{-1}\delta\nu_t
- N Z_n^{-1} r_n
- Z_n^{-1} r_{s,n}, \\
\delta\nu_p &= r_p - \delta\nu_t, \\
\delta\nu_n &= r_n + \delta\nu_t .
\end{aligned}
$$

The code stores:

- `d_dual[t] = \delta\nu_t`
- `d_value[t] = \delta t`
- `d_value[p] = \delta p`
- `d_value[n] = \delta n`
- `d_dual[p] = \delta\nu_p`
- `d_dual[n] = \delta\nu_n`

The reduced multiplier seen by the stage QP is exactly `\nu_t`, so the wrapper
exports

$$
d\_multiplier = \delta\nu_t.
$$

### 3.4 Contribution To The Stage QP

Because

$$
\delta\nu_t = M_d^{-1}(J_d \delta w + \hat r_d),
$$

the eliminated inequality block contributes

$$
J_d^T M_d^{-1}\hat r_d
$$

to the first-order correction and

$$
J_d^T M_d^{-1}J_d
$$

to the Hessian modification.

This is the same condensed structure as the equality block; only the local
diagonal `M_d` and the residual `\hat r_d` differ.

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

For one scalar inequality residual `g`, the active restoration multiplier is
`\nu_t`, and the wrapper initializes

$$
\nu_p = \rho_d-\nu_t,
\qquad
\nu_n = \rho_d+\nu_t,
\qquad
p = \frac{\mu}{\nu_p},
\qquad
n = \frac{\mu}{\nu_n},
\qquad
t = -g + p - n.
$$

This centers the local inequality-elastic block exactly:

$$
g+t-p+n=0,
\qquad
\rho_d-\nu_t-\nu_p=0,
\qquad
\rho_d+\nu_t-\nu_n=0,
$$

and

$$
\nu_t t \approx \mu,
\qquad
\nu_p p = \mu,
\qquad
\nu_n n = \mu.
$$

In the actual implementation, `\nu_t` is clamped strictly inside `(0,\rho_d)`
before these formulas are applied, so the centered initialization never
degenerates to `\nu_p \approx 0`.

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
\rho_d \mathbf{1}^T(p_d+n_d).
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
elastic residuals `c-p+n` and `g+t-p+n`.

### 6.2 Restoration Iteration Metric

During restoration iterations, the printed

$$
r(\mathrm{prim})
$$

is still the infinity norm of the **active restoration-overlay residual
vectors**. In particular it includes:

- hard residuals kept active in the overlay, such as dynamics
- equality elastic residuals `c-p+n`
- inequality elastic residuals `g+t-p+n`

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

The extra bound-state commit in the outer trial check matters. The outer normal
problem must be evaluated with

- outer slack `s \leftarrow t`
- outer multiplier `z \leftarrow \nu_t`

for the original IPM inequality residuals to be consistent with the candidate
restoration point.

### 6.4 Successful Exit Cleanup

On successful exit, the implementation also commits restoration inequality
bound state back to the normal IPM constraints:

- outer slack `s \leftarrow t`
- outer multiplier `z \leftarrow \nu_t`

and then re-evaluates the normal problem values and derivatives. Equality-side
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
M_d = \operatorname{diag}\!\left(\frac{t}{\nu_t}+\frac{p}{\nu_p}+\frac{n}{\nu_n}\right).
$$

Those are the exact local Schur complements used by the restoration overlay.
