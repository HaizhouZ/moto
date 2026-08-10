# Lifted Weighted-Gram Identity

Let

$$
R_l = \begin{bmatrix} Z_l & -l_y \end{bmatrix},
$$

where $Z_l \in \mathbb{R}^{n_l\times n_u}$ and
$l_y \in \mathbb{R}^{n_l\times n_x}$. For symmetric
$Q_{ll}\in\mathbb{R}^{n_l\times n_l}$,

$$
R_l^TQ_{ll}R_l =
\begin{bmatrix}
Z_l^TQ_{ll}Z_l & -Z_l^TQ_{ll}l_y \\
-l_y^TQ_{ll}Z_l & l_y^TQ_{ll}l_y
\end{bmatrix}.
$$

Thus one symmetric product contains the lifted `Q_zz`, input-state, and
`V_xx` contributions. This is an evaluation-order identity only; it does not
change the SQP model, regularization, derivatives, or nullspace definition.
