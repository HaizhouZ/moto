# Lifted Direction Recovery

For the projected linearized lifted relation

$$
dl = -l_0-l_xdx-l_udu,
$$

let the recovered input policy be

$$
du=d_u.k+d_u.Kdx.
$$

Materializing separate lifted policy terms would give

$$
d_l.k=-l_0-l_ud_u.k,
$$

and

$$
d_l.K=-l_x-l_ud_u.K.
$$

Their application is algebraically identical to direct recovery:

$$
d_l.k+d_l.Kdx=-l_0-l_xdx-l_u\left(d_u.k+d_u.Kdx\right)
              =-l_0-l_xdx-l_udu.
$$

For a residual correction, the constant residual term is absent:

$$
dl_{\mathrm{corr}}=-l_xdx_{\mathrm{corr}}-l_udu_{\mathrm{corr}}.
$$
