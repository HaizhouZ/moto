from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import moto
from moto import (
    approx_order as approx_order,
    casadi_manifold as casadi_manifold,
    constr as constr,
    cost as cost,
    dense_dynamics as dense_dynamics,
    endpoint as endpoint,
    expr as expr,
    field as field,
    func as func,
    ineq as ineq,
    lifted as lifted,
    pmm_constr as pmm_constr,
    semi_implicit_euler as semi_implicit_euler,
    sqp as ns_sqp_impl,
    sqp as sqp,
    stage_ocp as stage_ocp,
    sym as sym
)


field___x: moto.field = moto.field.field___x

field___u: moto.field = moto.field.field___u

field___y: moto.field = moto.field.field___y

field___l: moto.field = moto.field.field___l

field___s: moto.field = moto.field.field___s

field___p: moto.field = moto.field.field___p

field___dyn: moto.field = moto.field.field___dyn

field___lift: moto.field = moto.field.field___lift

field___eq_x: moto.field = moto.field.field___eq_x

field___eq_xu: moto.field = moto.field.field___eq_xu

field___ineq_x: moto.field = moto.field.field___ineq_x

field___ineq_xu: moto.field = moto.field.field___ineq_xu

field___eq_x_soft: moto.field = moto.field.field___eq_x_soft

field___eq_xu_soft: moto.field = moto.field.field___eq_xu_soft

field___cost: moto.field = moto.field.field___cost

field___pre_comp: moto.field = moto.field.field___pre_comp

field___post_comp: moto.field = moto.field.field___post_comp

field___usr_func: moto.field = moto.field.field___usr_func

field___func_stack: moto.field = moto.field.field___func_stack

field___usr_var: moto.field = moto.field.field___usr_var

field_NUM: moto.field = moto.field.field_NUM

field___undefined: moto.field = moto.field.field___undefined

class ocp_base:
    @overload
    def add(self, exprs: Sequence[ moto.expr  | moto.var]) -> None:
        """Add a list of expressions to the OCP problem"""

    @overload
    def add(self, ex: moto.expr) -> None:
        """Add an expression to the OCP problem"""

    def dim(self, field: moto.field) -> int:
        """Get the dimension of the field"""

    def wait_until_ready(self) -> None:
        """Wait until all expressions in the OCP problem are ready"""

    def is_active(self, arg: moto.expr) -> bool:
        """Check if a given argument is active in the OCP problem"""

    def print_summary(self) -> None:
        """Print a summary of the OCP problem"""

class ocp(ocp_base):
    pass

class sym_data:
    def __getitem__(self, arg: moto.var, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    def __setitem__(self, arg0: moto.var, arg1: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float, /) -> None: ...

class node_data:
    @property
    def prob(self) -> ocp: ...

    @property
    def value(self) -> sym_data: ...

    def data(self, function: moto.func) -> func_approx_data: ...

class func_approx_data:
    def __getitem__(self, arg: moto.var, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @property
    def v(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """Value vector reference"""

    def jac(self, arg: moto.var, /) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
        """Get the jacobian reference for the input variable"""

    def set_jac(self, arg0: moto.var, arg1: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], /) -> None: ...

approx_order_none: moto.approx_order = moto.approx_order.approx_order_none

approx_order_zero: moto.approx_order = moto.approx_order.approx_order_zero

approx_order_first: moto.approx_order = moto.approx_order.approx_order_first

approx_order_second: moto.approx_order = moto.approx_order.approx_order_second

class linesearch_config:
    @property
    def update_alpha_dual(self) -> bool:
        """Whether to update the dual step size during line search"""

    @update_alpha_dual.setter
    def update_alpha_dual(self, arg: bool, /) -> None: ...

    @property
    def eq_dual_alpha_source(self) -> linesearch_config.dual_alpha_source:
        """Source for dual step size for equality constraints"""

    @eq_dual_alpha_source.setter
    def eq_dual_alpha_source(self, arg: linesearch_config.dual_alpha_source, /) -> None: ...

    @property
    def ineq_dual_alpha_source(self) -> linesearch_config.dual_alpha_source:
        """Source for dual step size for inequality constraints"""

    @ineq_dual_alpha_source.setter
    def ineq_dual_alpha_source(self, arg: linesearch_config.dual_alpha_source, /) -> None: ...

    class dual_alpha_source(enum.Enum):
        dual_alpha_source_primal = 0

        dual_alpha_source_dual = 1

    dual_alpha_source_primal: dual_alpha_source = dual_alpha_source.dual_alpha_source_primal

    dual_alpha_source_dual: dual_alpha_source = dual_alpha_source.dual_alpha_source_dual
