from collections.abc import Callable, Iterable, Iterator, Sequence
import enum
from typing import Annotated, overload

import casadi
import casadi.casadi
import numpy
from numpy.typing import NDArray

from . import definition as definition, moto_pywrap as moto_pywrap


class approx_order(enum.Enum):
    """Public API type ``moto.approx_order``."""

    approx_order_none = 0

    approx_order_zero = 1

    approx_order_first = 2

    approx_order_second = 3

class casadi_manifold(sym):
    """Public API type ``moto.casadi_manifold``."""

    @staticmethod
    def create(name: str, q: casadi.SX, dq: casadi.SX, integrated: casadi.SX, other: casadi.SX, difference: casadi.SX, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> tuple[var, var]: ...

class constr(func):
    """Public API type ``moto.constr``."""

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr: ...

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0, field: field = field.field___undefined) -> constr: ...

    def cast_soft(self, type_name: str = 'pmm_constr') -> constr: ...

class cost(func):
    """Public API type ``moto.cost``."""

    @staticmethod
    def from_vector(name: str, value: casadi.SX, weight: object = 1.0, reference: object = 0.0) -> cost: ...

    @staticmethod
    def from_scalar(name: str, value: casadi.SX, weight: object = 1.0, reference: object = 0.0) -> cost: ...

    @property
    def weight(self) -> var: ...

    @property
    def reference(self) -> var: ...

class dense_dynamics(lifted):
    """Public API type ``moto.dense_dynamics``."""

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first) -> dense_dynamics: ...

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0) -> dense_dynamics: ...

    def mark_shared_inputs(self, shared_inputs: Sequence[var]) -> None: ...

class endpoint:
    """Public API type ``moto.endpoint``."""

    @overload
    def add(self, exprs: Sequence[ expr  | var]) -> None:
        """Add node-local expressions"""

    @overload
    def add(self, ex: expr) -> None:
        """Add a node-local expression"""

class expr:
    """Public API type ``moto.expr``."""

    def __bool__(self) -> bool: ...

    def __str__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def field(self) -> field: ...

    @property
    def dim(self) -> int: ...

    @property
    def uid(self) -> int: ...

    def finalize(self, block_until_ready: bool = True) -> bool: ...

    @property
    def tdim(self) -> int: ...

class field(enum.Enum):
    """Public API type ``moto.field``."""

    field___x = 0

    field___u = 1

    field___y = 2

    field___l = 3

    field___s = 4

    field___p = 5

    field___dyn = 6

    field___lift = 7

    field___eq_x = 8

    field___eq_xu = 9

    field___ineq_x = 10

    field___ineq_xu = 11

    field___eq_x_soft = 12

    field___eq_xu_soft = 13

    field___cost = 14

    field___pre_comp = 15

    field___post_comp = 16

    field___usr_func = 17

    field___func_stack = 18

    field___usr_var = 19

    field_NUM = 20

    field___undefined = 21

class func(expr):
    """Public API type ``moto.func``."""

    @property
    def in_args(self) -> list[var]: ...

    @property
    def value(self) -> Callable[[moto_pywrap.func_approx_data], None]: ...

    @value.setter
    def value(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def jacobian(self) -> Callable[[moto_pywrap.func_approx_data], None]: ...

    @jacobian.setter
    def jacobian(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def hessian(self) -> Callable[[moto_pywrap.func_approx_data], None]: ...

    @hessian.setter
    def hessian(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def order(self) -> approx_order: ...

    def __str__(self) -> str: ...

    def enable_if_all(self, args: Sequence[ expr  | var]) -> None: ...

    def disable_if_any(self, args: Sequence[ expr  | var]) -> None: ...

    def enable_if_any(self, args: Sequence[ expr  | var]) -> None: ...

    def add_argument(self, arg: var) -> None: ...

    def add_arguments(self, arg: Sequence[var], /) -> None: ...

    def set_analytic_jacobian(self, arg: var, jacobian: casadi.SX) -> None: ...

    def set_analytic_hessian(self, arg0: var, arg1: var, hessian: casadi.SX) -> None: ...

    def remap_arguments(self, remap: Sequence[tuple[var, var]]) -> func:
        """Create a fresh remapped function"""

    def reuse_remap(self, remap: Sequence[tuple[var, var]]) -> func:
        """Reuse the cached function for this remap"""

class ineq(constr):
    """Public API type ``moto.ineq``."""

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr: ...

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0, field: field = field.field___undefined) -> constr: ...

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, lb: object, ub: object, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr: ...

    @staticmethod
    def bounds(name: str, value: var, lb: object, ub: object, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr:
        """Create scalar or vector bounds directly on a variable"""

class lifted(constr):
    """Public API type ``moto.lifted``."""

    class partition:
        """Public API type ``moto.lifted.partition``."""

        @property
        def name(self) -> str: ...

        @property
        def field(self) -> field: ...

        @property
        def offset(self) -> int: ...

        @property
        def size(self) -> int: ...

    class block:
        """Public API type ``moto.lifted.block``."""

        @property
        def mx(self) -> casadi.MX: ...

        def param(self, default_val: object | None = None, name: str = '', dim: int = 1) -> var: ...

        def rows(self, begin: int, end: int) -> lifted.block: ...

        def add_diag(self, parameter: object, name: str = '', dim: int = 1) -> casadi.MX: ...

    class factor:
        """Public API type ``moto.lifted.factor``."""

        @property
        def matrix(self) -> casadi.MX: ...

        def solve(self, rhs: casadi.MX) -> casadi.MX: ...

    class system:
        """Public API type ``moto.lifted.system``."""

        @property
        def dyn_residual(self) -> casadi.MX: ...

        @property
        def lift_residual(self) -> casadi.MX: ...

        @property
        def action_rhs(self) -> casadi.MX: ...

        @overload
        def jac(self, equation: constr, variable: var) -> lifted.block: ...

        @overload
        def jac(self, equation: casadi.SX, variable: var) -> lifted.block: ...

        @overload
        def residual(self, equation: constr) -> casadi.MX: ...

        @overload
        def residual(self, equation: str) -> casadi.MX: ...

        @property
        def equations(self) -> list[lifted.partition]: ...

        @property
        def variables(self) -> list[lifted.partition]: ...

        def h_l(self) -> casadi.MX: ...

        def h_x(self) -> casadi.MX: ...

        def h_u(self) -> casadi.MX: ...

        def h(self) -> casadi.MX: ...

        def solve(self, matrix: casadi.MX, spd: bool = False) -> lifted.factor: ...

        def eliminate(self, solve: Callable[[casadi.MX], casadi.MX], intermediates: Sequence[lifted.intermediate] = []) -> lifted.elimination: ...

    class intermediate:
        """Public API type ``moto.lifted.intermediate``."""

        def __init__(self, name: str, value: casadi.MX) -> None: ...

        @property
        def name(self) -> str: ...

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def value(self) -> casadi.MX: ...

        @value.setter
        def value(self, arg: casadi.MX, /) -> None: ...

    class elimination:
        """Public API type ``moto.lifted.elimination``."""

        def __init__(self, response_x: casadi.MX, response_u: casadi.MX, response_residual: casadi.MX, intermediates: Sequence[lifted.intermediate], response_action: casadi.MX) -> None: ...

        @property
        def response_x(self) -> casadi.MX: ...

        @response_x.setter
        def response_x(self, arg: casadi.MX, /) -> None: ...

        @property
        def response_u(self) -> casadi.MX: ...

        @response_u.setter
        def response_u(self, arg: casadi.MX, /) -> None: ...

        @property
        def response_residual(self) -> casadi.MX: ...

        @response_residual.setter
        def response_residual(self, arg: casadi.MX, /) -> None: ...

        @property
        def intermediates(self) -> list[lifted.intermediate]: ...

        @intermediates.setter
        def intermediates(self, arg: Sequence[lifted.intermediate], /) -> None: ...

        @property
        def response_action(self) -> casadi.MX: ...

        @response_action.setter
        def response_action(self, arg: casadi.MX, /) -> None: ...

    @staticmethod
    def create(name: str, out: casadi.SX, lifted_args: Sequence[var], order: approx_order = approx_order.approx_order_second) -> lifted: ...

    def with_elimination_graph(self, builder: Callable[[system], elimination], subconstraints: Sequence[constr] = []) -> lifted: ...

    def set_elimination_graph(self, builder: Callable[[system], elimination]) -> lifted: ...

    def add_subconstraint(self, constraint: constr) -> None: ...

    @property
    def subconstraints(self) -> list[constr]: ...

    @property
    def lifted_args(self) -> list[var]: ...

    @property
    def elimination_parameters(self) -> list[var]: ...

class pmm_constr(constr):
    """Public API type ``moto.pmm_constr``."""

    @property
    def rho(self) -> float:
        """Dual penalty weight for the proximal multiplier method"""

    @rho.setter
    def rho(self, arg: float, /) -> None: ...

class semi_implicit_euler(lifted):
    """Public API type ``moto.semi_implicit_euler``."""

    class state(enum.Enum):
        """Public API type ``moto.semi_implicit_euler.state``."""

        pos = 0

        pos_vel = 1

    @staticmethod
    def create(name: str, out: casadi.SX, state: state = state.pos_vel, order: approx_order = approx_order.approx_order_first) -> semi_implicit_euler: ...

    def mark_shared_inputs(self, shared_inputs: Sequence[var]) -> None: ...

class sqp:
    """Public API type ``moto.sqp``."""

    def __init__(self, n_job: int = 4) -> None:
        """Constructor for the SQP solver with a specified number of jobs"""

    class stage_list:
        """Public API type ``moto.sqp.stage_list``."""

        @overload
        def __init__(self) -> None:
            """Default constructor"""

        @overload
        def __init__(self, arg: sqp.stage_list) -> None:
            """Copy constructor"""

        @overload
        def __init__(self, arg: Iterable[stage_ocp], /) -> None:
            """Construct from an iterable object"""

        def __len__(self) -> int: ...

        def __bool__(self) -> bool:
            """Check whether the vector is nonempty"""

        def __repr__(self) -> str: ...

        def __iter__(self) -> Iterator[stage_ocp]: ...

        @overload
        def __getitem__(self, arg: int, /) -> stage_ocp: ...

        @overload
        def __getitem__(self, arg: slice, /) -> sqp.stage_list: ...

        def clear(self) -> None:
            """Remove all items from list."""

        def append(self, arg: stage_ocp, /) -> None:
            """Append `arg` to the end of the list."""

        def insert(self, arg0: int, arg1: stage_ocp, /) -> None:
            """Insert object `arg1` before index `arg0`."""

        def pop(self, index: int = -1) -> stage_ocp:
            """Remove and return item at `index` (default last)."""

        def extend(self, arg: sqp.stage_list, /) -> None:
            """Extend `self` by appending elements from `arg`."""

        @overload
        def __setitem__(self, arg0: int, arg1: stage_ocp, /) -> None: ...

        @overload
        def __setitem__(self, arg0: slice, arg1: sqp.stage_list, /) -> None: ...

        @overload
        def __delitem__(self, arg: int, /) -> None: ...

        @overload
        def __delitem__(self, arg: slice, /) -> None: ...

        def __eq__(self, arg: object, /) -> bool: ...

        def __ne__(self, arg: object, /) -> bool: ...

        @overload
        def __contains__(self, arg: stage_ocp, /) -> bool: ...

        @overload
        def __contains__(self, arg: object, /) -> bool: ...

        def count(self, arg: stage_ocp, /) -> int:
            """Return number of occurrences of `arg`."""

        def remove(self, arg: stage_ocp, /) -> None:
            """Remove first occurrence of `arg`."""

    @property
    def st(self) -> endpoint:
        """Initial graph boundary"""

    @property
    def ed(self) -> endpoint:
        """Graph terminal boundary"""

    @property
    def stages(self) -> sqp.stage_list:
        """Mutable ordered graph-owned stage vector"""

    def update(self, n_iter: int = 1, verbose: bool = True, profile: bool = False) -> result_type:
        """Update the SQP solver for a given number of iterations"""

    def get_profile_report(self) -> profile_report:
        """Get the latest SQP wall-clock profile report"""

    @property
    def n_job(self) -> int:
        """Effective maximum number of SQP worker threads"""

    @property
    def settings(self) -> sqp.settings_type:
        """Get the settings of the SQP solver"""

    class ipm_config:
        """Public API type ``moto.sqp.ipm_config``."""

        @property
        def mu0(self) -> float:
            """Initial barrier parameter for the IPM solver"""

        @mu0.setter
        def mu0(self, arg: float, /) -> None: ...

        @property
        def warm_start(self) -> bool:
            """Whether to warm start the IPM solver"""

        @warm_start.setter
        def warm_start(self, arg: bool, /) -> None: ...

        @property
        def mu_method(self) -> sqp.adaptive_mu_t:
            """Adaptive mu method for the IPM solver"""

        @mu_method.setter
        def mu_method(self, arg: sqp.adaptive_mu_t, /) -> None: ...

        @property
        def mu_monotone_fraction_threshold(self) -> float:
            """
            Threshold for monotone decrease of mu (smaller is more likely to use monotone decrease)
            """

        @mu_monotone_fraction_threshold.setter
        def mu_monotone_fraction_threshold(self, arg: float, /) -> None: ...

        @property
        def mu_monotone_factor(self) -> float:
            """Factor for monotone decrease of mu (smaller -> faster decrease)"""

        @mu_monotone_factor.setter
        def mu_monotone_factor(self, arg: float, /) -> None: ...

        @property
        def globalization(self) -> bool:
            """Whether to use globalization in the IPM solver"""

        @globalization.setter
        def globalization(self, arg: bool, /) -> None: ...

    class iterative_refinement_setting:
        """Public API type ``moto.sqp.iterative_refinement_setting``."""

        @property
        def enabled(self) -> bool:
            """Whether to use iterative refinement"""

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def max_iters(self) -> int:
            """Maximum number of iterative refinement iterations"""

        @max_iters.setter
        def max_iters(self, arg: int, /) -> None: ...

        @property
        def prim_res_tol(self) -> float:
            """Primal residual tolerance for iterative refinement"""

        @prim_res_tol.setter
        def prim_res_tol(self, arg: float, /) -> None: ...

        @property
        def dual_res_tol(self) -> float:
            """Dual residual tolerance for iterative refinement"""

        @dual_res_tol.setter
        def dual_res_tol(self, arg: float, /) -> None: ...

    class restoration_settings:
        """Public API type ``moto.sqp.restoration_settings``."""

        @property
        def enabled(self) -> bool:
            """Whether restoration is enabled"""

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def max_iter(self) -> int:
            """Maximum number of restoration iterations"""

        @max_iter.setter
        def max_iter(self, arg: int, /) -> None: ...

        @property
        def rho_u(self) -> float:
            """Restoration proximal weight on u"""

        @rho_u.setter
        def rho_u(self, arg: float, /) -> None: ...

        @property
        def rho_y(self) -> float:
            """Restoration proximal weight on y"""

        @rho_y.setter
        def rho_y(self, arg: float, /) -> None: ...

        @property
        def rho_eq(self) -> float:
            """Elastic penalty weight for restoration equalities"""

        @rho_eq.setter
        def rho_eq(self, arg: float, /) -> None: ...

        @property
        def rho_ineq(self) -> float:
            """Elastic penalty weight for restoration inequalities"""

        @rho_ineq.setter
        def rho_ineq(self, arg: float, /) -> None: ...

        @property
        def restoration_improvement_frac(self) -> float:
            """
            Required fraction of primal infeasibility improvement to accept restoration exit
            """

        @restoration_improvement_frac.setter
        def restoration_improvement_frac(self, arg: float, /) -> None: ...

        @property
        def alpha_min_factor(self) -> float:
            """Tiny-step trigger factor used before entering restoration"""

        @alpha_min_factor.setter
        def alpha_min_factor(self, arg: float, /) -> None: ...

        @property
        def bound_mult_reset_threshold(self) -> float:
            """Reset copied-back bound multipliers when they exceed this threshold"""

        @bound_mult_reset_threshold.setter
        def bound_mult_reset_threshold(self, arg: float, /) -> None: ...

        @property
        def constr_mult_reset_threshold(self) -> float:
            """Reset copied-back equality multipliers when they exceed this threshold"""

        @constr_mult_reset_threshold.setter
        def constr_mult_reset_threshold(self, arg: float, /) -> None: ...

    class equality_multiplier_init_settings:
        """Public API type ``moto.sqp.equality_multiplier_init_settings``."""

        @property
        def enabled(self) -> bool:
            """Whether equality-type multipliers are rebuilt during initialization"""

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def rebuild_after_restoration_exit(self) -> bool:
            """
            Whether to rebuild equality-type multipliers after restoration exits successfully
            """

        @rebuild_after_restoration_exit.setter
        def rebuild_after_restoration_exit(self, arg: bool, /) -> None: ...

        @property
        def rho_eq(self) -> float:
            """
            PMM penalty used for equality-type constraints in the equality-init overlay
            """

        @rho_eq.setter
        def rho_eq(self, arg: float, /) -> None: ...

        @property
        def rf(self) -> sqp.iterative_refinement_setting:
            """
            Dedicated iterative-refinement settings used only during equality-multiplier initialization
            """

        @rf.setter
        def rf(self, arg: sqp.iterative_refinement_setting, /) -> None: ...

    class linesearch_setting(moto_pywrap.linesearch_config):
        """Public API type ``moto.sqp.linesearch_setting``."""

        @property
        def enabled(self) -> bool:
            """Whether to use line search"""

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def max_steps(self) -> int:
            """Maximum number of line search steps"""

        @max_steps.setter
        def max_steps(self, arg: int, /) -> None: ...

        @property
        def failure_strategy(self) -> sqp.linesearch_setting.failure_backup_strategy:
            """Line search failure backup strategy"""

        @failure_strategy.setter
        def failure_strategy(self, arg: sqp.linesearch_setting.failure_backup_strategy, /) -> None: ...

        @property
        def on_failure(self) -> sqp.linesearch_setting.on_failure_action:
            """Action to take after line search exhausts max_steps"""

        @on_failure.setter
        def on_failure(self, arg: sqp.linesearch_setting.on_failure_action, /) -> None: ...

        @property
        def method(self) -> sqp.search_method:
            """Line search method: filter (default) or merit_backtracking"""

        @method.setter
        def method(self, arg: sqp.search_method, /) -> None: ...

        @property
        def primal_gamma(self) -> float:
            """Primal improvement requirement for the filter (higher is stricter)"""

        @primal_gamma.setter
        def primal_gamma(self, arg: float, /) -> None: ...

        @property
        def dual_gamma(self) -> float:
            """Objective improvement requirement for the filter (higher is stricter)"""

        @dual_gamma.setter
        def dual_gamma(self, arg: float, /) -> None: ...

        @property
        def constr_vio_min_frac(self) -> float:
            """
            Threshold for switching condition (fraction of initial primal residual)
            """

        @constr_vio_min_frac.setter
        def constr_vio_min_frac(self, arg: float, /) -> None: ...

        @property
        def armijo_dec_frac(self) -> float:
            """
            Sufficient decrease tolerance (eta in Armijo condition), smaller -> more strict decrease requirement
            """

        @armijo_dec_frac.setter
        def armijo_dec_frac(self, arg: float, /) -> None: ...

        @property
        def s_phi(self) -> float:
            """
            IPOPT switching condition exponent on objective decrease (s_phi in IPOPT paper, Section 3.3)
            """

        @s_phi.setter
        def s_phi(self, arg: float, /) -> None: ...

        @property
        def s_theta(self) -> float:
            """
            IPOPT switching condition exponent on constraint violation (s_theta in IPOPT paper, Section 3.3)
            """

        @s_theta.setter
        def s_theta(self, arg: float, /) -> None: ...

        @property
        def merit_sigma(self) -> float:
            """
            Merit backtracking: weight on ||dual residual||^2 relative to ||constraint violation||^2 (default 1.0)
            """

        @merit_sigma.setter
        def merit_sigma(self, arg: float, /) -> None: ...

        @property
        def enable_flat_obj_accept(self) -> bool:
            """
            Accept step when objective is flat, iterate is nearly feasible, and step is non-trivial
            """

        @enable_flat_obj_accept.setter
        def enable_flat_obj_accept(self, arg: bool, /) -> None: ...

        @property
        def flat_obj_dec_tol(self) -> float:
            """
            Absolute full-step decrease below which the objective is considered flat
            """

        @flat_obj_dec_tol.setter
        def flat_obj_dec_tol(self, arg: float, /) -> None: ...

        @property
        def flat_obj_prim_tol(self) -> float:
            """Primal residual must be below this for flat-objective accept"""

        @flat_obj_prim_tol.setter
        def flat_obj_prim_tol(self, arg: float, /) -> None: ...

        @property
        def flat_obj_step_tol(self) -> float:
            """
            Step norm must exceed this for flat-objective accept (ensures non-trivial step)
            """

        @flat_obj_step_tol.setter
        def flat_obj_step_tol(self, arg: float, /) -> None: ...

        @property
        def backtrack_scheme(self) -> sqp.backtrack_scheme:
            """Backtracking scheme: linspace (default) or geometric"""

        @backtrack_scheme.setter
        def backtrack_scheme(self, arg: sqp.backtrack_scheme, /) -> None: ...

        @property
        def backtrack_factor(self) -> float:
            """Geometric reduction factor applied to alpha at each backtracking step"""

        @backtrack_factor.setter
        def backtrack_factor(self, arg: float, /) -> None: ...

        class failure_backup_strategy(enum.Enum):
            """
            Public API type ``moto.sqp.linesearch_setting.failure_backup_strategy``.
            """

            failure_backup_strategy_min_step = 0

            failure_backup_strategy_best_trial = 1

        failure_backup_strategy_min_step: failure_backup_strategy = failure_backup_strategy.failure_backup_strategy_min_step

        failure_backup_strategy_best_trial: failure_backup_strategy = failure_backup_strategy.failure_backup_strategy_best_trial

        class on_failure_action(enum.Enum):
            """Public API type ``moto.sqp.linesearch_setting.on_failure_action``."""

            on_failure_action_abort = 0

            on_failure_action_accept_fallback = 1

        on_failure_action_abort: on_failure_action = on_failure_action.on_failure_action_abort

        on_failure_action_accept_fallback: on_failure_action = on_failure_action.on_failure_action_accept_fallback

    class backtrack_scheme(enum.Enum):
        """Public API type ``moto.sqp.backtrack_scheme``."""

        backtrack_scheme_linspace = 0

        backtrack_scheme_geometric = 1

    backtrack_scheme_linspace: backtrack_scheme = backtrack_scheme.backtrack_scheme_linspace

    backtrack_scheme_geometric: backtrack_scheme = backtrack_scheme.backtrack_scheme_geometric

    class search_method(enum.Enum):
        """Public API type ``moto.sqp.search_method``."""

        search_method_filter = 0

        search_method_merit_backtracking = 1

    search_method_filter: search_method = search_method.search_method_filter

    search_method_merit_backtracking: search_method = search_method.search_method_merit_backtracking

    class initial_state_mode(enum.Enum):
        """Public API type ``moto.sqp.initial_state_mode``."""

        fixed = 0

        optimized = 1

    class settings_type:
        """Public API type ``moto.sqp.settings_type``."""

        @property
        def mu(self) -> float:
            """Barrier parameter for the IPM solver"""

        @property
        def ipm_conditional_corrector(self) -> bool:
            """Whether to use conditional corrector in the IPM solver"""

        @ipm_conditional_corrector.setter
        def ipm_conditional_corrector(self, arg: bool, /) -> None: ...

        @property
        def ipm(self) -> sqp.ipm_config:
            """IPM settings"""

        @property
        def rf(self) -> sqp.iterative_refinement_setting:
            """Iterative refinement settings"""

        @rf.setter
        def rf(self, arg: sqp.iterative_refinement_setting, /) -> None: ...

        @property
        def restoration(self) -> sqp.restoration_settings:
            """Restoration settings"""

        @property
        def eq_init(self) -> sqp.equality_multiplier_init_settings:
            """Equality multiplier initialization settings"""

        @property
        def initial_state(self) -> sqp.initial_state_mode:
            """
            Initial-state treatment: fixed (default) or optimized through an internal virtual stage
            """

        @initial_state.setter
        def initial_state(self, arg: sqp.initial_state_mode, /) -> None: ...

        @property
        def ls(self) -> sqp.linesearch_setting:
            """Line search settings"""

        @property
        def scaling(self) -> sqp.scaling_settings:
            """Jacobian scaling settings"""

        @scaling.setter
        def scaling(self, arg: sqp.scaling_settings, /) -> None: ...

        @property
        def no_except(self) -> bool:
            """Whether to suppress exceptions in parallel jobs"""

        @no_except.setter
        def no_except(self, arg: bool, /) -> None: ...

        @property
        def prim_tol(self) -> float:
            """Primal feasibility tolerance"""

        @prim_tol.setter
        def prim_tol(self, arg: float, /) -> None: ...

        @property
        def dual_tol(self) -> float:
            """Dual feasibility tolerance"""

        @dual_tol.setter
        def dual_tol(self, arg: float, /) -> None: ...

        @property
        def comp_tol(self) -> float:
            """Complementarity feasibility tolerance"""

        @comp_tol.setter
        def comp_tol(self, arg: float, /) -> None: ...

        @property
        def s_max(self) -> float:
            """
            IPOPT-style dual scaling parameter: s_d = max(s_max, ||λ||_1/n_constr)/s_max
            """

        @s_max.setter
        def s_max(self, arg: float, /) -> None: ...

    class scaling_settings:
        """Public API type ``moto.sqp.scaling_settings``."""

        @property
        def scaling_mode(self) -> sqp.scaling_settings.mode:
            """Scaling mode: none, gradient (default), or equilibrium"""

        @scaling_mode.setter
        def scaling_mode(self, arg: sqp.scaling_settings.mode, /) -> None: ...

        @property
        def equilibrium_iters(self) -> int:
            """Number of Ruiz iterations for equilibrium scaling"""

        @equilibrium_iters.setter
        def equilibrium_iters(self, arg: int, /) -> None: ...

        @property
        def min_scale(self) -> float:
            """Minimum scale factor clamp (avoids division by zero)"""

        @min_scale.setter
        def min_scale(self, arg: float, /) -> None: ...

        @property
        def update_ratio_threshold(self) -> float:
            """Recompute scales when dual_res / prim_res >= this threshold"""

        @update_ratio_threshold.setter
        def update_ratio_threshold(self, arg: float, /) -> None: ...

        class mode(enum.Enum):
            """Public API type ``moto.sqp.scaling_settings.mode``."""

            mode_none = 0

            mode_gradient = 1

            mode_equilibrium = 2

        mode_none: mode = mode.mode_none

        mode_gradient: mode = mode.mode_gradient

        mode_equilibrium: mode = mode.mode_equilibrium

    class adaptive_mu_t(enum.Enum):
        """Public API type ``moto.sqp.adaptive_mu_t``."""

        mehrotra_predictor_corrector = 0

        mehrotra_probing = 1

        quality_function_based = 2

        monotonic_decrease = 3

    class iter_result(enum.Enum):
        """Public API type ``moto.sqp.iter_result``."""

        iter_result_unknown = 0

        iter_result_success = 1

        iter_result_exceed_max_iter = 2

        iter_result_restoration_failed = 3

        iter_result_restoration_reached_max_iter = 4

        iter_result_infeasible_stationary = 5

    iter_result_unknown: iter_result = iter_result.iter_result_unknown

    iter_result_success: iter_result = iter_result.iter_result_success

    iter_result_exceed_max_iter: iter_result = iter_result.iter_result_exceed_max_iter

    iter_result_restoration_failed: iter_result = iter_result.iter_result_restoration_failed

    iter_result_restoration_reached_max_iter: iter_result = iter_result.iter_result_restoration_reached_max_iter

    iter_result_infeasible_stationary: iter_result = iter_result.iter_result_infeasible_stationary

    class profile_phase_stat:
        """Public API type ``moto.sqp.profile_phase_stat``."""

        @property
        def name(self) -> str: ...

        @property
        def total_ms(self) -> float: ...

        @property
        def avg_ms(self) -> float: ...

        @property
        def calls(self) -> int: ...

        @property
        def share_of_update(self) -> float: ...

    class profile_iteration:
        """Public API type ``moto.sqp.profile_iteration``."""

        @property
        def index(self) -> int: ...

        @property
        def total_ms(self) -> float: ...

        @property
        def ls_steps(self) -> int: ...

        @property
        def trial_evaluations(self) -> int: ...

    class profile_report:
        """Public API type ``moto.sqp.profile_report``."""

        @property
        def total_ms(self) -> float: ...

        @property
        def initialize_ms(self) -> float: ...

        @property
        def sqp_iterations(self) -> int: ...

        @property
        def trial_evaluations(self) -> int: ...

        @property
        def phases(self) -> list[sqp.profile_phase_stat]: ...

        @property
        def iterations(self) -> list[sqp.profile_iteration]: ...

    class iter_info:
        """Public API type ``moto.sqp.iter_info``."""

        @property
        def result(self) -> sqp.iter_result:
            """Result of the SQP iteration"""

        @property
        def solved(self) -> bool:
            """Whether the problem is solved"""

        @property
        def num_iter(self) -> int:
            """Number of iterations"""

        @num_iter.setter
        def num_iter(self, arg: int, /) -> None: ...

    class barrier_objective_info:
        """Public API type ``moto.sqp.barrier_objective_info``."""

        @property
        def cost(self) -> float: ...

        @property
        def barrier_value(self) -> float: ...

        @property
        def augmented_objective(self) -> float: ...

        @property
        def ls_objective(self) -> float: ...

    class primal_info:
        """Public API type ``moto.sqp.primal_info``."""

        @property
        def inf_res(self) -> float: ...

        @property
        def res_l1(self) -> float: ...

        @property
        def inf_comp(self) -> float: ...

    class dual_info:
        """Public API type ``moto.sqp.dual_info``."""

        @property
        def inf_res(self) -> float: ...

        @property
        def max_eq_norm(self) -> float: ...

        @property
        def max_ineq_norm(self) -> float: ...

        @property
        def max_norm(self) -> float: ...

    class barrier_step_info:
        """Public API type ``moto.sqp.barrier_step_info``."""

        @property
        def search_barrier_dir_deriv(self) -> float: ...

        @property
        def augmented_objective_fullstep_dec(self) -> float: ...

        @property
        def ls_objective_fullstep_dec(self) -> float: ...

    class step_info:
        """Public API type ``moto.sqp.step_info``."""

        @property
        def inf_prim_step(self) -> float: ...

        @property
        def inf_dual_step(self) -> float: ...

        @property
        def inf_eq_dual_step(self) -> float: ...

        @property
        def inf_ineq_dual_step(self) -> float: ...

    class kkt_info:
        """Public API type ``moto.sqp.kkt_info``."""

        @property
        def barrier_objective(self) -> sqp.barrier_objective_info: ...

        @property
        def primal(self) -> sqp.primal_info: ...

        @property
        def dual(self) -> sqp.dual_info: ...

        @property
        def barrier_step(self) -> sqp.barrier_step_info: ...

        @property
        def step(self) -> sqp.step_info: ...

    class result_type(kkt_info):
        """Public API type ``moto.sqp.result_type``."""

        @property
        def iter(self) -> sqp.iter_info:
            """Iteration metadata"""

        @property
        def result(self) -> sqp.iter_result:
            """Result of the SQP iteration"""

        @property
        def solved(self) -> bool:
            """Whether the problem is solved"""

        @property
        def num_iter(self) -> int:
            """Number of iterations"""

        @property
        def inf_prim_res(self) -> float: ...

        @property
        def inf_dual_res(self) -> float: ...

        @property
        def inf_comp_res(self) -> float: ...

    mehrotra_predictor_corrector: adaptive_mu_t = adaptive_mu_t.mehrotra_predictor_corrector

    mehrotra_probing: adaptive_mu_t = adaptive_mu_t.mehrotra_probing

    quality_function_based: adaptive_mu_t = adaptive_mu_t.quality_function_based

    monotonic_decrease: adaptive_mu_t = adaptive_mu_t.monotonic_decrease

    class data_type(moto_pywrap.node_data):
        """Public API type ``moto.sqp.data_type``."""

    @property
    def nodes(self) -> list[sqp.data_type]:
        """Ordered solver-node list"""

class stage_ocp(moto_pywrap.ocp):
    """Public API type ``moto.stage_ocp``."""

    @staticmethod
    def create() -> stage_ocp:
        """Create a new stage OCP problem"""

    def copy(self, disable: Sequence[ expr  | var] = [], enable: Sequence[ expr  | var] = []) -> stage_ocp:
        """Copy the stage, optionally changing its active expressions"""

    @overload
    def disable(self, exprs: Sequence[ expr  | var]) -> None:
        """Disable stage expressions"""

    @overload
    def disable(self, ex: expr) -> None:
        """Disable a stage expression"""

    @overload
    def enable(self, exprs: Sequence[ expr  | var]) -> None:
        """Enable stage expressions"""

    @overload
    def enable(self, ex: expr) -> None:
        """Enable a stage expression"""

    @overload
    def add(self, exprs: Sequence[ expr  | var]) -> None:
        """Add stage expressions"""

    @overload
    def add(self, ex: expr) -> None:
        """Add a stage expression"""

    @property
    def st(self) -> endpoint:
        """Stage start boundary"""

    @property
    def ed(self) -> endpoint:
        """Stage end boundary"""

class sym(expr):
    """Public API type ``moto.sym``."""

    def __str__(self) -> str: ...

    @property
    def default_value(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @default_value.setter
    def default_value(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float, /) -> None: ...

    @property
    def sx(self) -> casadi.SX: ...

    def clone(self, name: str) -> var:
        """Clone into an independent symbol with a fresh uid"""

    def symbolic_integrate(self, x: casadi.SX, dx: casadi.SX) -> casadi.SX: ...

    def symbolic_difference(self, x1: casadi.SX, x0: casadi.SX) -> casadi.SX:
        """difference from x0 to x1, i.e., x1 - x0"""

    def integrate(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], dx: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], alpha: float = 1.0) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    def difference(self, x1: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

    @staticmethod
    def symbol(name: str, dim: int = 1, field: field = field.field___undefined, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var: ...

    @staticmethod
    def states(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> tuple[var, var]: ...

    @staticmethod
    def inputs(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var: ...

    @staticmethod
    def lifted(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var: ...

    @staticmethod
    def params(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var: ...

def stage() -> stage_ocp:
    """Create an authored OCP stage."""

class var(casadi.casadi.SX):
    """A CasADi expression carrying a registered moto symbol."""

    def __init__(self, s: sym):
        """Construct a symbolic variable from its registered moto symbol."""

    @property
    def sym(self) -> sym:
        """The registered moto symbol represented by this expression."""

    def symbolic_integrate(self, x: casadi.casadi.SX, dx: casadi.casadi.SX) -> casadi.casadi.SX:
        """integrate from x with dx, i.e., x + dx"""

    def symbolic_difference(self, x1: casadi.casadi.SX, x0: casadi.casadi.SX) -> casadi.casadi.SX:
        """difference from x0 to x1, i.e., x1 - x0"""

    def clone(self, name: str) -> var:
        """Clone into an independent symbol with a fresh identity."""

    def integrate(self, x: numpy.ndarray, dx: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
        """integrate from x with dx, i.e., x + alpha * dx"""

    def difference(self, x1: numpy.ndarray, x0: numpy.ndarray) -> numpy.ndarray:
        """difference from x0 to x1, i.e., x1 - x0"""

    def finalize(self):
        """Finalize the underlying symbol."""

    @property
    def name(self):
        """Symbol name."""

    @property
    def dim(self):
        """Storage dimension."""

    @property
    def tdim(self):
        """Tangent-space dimension."""

    @property
    def default_value(self):
        """Default numeric value."""

    @default_value.setter
    def default_value(self, val):
        """Set the default numeric value."""

    @property
    def uid(self):
        """Stable symbol identity."""

    @property
    def sx(self):
        """Underlying CasADi SX expression."""

__all__: list = ['approx_order', 'casadi_manifold', 'constr', 'cost', 'dense_dynamics', 'endpoint', 'expr', 'field', 'func', 'ineq', 'lifted', 'pmm_constr', 'semi_implicit_euler', 'sqp', 'stage', 'stage_ocp', 'sym', 'var']
