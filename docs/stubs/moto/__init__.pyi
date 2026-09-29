from collections.abc import Callable, Iterable, Iterator, Sequence
import enum
from typing import Annotated, overload

import casadi
import casadi.casadi
import numpy
from numpy.typing import NDArray

from . import definition as definition, moto_pywrap as moto_pywrap


class approx_order(enum.Enum):
    """
    Derivative order requested from a function: value only, first order, or exact second order.
    """

    approx_order_none = 0

    approx_order_zero = 1

    approx_order_first = 2

    approx_order_second = 3

class casadi_manifold(sym):
    """
    State symbol with user-supplied CasADi integration and difference maps for a nonlinear manifold.
    """

    @staticmethod
    def create(name: str, q: casadi.SX, dq: casadi.SX, integrated: casadi.SX, other: casadi.SX, difference: casadi.SX, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> tuple[var, var]:
        """
        Create paired current and terminal manifold states. ``integrated`` defines integrate(q, dq), while ``difference`` defines the tangent displacement from q to other
        """

class constr(func):
    """
    Hard equality constraint. Plain outputs are residuals constrained to zero; ``lhs == rhs`` is normalized to ``lhs - rhs == 0``.
    """

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr:
        """
        Create a hard equality from a residual or CasADi equality relation. Symbol arguments and the solver field are inferred
        """

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0, field: field = field.field___undefined) -> constr:
        """
        Allocate a dimension-only equality for an advanced custom runtime implementation
        """

    def cast_soft(self, type_name: str = 'pmm_constr') -> constr:
        """
        Convert this equality to a soft PMM constraint before adding it to a stage
        """

class cost(func):
    """
    Weighted tracking cost with node-local weight and reference parameters.
    """

    @staticmethod
    def from_vector(name: str, value: casadi.SX, weight: object = 1.0, reference: object = 0.0) -> cost:
        """
        Create a weighted least-squares cost from a vector residual.

        The objective is ``0.5 * (value - reference).T * diag(weight) *
        (value - reference)``. Moto keeps the vector residual and automatically uses
        its weighted Gauss--Newton Hessian, avoiding exact second derivatives of a
        nonlinear residual. ``value`` must contain at least two elements.

        Numeric weights and references become node-local parameter symbols; scalars
        broadcast to the required dimension. Pass explicit ``moto.sym.params`` to
        share parameters between expressions. The resulting handles are available as
        ``cost.weight`` and ``cost.reference``.
        """

    @staticmethod
    def from_scalar(name: str, value: casadi.SX, weight: object = 1.0, reference: object = 0.0) -> cost:
        """
        Create a scalar tracking cost using exact differentiation.

        The scalar objective is ``0.5 * weight * (value - reference)**2``. Unlike
        ``from_vector``, this produces a scalar cost expression and Moto uses its
        ordinary exact second-order approximation. For a nonlinear scalar residual,
        that exact Hessian can be indefinite; use ``from_vector`` when a
        Gauss--Newton residual model is intended.

        Numeric weight and reference values become node-local parameter symbols and
        remain adjustable through ``cost.weight`` and ``cost.reference``.
        """

    @property
    def weight(self) -> var:
        """
        Node-local diagonal weight parameter; change it through ``node.value[cost.weight]`` without recompiling
        """

    @property
    def reference(self) -> var:
        """
        Node-local tracking reference; change it through ``node.value[cost.reference]`` without recompiling
        """

class dense_dynamics(dynamics):
    """
    General implicit dynamics residual projected at runtime by a dense LU factorization of its next-state Jacobian.
    """

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first) -> dense_dynamics:
        """
        Create dense implicit dynamics h(x, u, xn) = 0. The Jacobian with respect to xn must be square and nonsingular at runtime
        """

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0) -> dense_dynamics:
        """
        Allocate dimension-only dense dynamics for an advanced custom runtime implementation
        """

    def mark_shared_inputs(self, shared_inputs: Sequence[var]) -> None:
        """
        Mark input symbols whose projected Jacobian columns are shared with neighboring solver stages
        """

class dynamics(constr):
    """
    Dynamics group that eliminates its predicted-state direction and optional user-selected lifted variables in the local QP.
    """

    class partition:
        """Named row or column partition in a lifted elimination system."""

        @property
        def name(self) -> str:
            """Expression or symbol name"""

        @property
        def uid(self) -> int:
            """Identity used in this elimination graph"""

        @property
        def source_uid(self) -> int:
            """Identity of the authored source handle"""

        @property
        def field(self) -> field:
            """Solver field assigned to the partition"""

        @property
        def offset(self) -> int:
            """Starting row or column in the packed system"""

        @property
        def size(self) -> int:
            """Partition dimension"""

    class block:
        """
        Shaped MX Jacobian block exposed to an elimination-graph builder; structural zeros keep their full shape.
        """

        @property
        def mx(self) -> casadi.MX:
            """CasADi MX value of this block"""

        def param(self, default_val: object | None = None, name: str = '', dim: int = 1) -> var:
            """
            Create or reuse a node-local elimination parameter associated with this block
            """

        def rows(self, begin: int, end: int) -> dynamics.block:
            """Return a row slice while preserving elimination metadata"""

        def add_diag(self, parameter: object, name: str = '', dim: int = 1) -> casadi.MX:
            """
            Add a scalar or vector parameter to the block diagonal and return the resulting MX matrix
            """

    class factor:
        """
        Reusable symbolic factorization handle; repeated solves share one runtime factorization.
        """

        @property
        def matrix(self) -> casadi.MX:
            """Matrix represented by this factorization"""

        def solve(self, rhs: casadi.MX) -> casadi.MX:
            """Solve the factored system for one or more right-hand sides"""

    class system:
        """
        Packed symbolic ``[dynamics; lifted equalities]`` system passed to a user elimination builder.
        """

        @property
        def dyn_residual(self) -> casadi.MX:
            """Packed dynamics residual"""

        @property
        def lift_residual(self) -> casadi.MX:
            """Packed lifted-equality residual"""

        @property
        def action_rhs(self) -> casadi.MX:
            """Forward-action right-hand side used for recovery and refinement"""

        @overload
        def jac(self, equation: constr, variable: var) -> dynamics.block:
            """
            Return the Jacobian block selected by authored equation and variable handles
            """

        @overload
        def jac(self, equation: casadi.SX, variable: var) -> dynamics.block:
            """Return the Jacobian block of an SX equation with respect to a variable"""

        @overload
        def residual(self, equation: constr) -> casadi.MX:
            """Return the residual block for an authored constraint handle"""

        @overload
        def residual(self, equation: str) -> casadi.MX:
            """Return a residual block by its expression name"""

        @property
        def equations(self) -> list[dynamics.partition]:
            """Packed dynamics and lifted-equality row partitions"""

        @property
        def variables(self) -> list[dynamics.partition]:
            """Packed predicted-state and lifted-variable column partitions"""

        def h_l(self) -> casadi.MX:
            """Return the Jacobian with respect to packed eliminated variables [y; l]"""

        def h_x(self) -> casadi.MX:
            """Return the Jacobian with respect to the current state"""

        def h_u(self) -> casadi.MX:
            """Return the Jacobian with respect to uneliminated interval inputs"""

        def h(self) -> casadi.MX:
            """Return the packed residual [dyn; lift]"""

        def solve(self, matrix: casadi.MX, spd: bool = False) -> dynamics.factor:
            """
            Create a reusable symbolic factorization; set spd only for a symmetric positive-definite matrix
            """

        def eliminate(self, solve: Callable[[casadi.MX], casadi.MX], intermediates: Sequence[dynamics.intermediate] = []) -> dynamics.elimination:
            """
            Apply one unsigned inverse action to all required projection columns and return the completed elimination graph
            """

    class intermediate:
        """
        Named MX intermediate cached and exposed by a lifted elimination graph.
        """

        def __init__(self, name: str, value: casadi.MX) -> None:
            """Create a named symbolic intermediate"""

        @property
        def name(self) -> str:
            """Intermediate name"""

        @name.setter
        def name(self, arg: str, /) -> None: ...

        @property
        def value(self) -> casadi.MX:
            """Intermediate MX expression"""

        @value.setter
        def value(self, arg: casadi.MX, /) -> None: ...

    class elimination:
        """
        Unsigned inverse responses returned by a lifted elimination-graph builder.
        """

        def __init__(self, response_x: casadi.MX, response_u: casadi.MX, response_residual: casadi.MX, intermediates: Sequence[dynamics.intermediate], response_action: casadi.MX) -> None: ...

        @property
        def response_x(self) -> casadi.MX:
            """Response h_l^-1 h_x"""

        @response_x.setter
        def response_x(self, arg: casadi.MX, /) -> None: ...

        @property
        def response_u(self) -> casadi.MX:
            """Response h_l^-1 h_u"""

        @response_u.setter
        def response_u(self, arg: casadi.MX, /) -> None: ...

        @property
        def response_residual(self) -> casadi.MX:
            """Response h_l^-1 h"""

        @response_residual.setter
        def response_residual(self, arg: casadi.MX, /) -> None: ...

        @property
        def intermediates(self) -> list[dynamics.intermediate]:
            """Named symbolic intermediates to cache at runtime"""

        @intermediates.setter
        def intermediates(self, arg: Sequence[dynamics.intermediate], /) -> None: ...

        @property
        def response_action(self) -> casadi.MX:
            """Forward inverse action used for recovery and refinement"""

        @response_action.setter
        def response_action(self, arg: casadi.MX, /) -> None: ...

    def with_elimination_graph(self, builder: Callable[[system], elimination], variables: Sequence[var], constraints: Sequence[constr]) -> dynamics:
        """
        Return a dynamics group whose selected ordinary interval inputs and hard equalities are marked for lifted local elimination. Configure the group before adding any selected handle to a stage. The original handles remain valid for costs, bounds, warm starts, and activation.
        """

    @property
    def subconstraints(self) -> list[constr]:
        """Hard equalities owned by this coupled elimination group"""

    @property
    def lifted_args(self) -> list[var]:
        """
        Ordinary interval variables whose local QP directions are eliminated by this group
        """

    @property
    def elimination_parameters(self) -> list[var]:
        """Node-local parameters created while authoring the elimination graph"""

class endpoint:
    """
    Graph boundary view accepting state-only costs and constraints authored on x.
    """

    @overload
    def add(self, exprs: Sequence[ expr  | var]) -> None:
        """
        Add state-only expressions to this boundary; graph composition performs endpoint lowering
        """

    @overload
    def add(self, ex: expr) -> None:
        """
        Add one state-only expression to this boundary; inputs and dynamics are rejected
        """

class expr:
    """Identity-bearing symbolic model expression shared by copied handles."""

    def __bool__(self) -> bool:
        """Return whether this handle contains an expression"""

    def __str__(self) -> str: ...

    @property
    def name(self) -> str:
        """Stable generated-function name"""

    @property
    def field(self) -> field:
        """Solver storage role assigned to the expression"""

    @property
    def dim(self) -> int:
        """Output storage dimension"""

    @property
    def uid(self) -> int:
        """Stable identity shared by copied handles"""

    def finalize(self, block_until_ready: bool = True) -> bool:
        """
        Finalize dependencies and generated callbacks. Graph realization normally calls this automatically
        """

    @property
    def tdim(self) -> int:
        """Output tangent dimension, which may differ from dim on manifolds"""

class field(enum.Enum):
    """
    Internal storage and solver role assigned to symbols and expressions. Ordinary modeling APIs infer this automatically.
    """

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
    """
    Finalizable symbolic function with inferred symbol arguments and generated value and derivative callbacks.
    """

    @property
    def in_args(self) -> list[var]:
        """
        Active input symbols inferred from the symbolic output plus any explicit arguments
        """

    @property
    def value(self) -> Callable[[moto_pywrap.func_approx_data], None]:
        """
        Low-level runtime value callback; normally generated from the symbolic expression
        """

    @value.setter
    def value(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def jacobian(self) -> Callable[[moto_pywrap.func_approx_data], None]:
        """Low-level runtime Jacobian callback; normally generated automatically"""

    @jacobian.setter
    def jacobian(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def hessian(self) -> Callable[[moto_pywrap.func_approx_data], None]:
        """Low-level runtime Hessian callback; normally generated automatically"""

    @hessian.setter
    def hessian(self, arg: Callable[[moto_pywrap.func_approx_data], None], /) -> None: ...

    @property
    def order(self) -> approx_order:
        """Highest derivative order requested for this function"""

    def __str__(self) -> str: ...

    def enable_if_all(self, args: Sequence[ expr  | var]) -> None:
        """
        Enable this function in a stage only when every listed expression is active
        """

    def disable_if_any(self, args: Sequence[ expr  | var]) -> None:
        """
        Disable this function in a stage when any listed expression is inactive
        """

    def enable_if_any(self, args: Sequence[ expr  | var]) -> None:
        """
        Enable this function in a stage when at least one listed expression is active
        """

    def add_argument(self, arg: var) -> None:
        """
        Add an explicit symbol argument before finalization; ordinary SX dependencies are inferred automatically
        """

    def add_arguments(self, args: Sequence[var]) -> None:
        """
        Add explicit symbol arguments before finalization; ordinary SX dependencies are inferred automatically
        """

    def set_analytic_jacobian(self, arg: var, jacobian: casadi.SX) -> None:
        """
        Provide an analytic output Jacobian with respect to one symbol, replacing automatic differentiation for that block
        """

    def set_analytic_hessian(self, arg0: var, arg1: var, hessian: casadi.SX) -> None:
        """
        Provide an analytic Hessian block, replacing automatic differentiation for that argument pair
        """

    def remap_arguments(self, remap: Sequence[tuple[var, var]]) -> func:
        """Create a fresh remapped function"""

    def reuse_remap(self, remap: Sequence[tuple[var, var]]) -> func:
        """Reuse the cached function for this remap"""

    @staticmethod
    def canonical(prefix: str, identity: object, symbols: Iterable[object], function_factory: Callable[[str], object]) -> object:
        """Create once, then remap one canonical generated function."""

class ineq(constr):
    """
    Hard inequality constraint handled by the interior-point method. Plain outputs mean residual <= 0; relational expressions are normalized automatically.
    """

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr:
        """
        Create an inequality from a residual or CasADi <, <=, >, or >= relation. Symbol arguments and the solver field are inferred
        """

    @overload
    @staticmethod
    def create(name: str, order: approx_order = approx_order.approx_order_first, dim: int = 0, field: field = field.field___undefined) -> constr:
        """
        Allocate a dimension-only inequality for an advanced custom runtime implementation
        """

    @overload
    @staticmethod
    def create(name: str, out: casadi.SX, lb: object, ub: object, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr:
        """
        Create elementwise lower and upper bounds on an arbitrary SX expression; bounds may be numeric or direct parameter symbols
        """

    @staticmethod
    def bounds(name: str, value: var, lb: object, ub: object, order: approx_order = approx_order.approx_order_first, field: field = field.field___undefined) -> constr:
        """
        Create elementwise bounds directly on a variable. Scalar numeric bounds broadcast; vector bounds must match the variable dimension
        """

class pmm_constr(constr):
    """Soft equality constraint enforced by the proximal multiplier method."""

    @property
    def rho(self) -> float:
        """Dual penalty weight for the proximal multiplier method"""

    @rho.setter
    def rho(self, arg: float, /) -> None: ...

class precompute(moto_pywrap.custom_func):
    """
    Generated node-local symbolic precompute with automatic dependency placement
    """

    @staticmethod
    def create(name: str, outputs: Sequence[casadi.SX]) -> precompute:
        """Create one precompute from SX outputs and allocate public value caches"""

    @property
    def outputs(self) -> list[var]:
        """Public node-local value caches populated before dependent functions"""

    @overload
    def instantiate(self, inputs: Sequence[var]) -> precompute:
        """
        Instantiate positionally on new inputs; Moto allocates and reuses value and derivative caches
        """

    @overload
    def instantiate(self, input_remap: Sequence[tuple[var, var]]) -> precompute:
        """
        Instantiate with an input-only remap; Moto allocates and reuses value and derivative caches
        """

    def reuse_remap(self, remap: Sequence[tuple[var, var]]) -> precompute:
        """
        Low-level complete-symbol remap that reuses this generated implementation
        """

    @staticmethod
    def canonical(prefix: str, identity: object, inputs: Iterable[object], output_factory: Callable[[], Iterable[object]]) -> tuple[object, ...]:
        """
        Create once per structural identity and return remapped value caches.

                ``output_factory`` is evaluated only for the first call with a given
                ``prefix`` and hashable ``identity``. Later calls instantiate the
                canonical generated precompute on ``inputs`` while Moto allocates and
                reuses its value and derivative caches.
        """

class semi_implicit_euler(dynamics):
    """
    Structured dynamics using a pre-generated semi-implicit Euler projection instead of a dense next-state solve.
    """

    class state(enum.Enum):
        """State layout used by the structured Euler projection."""

        pos = 0
        """Position-only kinematic state"""

        pos_vel = 1
        """
        Complete position-and-velocity state with semi-implicit block structure
        """

    @staticmethod
    def create(name: str, out: casadi.SX, state: state = state.pos_vel, order: approx_order = approx_order.approx_order_first) -> semi_implicit_euler:
        """
        Create structured Euler dynamics from a residual written on current x, interval u, and terminal xn
        """

    def mark_shared_inputs(self, shared_inputs: Sequence[var]) -> None:
        """
        Mark input symbols whose projected Jacobian columns are shared with neighboring solver stages
        """

class sqp:
    """Stage-structured nonlinear OCP model and nonsmooth SQP solver."""

    def __init__(self, n_job: int = 4) -> None:
        """Constructor for the SQP solver with a specified number of jobs"""

    class stage_list:
        """Mutable ordered collection of authored OCP stages."""

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
    def stages(self) -> stage_list:
        """Mutable ordered graph-owned stage vector"""

    def update(self, n_iter: int = 1, verbose: bool = True, profile: bool = False) -> result_type:
        """Update the SQP solver for a given number of iterations"""

    def get_profile_report(self) -> profile_report:
        """Get the latest SQP wall-clock profile report"""

    @property
    def n_job(self) -> int:
        """Effective maximum number of SQP worker threads"""

    @property
    def settings(self) -> settings_type:
        """Top-level SQP solver settings"""

    @property
    def linear_solve_last(self) -> linear_solve_info:
        """Diagnostics for the most recent Newton direction"""

    class ipm_config:
        """Interior-point method options for inequality constraints."""

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
        def mu_method(self) -> adaptive_mu_t:
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

    class regularization_settings:
        """
        Adaptive primal regularization and Newton-direction validation options.
        """

        @property
        def enabled(self) -> bool:
            """
            Whether failed or inaccurate Newton directions are retried with primal regularization (default: true)
            """

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def validate_direction(self) -> bool:
            """
            Whether to form normalized full recovered KKT residuals and reject inaccurate directions (default: false)
            """

        @validate_direction.setter
        def validate_direction(self, arg: bool, /) -> None: ...

        @property
        def initial(self) -> float:
            """
            Initial positive primal regularization after an unregularized attempt fails (default: 1e-4)
            """

        @initial.setter
        def initial(self, arg: float, /) -> None: ...

        @property
        def increase_factor(self) -> float:
            """
            Multiplier applied to primal regularization between retry attempts (default: 10)
            """

        @increase_factor.setter
        def increase_factor(self, arg: float, /) -> None: ...

        @property
        def decrease_factor(self) -> float:
            """
            Multiplier applied to the last successful regularization when seeding a later solve (default: 1/3)
            """

        @decrease_factor.setter
        def decrease_factor(self, arg: float, /) -> None: ...

        @property
        def maximum(self) -> float:
            """Upper bound for an attempted primal regularization (default: 1e8)"""

        @maximum.setter
        def maximum(self, arg: float, /) -> None: ...

        @property
        def max_attempts(self) -> int:
            """
            Maximum Newton solve attempts per direction, including the unregularized attempt (default: 14)
            """

        @max_attempts.setter
        def max_attempts(self, arg: int, /) -> None: ...

        @property
        def residual_tolerance(self) -> float:
            """
            Acceptance tolerance for normalized full direction residuals when validation is enabled (default: 1e-6)
            """

        @residual_tolerance.setter
        def residual_tolerance(self, arg: float, /) -> None: ...

    class linear_solve_status(enum.Enum):
        """Acceptance status of a Newton-direction linear solve."""

        success = 0

        factorization_failed = 1

        inaccurate_direction = 2

        inconsistent_equalities = 3

        nonfinite_direction = 4

    class linear_solve_info:
        """Diagnostics from the most recent Newton-direction linear solve."""

        @property
        def status(self) -> linear_solve_status:
            """Acceptance status of the most recent Newton direction"""

        @property
        def attempts(self) -> int:
            """Number of Newton solve attempts used by the most recent direction"""

        @property
        def regularization(self) -> float:
            """Primal regularization used by the most recent solve attempt"""

        @property
        def stationarity_residual(self) -> float:
            """
            Maximum normalized recovered-stationarity residual from final direction validation
            """

        @property
        def equality_residual(self) -> float:
            """
            Maximum normalized hard-equality residual from final direction validation
            """

        @property
        def inequality_residual(self) -> float:
            """
            Maximum normalized inequality, soft-equality, or restoration residual from final direction validation
            """

    class iterative_refinement_setting:
        """Residual tolerances and iteration limits for iterative refinement."""

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
        """Feasibility-restoration phase configuration."""

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
        """Equality-multiplier recovery and initialization options."""

        @property
        def enabled(self) -> bool:
            """
            Master switch for equality and soft-equality multiplier recovery (default: true)
            """

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def recover_on_warm_start(self) -> bool:
            """
            Whether warm SQP initialization recovers equality multipliers instead of preserving them (default: true)
            """

        @recover_on_warm_start.setter
        def recover_on_warm_start(self, arg: bool, /) -> None: ...

        @property
        def rebuild_after_restoration_exit(self) -> bool:
            """
            Whether to recover equality multipliers after restoration exits successfully (default: true)
            """

        @rebuild_after_restoration_exit.setter
        def rebuild_after_restoration_exit(self, arg: bool, /) -> None: ...

        @property
        def rho_eq(self) -> float:
            """
            PMM penalty used for equality-type constraints in the equality-init overlay (default: 1)
            """

        @rho_eq.setter
        def rho_eq(self, arg: float, /) -> None: ...

        @property
        def rf(self) -> iterative_refinement_setting:
            """
            Dedicated iterative-refinement settings used only during equality-multiplier recovery (disabled by default)
            """

        @rf.setter
        def rf(self, arg: sqp.iterative_refinement_setting, /) -> None: ...

    class linesearch_setting(moto_pywrap.linesearch_config):
        """Filter or merit-backtracking globalization configuration."""

        @property
        def enabled(self) -> bool:
            """Whether to use line search"""

        @enabled.setter
        def enabled(self, arg: bool, /) -> None: ...

        @property
        def max_steps(self) -> int:
            """
            Optional maximum number of backtracking reductions; zero uses the computed minimum step only
            """

        @max_steps.setter
        def max_steps(self, arg: int, /) -> None: ...

        @property
        def failure_strategy(self) -> failure_backup_strategy:
            """Line search failure backup strategy"""

        @failure_strategy.setter
        def failure_strategy(self, arg: sqp.linesearch_setting.failure_backup_strategy, /) -> None: ...

        @property
        def on_failure(self) -> on_failure_action:
            """Action to take after line search exhausts max_steps"""

        @on_failure.setter
        def on_failure(self, arg: sqp.linesearch_setting.on_failure_action, /) -> None: ...

        @property
        def method(self) -> search_method:
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
        def alpha_min_frac(self) -> float:
            """
            IPOPT gamma_alpha safety factor for the computed minimum filter step (default: 0.05)
            """

        @alpha_min_frac.setter
        def alpha_min_frac(self, arg: float, /) -> None: ...

        @property
        def watchdog_shortened_iter_trigger(self) -> int:
            """
            Consecutive accepted shortened steps before starting the IPOPT watchdog; zero disables it (default: 10)
            """

        @watchdog_shortened_iter_trigger.setter
        def watchdog_shortened_iter_trigger(self, arg: int, /) -> None: ...

        @property
        def watchdog_trial_iter_max(self) -> int:
            """
            Maximum provisional watchdog iterations before restoring its reference iterate (default: 3)
            """

        @watchdog_trial_iter_max.setter
        def watchdog_trial_iter_max(self, arg: int, /) -> None: ...

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
        def backtrack_scheme(self) -> backtrack_scheme:
            """Backtracking scheme: geometric (default) or linspace"""

        @backtrack_scheme.setter
        def backtrack_scheme(self, arg: sqp.backtrack_scheme, /) -> None: ...

        @property
        def backtrack_factor(self) -> float:
            """Geometric reduction factor applied to alpha at each backtracking step"""

        @backtrack_factor.setter
        def backtrack_factor(self, arg: float, /) -> None: ...

        class failure_backup_strategy(enum.Enum):
            """Fallback trial point selected after line-search failure."""

            failure_backup_strategy_min_step = 0

            failure_backup_strategy_best_trial = 1

        failure_backup_strategy_min_step: failure_backup_strategy = failure_backup_strategy.failure_backup_strategy_min_step

        failure_backup_strategy_best_trial: failure_backup_strategy = failure_backup_strategy.failure_backup_strategy_best_trial

        class on_failure_action(enum.Enum):
            """Action taken when line search cannot accept a trial step."""

            on_failure_action_abort = 0

            on_failure_action_accept_fallback = 1

        on_failure_action_abort: on_failure_action = on_failure_action.on_failure_action_abort

        on_failure_action_accept_fallback: on_failure_action = on_failure_action.on_failure_action_accept_fallback

    class backtrack_scheme(enum.Enum):
        """Step-size sequence used during line-search backtracking."""

        backtrack_scheme_linspace = 0

        backtrack_scheme_geometric = 1

    backtrack_scheme_linspace: backtrack_scheme = backtrack_scheme.backtrack_scheme_linspace

    backtrack_scheme_geometric: backtrack_scheme = backtrack_scheme.backtrack_scheme_geometric

    class search_method(enum.Enum):
        """Globalization method used to accept or reject SQP steps."""

        search_method_filter = 0

        search_method_merit_backtracking = 1

    search_method_filter: search_method = search_method.search_method_filter

    search_method_merit_backtracking: search_method = search_method.search_method_merit_backtracking

    class initial_state_mode(enum.Enum):
        """Whether the initial state is fixed or optimized."""

        fixed = 0

        optimized = 1

    class settings_type:
        """Top-level SQP configuration, available from ``sqp.settings``."""

        @property
        def mu(self) -> float:
            """Barrier parameter for the IPM solver"""

        @property
        def ipm_conditional_corrector(self) -> bool:
            """Whether to use conditional corrector in the IPM solver"""

        @ipm_conditional_corrector.setter
        def ipm_conditional_corrector(self, arg: bool, /) -> None: ...

        @property
        def ipm(self) -> ipm_config:
            """IPM settings"""

        @property
        def rf(self) -> iterative_refinement_setting:
            """Iterative refinement settings"""

        @rf.setter
        def rf(self, arg: sqp.iterative_refinement_setting, /) -> None: ...

        @property
        def regularization(self) -> regularization_settings:
            """Adaptive Newton direction safeguards"""

        @regularization.setter
        def regularization(self, arg: sqp.regularization_settings, /) -> None: ...

        @property
        def restoration(self) -> restoration_settings:
            """Restoration settings"""

        @property
        def eq_init(self) -> equality_multiplier_init_settings:
            """Equality multiplier initialization settings"""

        @property
        def initial_state(self) -> initial_state_mode:
            """
            Initial-state treatment: fixed (default) or optimized through an internal virtual stage
            """

        @initial_state.setter
        def initial_state(self, arg: sqp.initial_state_mode, /) -> None: ...

        @property
        def ls(self) -> linesearch_setting:
            """Line search settings"""

        @property
        def scaling(self) -> scaling_settings:
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
        """Jacobian scaling mode and recomputation thresholds."""

        @property
        def scaling_mode(self) -> mode:
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
            """Available Jacobian scaling algorithms."""

            mode_none = 0

            mode_gradient = 1

            mode_equilibrium = 2

        mode_none: mode = mode.mode_none

        mode_gradient: mode = mode.mode_gradient

        mode_equilibrium: mode = mode.mode_equilibrium

    class adaptive_mu_t(enum.Enum):
        """Barrier-parameter update strategies."""

        mehrotra_predictor_corrector = 0

        mehrotra_probing = 1

        quality_function_based = 2

        monotonic_decrease = 3

    class iter_result(enum.Enum):
        """Termination status returned by an SQP update."""

        iter_result_unknown = 0

        iter_result_success = 1

        iter_result_exceed_max_iter = 2

        iter_result_restoration_failed = 3

        iter_result_restoration_reached_max_iter = 4

        iter_result_infeasible_stationary = 5

        iter_result_numerical_failure = 6

    iter_result_unknown: iter_result = iter_result.iter_result_unknown

    iter_result_success: iter_result = iter_result.iter_result_success

    iter_result_exceed_max_iter: iter_result = iter_result.iter_result_exceed_max_iter

    iter_result_restoration_failed: iter_result = iter_result.iter_result_restoration_failed

    iter_result_restoration_reached_max_iter: iter_result = iter_result.iter_result_restoration_reached_max_iter

    iter_result_infeasible_stationary: iter_result = iter_result.iter_result_infeasible_stationary

    iter_result_numerical_failure: iter_result = iter_result.iter_result_numerical_failure

    class profile_phase_stat:
        """Aggregated wall-clock statistics for one solver phase."""

        @property
        def name(self) -> str:
            """Stable solver phase name"""

        @property
        def total_ms(self) -> float:
            """Total wall-clock milliseconds spent in this phase"""

        @property
        def avg_ms(self) -> float:
            """Average milliseconds per recorded call"""

        @property
        def calls(self) -> int:
            """Number of recorded calls"""

        @property
        def share_of_update(self) -> float:
            """Fraction of the complete update wall time"""

    class profile_iteration:
        """Wall-clock and trial-evaluation statistics for one SQP iteration."""

        @property
        def index(self) -> int:
            """One-based SQP iteration index"""

        @property
        def total_ms(self) -> float:
            """Total wall-clock milliseconds for the iteration"""

        @property
        def ls_steps(self) -> int:
            """Number of line-search backtracking reductions"""

        @property
        def trial_evaluations(self) -> int:
            """Number of nonlinear trial-point evaluations"""

    class profile_report:
        """Wall-clock profile collected by the most recent profiled update."""

        @property
        def total_ms(self) -> float:
            """Total wall-clock milliseconds for the profiled update"""

        @property
        def initialize_ms(self) -> float:
            """Milliseconds spent initializing and linearizing the solve"""

        @property
        def sqp_iterations(self) -> int:
            """Number of recorded SQP iterations"""

        @property
        def trial_evaluations(self) -> int:
            """Total nonlinear trial-point evaluations"""

        @property
        def phases(self) -> list[profile_phase_stat]:
            """Aggregated statistics for phases that were executed"""

        @property
        def iterations(self) -> list[profile_iteration]:
            """Per-iteration wall-clock statistics"""

    class iter_info:
        """Termination status and iteration count."""

        @property
        def result(self) -> iter_result:
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
        """Cost, barrier, and line-search objective values at an iterate."""

        @property
        def cost(self) -> float:
            """Original nonlinear objective value"""

        @property
        def barrier_value(self) -> float:
            """Interior-point logarithmic barrier contribution"""

        @property
        def augmented_objective(self) -> float:
            """Objective plus active barrier and augmentation terms"""

        @property
        def ls_objective(self) -> float:
            """Objective value used by globalization"""

    class primal_info:
        """Primal feasibility and complementarity residual summary."""

        @property
        def inf_res(self) -> float:
            """Infinity norm of primal constraint violation"""

        @property
        def res_l1(self) -> float:
            """L1 norm of primal constraint violation"""

        @property
        def inf_comp(self) -> float:
            """Infinity norm of complementarity residual"""

    class dual_info:
        """Dual stationarity and multiplier-norm summary."""

        @property
        def inf_res(self) -> float:
            """Infinity norm of the Lagrangian stationarity residual"""

        @property
        def max_eq_norm(self) -> float:
            """Largest hard-equality multiplier norm"""

        @property
        def max_ineq_norm(self) -> float:
            """Largest inequality or soft-constraint multiplier norm"""

        @property
        def max_norm(self) -> float:
            """Largest multiplier norm across all constraint types"""

    class barrier_step_info:
        """Predicted barrier and line-search objective change for an SQP step."""

        @property
        def search_barrier_dir_deriv(self) -> float:
            """Directional derivative of the barrier search objective"""

        @property
        def augmented_objective_fullstep_dec(self) -> float:
            """Predicted augmented-objective decrease for a full step"""

        @property
        def ls_objective_fullstep_dec(self) -> float:
            """Predicted globalization-objective decrease for a full step"""

    class step_info:
        """Infinity norms of the latest primal and dual steps."""

        @property
        def inf_prim_step(self) -> float:
            """Infinity norm of the primal step"""

        @property
        def inf_dual_step(self) -> float:
            """Infinity norm of the complete dual step"""

        @property
        def inf_eq_dual_step(self) -> float:
            """Infinity norm of the hard-equality multiplier step"""

        @property
        def inf_ineq_dual_step(self) -> float:
            """Infinity norm of inequality and soft-constraint multiplier steps"""

    class kkt_info:
        """KKT residual, objective, and step diagnostics."""

        @property
        def barrier_objective(self) -> barrier_objective_info:
            """Objective and barrier values"""

        @property
        def primal(self) -> primal_info:
            """Primal feasibility and complementarity summary"""

        @property
        def dual(self) -> dual_info:
            """Dual stationarity and multiplier summary"""

        @property
        def barrier_step(self) -> barrier_step_info:
            """Predicted objective changes for the latest direction"""

        @property
        def step(self) -> step_info:
            """Primal and dual step norms"""

    class result_type(kkt_info):
        """
        Result returned by ``sqp.update``, including termination and KKT diagnostics.
        """

        @property
        def iter(self) -> iter_info:
            """Iteration metadata"""

        @property
        def result(self) -> iter_result:
            """Result of the SQP iteration"""

        @property
        def solved(self) -> bool:
            """Whether the problem is solved"""

        @property
        def num_iter(self) -> int:
            """Number of iterations"""

        @property
        def inf_prim_res(self) -> float:
            """Convenience view of ``primal.inf_res``"""

        @property
        def inf_dual_res(self) -> float:
            """Convenience view of ``dual.inf_res``"""

        @property
        def inf_comp_res(self) -> float:
            """Convenience view of ``primal.inf_comp``"""

    mehrotra_predictor_corrector: adaptive_mu_t = adaptive_mu_t.mehrotra_predictor_corrector

    mehrotra_probing: adaptive_mu_t = adaptive_mu_t.mehrotra_probing

    quality_function_based: adaptive_mu_t = adaptive_mu_t.quality_function_based

    monotonic_decrease: adaptive_mu_t = adaptive_mu_t.monotonic_decrease

    class data_type(moto_pywrap.node_data):
        """
        Realized node data containing symbol values and function approximations.
        """

        @property
        def prob(self) -> moto_pywrap.ocp:
            """Finalized OCP problem represented by this solver node"""

        @property
        def value(self) -> moto_pywrap.sym_data:
            """Node-local symbol values"""

        def data(self, function: func) -> moto_pywrap.func_approx_data:
            """Runtime value and derivative storage for a generated function"""

    @property
    def nodes(self) -> list[data_type]:
        """Ordered solver-node list"""

class stage_ocp(moto_pywrap.ocp):
    """
    Authored interval stage containing dynamics, path terms, and start/end boundary views.
    """

    @staticmethod
    def create() -> stage_ocp:
        """
        Create an empty authored interval stage; prefer the public ``moto.stage()`` helper
        """

    def copy(self, disable: Sequence[ expr  | var] = [], enable: Sequence[ expr  | var] = []) -> stage_ocp:
        """
        Create an independent stage container while sharing immutable expression handles; optionally change active expressions
        """

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
        """Add interval dynamics, costs, or path constraints"""

    @overload
    def add(self, ex: expr) -> None:
        """Add one interval dynamics, cost, or path constraint"""

    @property
    def st(self) -> endpoint:
        """
        Incoming state-only boundary; on a connected graph it shares the predecessor's terminal state
        """

    @property
    def ed(self) -> endpoint:
        """
        Outgoing state-only boundary represented by this interval's terminal state
        """

class sym(expr):
    """
    Registered Moto symbol with a CasADi SX value, field role, identity, and node-initialization default.
    """

    def __str__(self) -> str: ...

    @property
    def default_value(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        Default copied into active node storage when the graph is first realized; a scalar broadcasts to the symbol dimension
        """

    @default_value.setter
    def default_value(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float, /) -> None: ...

    @property
    def sx(self) -> casadi.SX:
        """Underlying CasADi SX expression used to author formulas"""

    def clone(self, name: str) -> var:
        """Clone into an independent symbol with a fresh uid"""

    def symbolic_integrate(self, x: casadi.SX, dx: casadi.SX) -> casadi.SX:
        """Apply the symbol's manifold integration rule to symbolic values"""

    def symbolic_difference(self, x1: casadi.SX, x0: casadi.SX) -> casadi.SX:
        """difference from x0 to x1, i.e., x1 - x0"""

    def integrate(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], dx: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], alpha: float = 1.0) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """Numerically integrate x by alpha times a tangent increment"""

    def difference(self, x1: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """Return the numerical tangent displacement from x0 to x1"""

    @staticmethod
    def symbol(name: str, dim: int = 1, field: field = field.field___undefined, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var:
        """Create a registered symbol in an explicitly selected field"""

    @staticmethod
    def states(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> tuple[var, var]:
        """
        Create the paired current-state and interval-terminal-state symbols (x, xn) with one shared default
        """

    @staticmethod
    def inputs(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var:
        """Create an interval input decision variable"""

    @staticmethod
    def params(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var:
        """
        Create node-local numeric parameters that can change without regenerating derivatives
        """

    @staticmethod
    def usr_var(name: str, dim: int = 1, default_val: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | float | None = None) -> var:
        """
        Create a user-defined nonstandard storage symbol for advanced extensions
        """

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
        """Apply this symbol's manifold integration rule to symbolic values."""

    def symbolic_difference(self, x1: casadi.casadi.SX, x0: casadi.casadi.SX) -> casadi.casadi.SX:
        """Return the symbolic tangent displacement from ``x0`` to ``x1``."""

    def clone(self, name: str) -> var:
        """Clone into an independent symbol with a fresh identity."""

    def integrate(self, x: numpy.ndarray, dx: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
        """
        Numerically integrate ``x`` by ``alpha * dx`` on this symbol's manifold.
        """

    def difference(self, x1: numpy.ndarray, x0: numpy.ndarray) -> numpy.ndarray:
        """Return the numerical tangent displacement from ``x0`` to ``x1``."""

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
        """
        Value copied into each active node when its runtime storage is created.
        """

    @default_value.setter
    def default_value(self, val):
        """
        Set the node-construction default; this does not rewrite existing nodes.
        """

    @property
    def uid(self):
        """Stable symbol identity."""

    @property
    def sx(self):
        """Underlying CasADi SX expression."""

__all__: list = ['__version__', 'approx_order', 'casadi_manifold', 'constr', 'cost', 'dense_dynamics', 'dynamics', 'endpoint', 'expr', 'field', 'func', 'ineq', 'pmm_constr', 'precompute', 'semi_implicit_euler', 'sqp', 'stage', 'stage_ocp', 'sym', 'var']
