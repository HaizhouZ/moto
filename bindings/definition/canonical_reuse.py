"""Process-wide canonical symbolic reuse for the Python modeling API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import re
from threading import RLock


@dataclass(frozen=True)
class _PrecomputeSource:
    producer: object
    inputs: tuple


@dataclass(frozen=True)
class _FunctionSource:
    function: object
    symbols: tuple


_precompute_sources = {}
_function_sources = {}
_reuse_lock = RLock()


def _key_and_name(prefix, identity):
    try:
        hash(identity)
    except TypeError as error:
        raise TypeError("canonical structural identity must be hashable") from error
    safe_prefix = re.sub(r"[^0-9A-Za-z_]", "_", str(prefix))
    if not safe_prefix or safe_prefix[0].isdigit():
        safe_prefix = f"canonical_{safe_prefix}"
    digest = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:16]
    return (safe_prefix, identity), f"{safe_prefix}_{digest}"


def install_canonical_reuse(precompute_type, function_type):
    """Attach canonical lazy-reuse operations to the native public types."""

    def canonical_precompute(
        prefix: str,
        identity: object,
        inputs: Iterable[object],
        output_factory: Callable[[], Iterable[object]],
    ) -> tuple[object, ...]:
        """Return public cache outputs from one canonical lazy precompute."""
        key, name = _key_and_name(prefix, identity)
        current_inputs = tuple(inputs)
        with _reuse_lock:
            source = _precompute_sources.get(key)
            if source is None:
                producer = precompute_type.create(name, list(output_factory()))
                source = _PrecomputeSource(producer, current_inputs)
                _precompute_sources[key] = source
                return tuple(producer.outputs)
        if len(source.inputs) != len(current_inputs):
            raise ValueError(
                f"canonical precompute {name} expected "
                f"{len(source.inputs)} public inputs, got "
                f"{len(current_inputs)}"
            )
        producer = source.producer.instantiate(
            list(zip(source.inputs, current_inputs))
        )
        return tuple(producer.outputs)

    def canonical_function(
        prefix: str,
        identity: object,
        symbols: Iterable[object],
        function_factory: Callable[[str], object],
    ) -> object:
        """Create once, then remap one canonical generated function."""
        key, name = _key_and_name(prefix, identity)
        current_symbols = tuple(symbols)
        with _reuse_lock:
            source = _function_sources.get(key)
            if source is None:
                function = function_factory(name)
                if function.name != name:
                    raise ValueError(
                        f"canonical function factory returned {function.name}; "
                        f"expected {name}"
                    )
                source = _FunctionSource(function, current_symbols)
                _function_sources[key] = source
                return function
        if len(source.symbols) != len(current_symbols):
            raise ValueError(
                f"canonical function {name} expected "
                f"{len(source.symbols)} public symbols, got "
            )
        return source.function.reuse_remap(
            list(zip(source.symbols, current_symbols))
        )

    canonical_precompute.__module__ = "moto"
    canonical_precompute.__name__ = "canonical"
    canonical_precompute.__qualname__ = "precompute.canonical"
    canonical_function.__module__ = "moto"
    canonical_function.__name__ = "canonical"
    canonical_function.__qualname__ = "func.canonical"
    precompute_type.canonical = staticmethod(canonical_precompute)
    function_type.canonical = staticmethod(canonical_function)
