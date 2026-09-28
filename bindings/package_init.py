__version__ = "0.1.0"

from . import moto_pywrap as _moto_pywrap
from .definition.public_api import (
    PUBLIC_BINDINGS as _PUBLIC_BINDINGS,
    export_public_bindings as _export_public_bindings,
    publish_type as _publish_type,
)

_export_public_bindings(_moto_pywrap, globals())

from .definition.var import var  # noqa: E402,F401
_publish_type(var, "var")
from .definition.canonical_reuse import install_canonical_reuse as _install_canonical_reuse  # noqa: E402

_install_canonical_reuse(precompute, func)

__all__ = sorted((*_PUBLIC_BINDINGS, "stage", "var"))
