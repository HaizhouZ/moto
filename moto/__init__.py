from __future__ import annotations

from ._bootstrap import load_local_extension

__version__ = "0.1.0"


_moto_pywrap = load_local_extension()

for _name in dir(_moto_pywrap):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_moto_pywrap, _name)

from bindings.definition.shifted_array import *  # noqa: E402,F401,F403
from bindings.definition.var import var  # noqa: E402,F401
from bindings.definition.sqp import *  # noqa: E402,F401,F403
