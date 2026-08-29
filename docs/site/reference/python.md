# Python API reference

The Python reference is generated from the `.pyi` emitted by
`nanobind.stubgen` after `moto_pywrap` links. Each class, function, property,
and enum is rendered as a linkable API object. The documentation build
analyzes the tracked stub statically, so it does not compile or import Moto.

The tracked snapshot is refreshed from a local build with:

```bash
cmake --build build --target update_python_doc_stub -j4
```

The high-level `moto.sqp`, `moto.stage()`, and `moto.var` wrappers define the
stage-centric user surface described in the tutorials. The generated reference
also contains lower-level nanobind types used to implement that surface.

```{toctree}
:maxdepth: 2

api/moto/index
```
