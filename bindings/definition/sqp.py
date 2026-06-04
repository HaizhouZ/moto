from moto import _moto_pywrap

ns_sqp = _moto_pywrap.ns_sqp_impl

__all__ = ["sqp"]


class _graph_proxy:
    def __init__(self, owner: "sqp"):
        self._owner = owner

    def add_path(self, start, end, edge, n_edges: int):
        """Add a linear path of exactly n_edges intervals."""
        self._owner._add_path(start, end, edge, n_edges)

    def flatten_nodes(self):
        return self._owner._flatten_nodes()


class sqp(ns_sqp):

    def __init__(self, n_job: int = 4):
        super().__init__(n_job)
        self._graph_proxy = _graph_proxy(self)

    @property
    def graph(self):
        return self._graph_proxy
