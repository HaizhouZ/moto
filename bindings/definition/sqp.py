from moto import _moto_pywrap

ns_sqp = _moto_pywrap.ns_sqp_impl

__all__ = ["sqp"]


class sqp(ns_sqp):

    def __init__(self, n_job: int = 4):
        super().__init__(n_job)
