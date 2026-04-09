from moto import ns_sqp_impl as ns_sqp
from typing import Callable, List, Any
from multiprocessing import Pool
import inspect


class sqp(ns_sqp):

    def __init__(self, n_job: int = 4):
        super().__init__(n_job)
        self.pool = Pool(processes=n_job)

    def for_each(self, callback: Callable[[ns_sqp.data_type], None]):
        """
        Apply a function to each node in the active_data in parallel. Does not work

        Args:
            ndoes: The data nodes from graph_type.flatten_nodes.
            python concurrent: The function to apply to each node.
        """
        for node in self.active_data.flatten_nodes():
            callback(node)

    def __apply(self, forward: bool, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the active_data in forward order.

        Args:
            forward: If True, apply in forward order; otherwise, apply in backward order.
            callback: The function to apply to each node.
        """
        sig = inspect.signature(callback)
        params = [param.annotation for param in sig.parameters.values()]
        is_unary = params == [ns_sqp.data_type]
        is_unary_with_tid = params == [int, ns_sqp.data_type]
        is_binary = params == [ns_sqp.data_type, ns_sqp.data_type]
        is_binary_with_tid = params == [int, ns_sqp.data_type, ns_sqp.data_type]
        nodes = list(self.active_data.flatten_nodes())
        if not forward:
            nodes = list(reversed(nodes))
        if is_unary or is_unary_with_tid:
            end_idx = len(nodes) if early_stop == -1 else min(early_stop, len(nodes))
            if is_unary:
                for node in nodes[:end_idx]:
                    callback(node)
            elif is_unary_with_tid:
                for tid, node in enumerate(nodes[:end_idx]):
                    callback(tid, node)
        elif is_binary or is_binary_with_tid:
            end_idx = len(nodes) if early_stop == -1 else min(early_stop + 1, len(nodes))
            pair_count = max(0, end_idx - 1)
            if is_binary:
                for i in range(pair_count):
                    callback(nodes[i], nodes[i + 1])
                if none_on_end and end_idx > 0:
                    callback(nodes[end_idx - 1], None)
            elif is_binary_with_tid:
                for tid, i in enumerate(range(pair_count)):
                    callback(tid, nodes[i], nodes[i + 1])
                if none_on_end and end_idx > 0:
                    callback(pair_count, nodes[end_idx - 1], None)
        else:
            raise TypeError("Callback must be a unary or binary function. Arg: ", params)

    def apply_forward(self, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the active_data in forward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(True, callback, none_on_end, early_stop)

    def apply_backward(self, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the active_data in backward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(False, callback, none_on_end, early_stop)
