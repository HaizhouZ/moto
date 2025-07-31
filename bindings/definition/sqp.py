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
        Apply a function to each node in the graph in parallel. Does not work

        Args:
            ndoes: The data nodes from graph_type.flatten_nodes.
            python concurrent: The function to apply to each node.
        """
        for node in self.graph.flatten_nodes():
            callback(node)

    def __apply(self, forward: bool, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the graph in forward order.

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
        if is_unary or is_unary_with_tid:
            view = self.graph.forward_view() if forward else self.graph.backward_view()
            if is_unary:
                while view.update():
                    for node in view[:early_stop]:
                        callback(node)
            elif is_unary_with_tid:
                while view.update():
                    for tid, node in enumerate(view[:early_stop]):
                        callback(tid, node)
        elif is_binary or is_binary_with_tid:
            view = self.graph.forward_view() if forward else self.graph.backward_view()
            if early_stop > 0:
                view = view[: min(early_stop + 1, len(view))]
            if is_binary:
                while view.update():
                    for i in range(len(view) - 1):
                        callback(view[i], view[i + 1])
            elif is_binary_with_tid:
                while view.update():
                    for tid, i in enumerate(range(len(view) - 1)):
                        callback(tid, view[i], view[i + 1])
        else:
            raise TypeError("Callback must be a unary or binary function. Arg: ", params)

    def apply_forward(self, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the graph in forward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(True, callback, none_on_end, early_stop)

    def apply_backward(self, callback: Callable[[Any], None], none_on_end: bool = False, early_stop: int = -1):
        """
        Apply a function to each node in the graph in backward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(False, callback, none_on_end, early_stop)
