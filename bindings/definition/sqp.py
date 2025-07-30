from moto import ns_sqp_impl as ns_sqp
from typing import Callable, List, Any
from multiprocessing import Pool


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

    def __apply(self, forward: bool, callback: Callable[[Any], None], none_on_end: bool = False):
        """
        Apply a function to each node in the graph in forward order.

        Args:
            forward: If True, apply in forward order; otherwise, apply in backward order.
            callback: The function to apply to each node.
        """
        is_unary = type(callback) is Callable[[ns_sqp.data_type], None]
        is_unary_with_tid = type(callback) is Callable[[int, ns_sqp.data_type], None]
        is_binary = type(callback) is Callable[[ns_sqp.data_type, ns_sqp.data_type], None]
        is_binary_with_tid = type(callback) is Callable[[int, ns_sqp.data_type, ns_sqp.data_type], None]
        if is_unary or is_unary_with_tid:
            view = self.graph.forward_view() if forward else self.graph.backward_view()
            if is_unary:
                while view.update():
                    for node in view.nodes:
                        callback(node.data)
            elif is_unary_with_tid:
                while view.update():
                    for tid, node in enumerate(view.nodes):
                        callback(tid, node.data)
        elif is_binary or is_binary_with_tid:
            view = self.graph.forward_view() if forward else self.graph.backward_view()
            if is_binary:
                while view.update():
                    for i in range(len(view.nodes) - 1):
                        callback(view.nodes[i].data, view.nodes[i + 1].data)
            elif is_binary_with_tid:
                while view.update():
                    for tid, i in enumerate(range(len(view.nodes) - 1)):
                        callback(tid, view.nodes[i].data, view.nodes[i + 1].data)
        else:
            raise TypeError("Callback must be a unary or binary function.")
            

    def apply_forward(self, callback: Callable[[Any], None], none_on_end: bool = False):
        """
        Apply a function to each node in the graph in forward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(True, callback, none_on_end)

    def apply_backward(self, callback: Callable[[Any], None], none_on_end: bool = False):
        """
        Apply a function to each node in the graph in backward order.

        Args:
            callback: The function to apply to each node.
        """
        self.__apply(False, callback, none_on_end)