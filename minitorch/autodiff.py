from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_plus[arg] += epsilon

    vals_minus = list(vals)
    vals_minus[arg] -= epsilon

    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Calculates derivatives during backprop"""
        ...

    @property
    def unique_id(self) -> int:
        """A unique id for the variable"""
        ...

    def is_leaf(self) -> bool:
        """Variable is a leaf in the computation graph"""
        ...

    def is_constant(self) -> bool:
        """Variable is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns all parents of the variable in the computation graph"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Runs the chain rule during backpropagation"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # visited = []
    # final = []

    # def visit(v: Variable) -> None:
    #     if v.unique_id in visited:
    #         return
    #     visited.append(v.unique_id)

    #     for parent in v.parents:
    #         visit(parent)

    #     if not v.is_constant():
    #         final.append(v)

    # visit(variable)

    # final.reverse()
    # return final
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # vars = topological_sort(variable)
    # derivs = {v.unique_id: 0 for v in vars}
    # derivs[variable.unique_id] = deriv
    # for var in vars:
    #     if var.is_leaf():
    #         var.accumulate_derivative(derivs[var.unique_id])
    #     else:
    #         chain = var.chain_rule(derivs[var.unique_id])
    #         for v, d in chain:
    #             derivs[v.unique_id] += d

    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved tensor values"""
        return self.saved_values
