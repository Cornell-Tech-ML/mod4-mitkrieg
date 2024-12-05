from __future__ import annotations

from typing import TYPE_CHECKING
import minitorch
from abc import ABC

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction(ABC):
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies an scalar function operation on the scalar and creates history"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """$f(x, y) = x + y$"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """$f'_x(x, y) = 1$, $f'_y(x, y) = 1$"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """$f(x) = log(x)$"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """$f'(x) = 1/d * a"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """$f(x, y) = x * y$"""
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """$f'_x(x, y) = y$, $f'_y(x, y) = x$"""
        x, y = ctx.saved_values
        f_x_prime = y
        f_y_prime = x
        return f_x_prime * d_output, f_y_prime * d_output


class Inv(ScalarFunction):
    """The reciprocal function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """$f(x) = 1/x$"""
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """$f'(x) = -1/(x^2)$"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """The negative function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """$f(x) = -x$"""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """$f'(x) = -1$"""
        return -1 * d_output


class Sigmoid(ScalarFunction):
    r"""The sigmoid function $\frac{1}{1+e^{-x}}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        r"""$f(x) = \frac{1}{1+e^{-x}}$"""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        r"""$f'(x) = \sigma(a) (1-\sigma(a)$"""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    r"""The rectalinear unit function $max(0,x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        r"""$f(x) = \max{0}{x}$"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """f'(x) = 0 if < 0 else x"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """The exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """$f(x) = e^x$"""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """$f'(x) = e^x$"""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Computes 1 if a is less than b otherwise 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """f(a,b) = True if a < b else False"""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """f'a(a,b) = 0, f'b(a,b) = 0"""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Computes 1 if a equals b otherwise 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes true if a = b else false"""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """f'a(a,b) = 0, f'b(a,b) = 0"""
        return 0.0, 0.0
