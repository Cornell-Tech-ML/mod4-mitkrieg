"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Generator, Sequence


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Performs multiplication on two floats

    Args:
    ----
        x (float): the multiplicand to perform multiplication on
        y (float): the multiplier

    Returns:
    -------
        float: the product, x * y

    """
    return x * y


# - id
def id(a: float) -> float:
    """Returns the input unchanged

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: the same value as input a

    """
    return a


# - add
def add(x: float, y: float) -> float:
    """Performs addition on two floats

    Args:
    ----
        x (float): the first addend
        y (float): the second addend

    Returns:
    -------
        float: the sum, x + y

    """
    return x + y


# - neg
def neg(a: float) -> float:
    """Negates a number

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: negative of a

    """
    return -a


# - lt
def lt(x: float, y: float) -> bool:
    """Evaluates if `x` is less than `y`

    Args:
    ----
        x (float): a number
        y (float): a number

    Returns:
    -------
        bool: True if `x` is less than `y`, otherwise false

    """
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    """Evaluates if `x` is equal to `y`

    Args:
    ----
        x (float): a number
        y (float): a number

    Returns:
    -------
        bool: True if `x` is equal to `y`, otherwise false

    """
    return x == y


# - max
def max(x: float, y: float) -> float:
    """Finds the larger of two numbers `x` and `y`

    Args:
    ----
        x (float): a number
        y (float): a number

    Returns:
    -------
        float: the larger value between `x` and `y`

    """
    if x <= y:
        return y
    else:
        return x


# - is_close
def is_close(x: float, y: float) -> bool:
    """Returns true if the value of y is within + or - 0.01 of x

    Args:
    ----
        x (float): a number
        y (float): a number

    Returns:
    -------
        bool: True if y is within + or - 0.01 of x, otherwise False

    """
    return abs(x - y) <= 1e-2


# - sigmoid
def sigmoid(a: float) -> float:
    r"""Returns the value of the sigmoid function evaluated at `a`

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: f(a) where $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    """
    if a >= 0:
        return 1.0 / (1.0 + math.e ** (-a))
    else:
        return math.e ** (a) / (1.0 + math.e ** (a))


# - relu
def relu(a: float) -> float:
    """Returns the value of the rectified linear unit function evaluated at `a`

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: f(a) where f(x) = max(0,x)

    """
    return a if a > 0 else 0.0


# - log
def log(a: float) -> float:
    """Returns the value of the natural logarithm (log base e) evaluated at `a`

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: f(a) where f(x) = ln(x)

    """
    return math.log(a)


# - exp
def exp(a: float) -> float:
    """Returns the value of the exponential function evaluated at `a`

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: f(a) where f(x) = e^x

    """
    return math.exp(a)


# - inv
def inv(a: float) -> float:
    """Finds the value of the reciprocal

    Args:
    ----
        a (float): a number

    Returns:
    -------
        float: the reciprocal, 1 / a

    """
    return 1.0 / a


# - log_back
def log_back(d: float, a: float) -> float:
    """Returns the derivative of log evaluated at d then multiplied by a

    Args:
    ----
        d (float): a number to evaluate the derivative of the logarithmic function at
        a (float): a number to multiply the derivative by

    Returns:
    -------
        float: the derivative of log evaluated at d then multiplied by a

    """
    return (1.0 / d) * a


# - inv_back
def inv_back(d: float, a: float) -> float:
    """Returns the derivative of inv evaluated at d then multiplied by a

    Args:
    ----
        d (float): a number to evaluate the derivative of the reciprocal function at
        a (float): a number to multiply the derivative by

    Returns:
    -------
        float: the derivative of inv evaluated at d then multiplied by a

    """
    return -math.pow(d, -2) * a


# - relu_back
def relu_back(d: float, a: float) -> float:
    """Returns the derivative of relu evaluated at d then multiplied by a

    Args:
    ----
        d (float): a number to evaluate the derivative of the retified linear unit function at
        a (float): a number to multiply the derivative by

    Returns:
    -------
        float: the derivative of relu evaluated at d then multiplied by a

    """
    if d <= 0:
        return 0.0
    else:
        return a


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Generator]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        func (Callable): a function to apply to each value of the iterable

    Returns:
    -------
        Callable: A new function that will apply `func` to every element of an iterable

    """

    def inner(collection: Iterable[float]) -> Generator:
        return (func(element) for element in collection)

    return inner


# - zipWith
def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Generator]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
        func (Callable): a function that will combine elements from two iterables

    Returns:
    -------
        Callable: A new function that will combine elements from two iterables using `func`

    """

    def inner(
        collection_a: Iterable[float], collection_b: Iterable[float]
    ) -> Generator:
        return (func(*element) for element in zip(collection_a, collection_b))

    return inner


# - reduce
def reduce(func: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        func (Callable): a function that will reduce all elements in an iterable to a single value

    Returns:
    -------
        Callable: A new function that will reduce elements from an iterable to a single value using `func`

    """

    def inner(collection: Iterable[float]) -> float:
        current = None
        for element in collection:
            if current is None:
                current = element
            else:
                current = func(current, element)

        if current is None:
            raise ValueError("No values to reduce")
        return current

    return inner


#
# Use these to implement
# - negList : negate a list
def negList(ls: list[float]) -> list[float]:
    """Negate all elements in a list using `map`

    Args:
    ----
        ls (list): a list of floats

    Returns:
    -------
        list: a list with all values of `ls` negated

    """
    return list(map(neg)(ls))


# - addLists : add two lists together
def addLists(ls1: list[float], ls2: list[float]) -> list[float]:
    """Add corresponding elements from two lists using `zipWith`

    Args:
    ----
        ls1 (list): a list of floats
        ls2 (list): a list of floats

    Returns:
    -------
        list: a list with values of ls1 and ls2 added together

    """
    if len(ls1) != len(ls2):
        raise ValueError("Lists must be the same length")

    return list(zipWith(add)(ls1, ls2))


# - sum: sum lists
def sum(ls: list[float]) -> float:
    """Sum all elements in a list using `reduce`

    Args:
    ----
        ls (list): a list of floats

    Returns:
    -------
        float: sum of all elements in list

    """
    if len(ls) == 0:
        return 0.0
    return reduce(add)(ls)


# - prod: take the product of lists
def prod(ls: list[float] | Sequence[float]) -> float:
    """Calculate the product of all elements in a list using `reduce`

    Args:
    ----
        ls (list): a list of floats

    Returns:
    -------
        float: product of all elements in list

    """
    return reduce(mul)(ls)
