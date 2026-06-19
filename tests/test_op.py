from src.deep_learning.op import DIVIDE
from src.deep_learning.op import Power
from src.deep_learning.op import MULTIPLY
from src.deep_learning.op import Op
import pytest


def _all_ops() -> list[Op]:
  all_ops: list[Op] = []
  # add those with __init__() parameters
  all_ops.extend([Power(2)])
  # add others
  all_ops.extend([subclass() for subclass in Op.__subclasses__() if subclass != Power])
  return all_ops


@pytest.mark.parametrize('op', _all_ops(), ids=lambda op: op.to_string())
def test_backward_length_matches_inputs(op: Op):
  '''backward must return one local derivative per input, whatever the op's arity.'''
  # Variadic ops (arity None) accept any count; exercise a couple of them.
  # 2.0 keeps every op in its domain (log needs >0 and base != 1, etc.).
  arities = [2, 3] if op.arity is None else [op.arity]
  for arity in arities:
    inputs = [2.0] * arity
    assert len(op.backward(inputs)) == arity


def test_multiply_backward_simple():
  assert MULTIPLY.backward([2.0, 3.0]) == [3.0, 2.0]


def test_multiply_backward_no_zeros():
  '''Local derivative of a product is the product of the other factors.'''
  assert MULTIPLY.backward([2.0, 3.0, 4.0]) == [12.0, 8.0, 6.0]


def test_multiply_backward_single_zero():
  '''A single zero factor zeroes every OTHER position, but not its own.'''
  assert MULTIPLY.backward([2.0, 0.0, 4.0]) == [0.0, 8.0, 0.0]


def test_multiply_backward_two_zeros():
  '''Two zero factors make every position 0 -- the case division can't handle.'''
  assert MULTIPLY.backward([0.0, 0.0, 4.0]) == [0.0, 0.0, 0.0]


def test_divide():
  a = 10.0
  b = 5.0
  assert DIVIDE.forward([a, b]) == a / b
  # d/da (a/b) = 1/b ; d/db (a/b) = -a/b**2
  assert DIVIDE.backward([a, b]) == [1 / b, -a / b ** 2]
