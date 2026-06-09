from src.deep_learning.op import MULTIPLY
import pytest


def test_multiply_backward_no_zeros():
  '''Local derivative of a product is the product of the other factors.'''
  assert MULTIPLY.backward([2.0, 3.0, 4.0]) == [12.0, 8.0, 6.0]


def test_multiply_backward_single_zero():
  '''A single zero factor zeroes every OTHER position, but not its own.'''
  assert MULTIPLY.backward([2.0, 0.0, 4.0]) == [0.0, 8.0, 0.0]


def test_multiply_backward_two_zeros():
  '''Two zero factors make every position 0 -- the case division can't handle.'''
  assert MULTIPLY.backward([0.0, 0.0, 4.0]) == [0.0, 0.0, 0.0]
