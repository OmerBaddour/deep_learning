from src.deep_learning.value import Value
from src.deep_learning.op import MULTIPLY
from src.deep_learning.op import PLUS
from src.deep_learning.util import draw
from src.deep_learning.util import sum_mean_squared_error
from src.deep_learning.util import softmax
from src.deep_learning.util import cross_entropy
import pytest


@pytest.fixture
def simple_value_list() -> list[Value]:
  return [
      Value(1.0, label='a'),
      Value(2.0, label='b'),
      Value(1.0, label='c'),
  ]

def test_draw() -> None:
  a = Value(2, label='a')
  b = Value(3, label='b')
  c = Value(label='c', children=[a, b], op=PLUS)

  d = Value(5, label='d')
  e = Value(6, label='e')
  f = Value(label='f', children=[d, e], op=MULTIPLY)

  L = Value(label='L', children=[c, f], op=PLUS)
  draw(L)


def test_sum_mean_squared_error(simple_value_list: list[Value]) -> None:
  outputs = [value.data + 1 for value in simple_value_list]
  predicted_outputs = simple_value_list
  error_value = sum_mean_squared_error(
      outputs,
      predicted_outputs,
  )
  assert error_value.data == 1.0


def test_softmax(simple_value_list: list[Value]) -> None:
  inputs = simple_value_list
  outputs = softmax(inputs)
  final = Value(
      label=PLUS.to_string(),
      op=PLUS,
      children=outputs,
  )
  final.forward()

  assert len(outputs) == len(inputs)
  assert sum([output.data for output in outputs]) == pytest.approx(1)
  for output in outputs:
    assert 0 <= output.data <= 1
  

def test_cross_entropy(simple_value_list: list[Value]) -> None:
  # self cross entropy should be the minimum
  distribution = softmax(simple_value_list)
  for value in distribution:
    value.forward()

  self_cross_entropy = cross_entropy(
      distribution=[value.data for value in distribution],
      predicted_distribution=distribution,
  )

  other_distribution = softmax([Value(value.data ** 2) for value in simple_value_list])
  for value in other_distribution:
    value.forward()
  other_cross_entropy = cross_entropy(
      distribution=[value.data for value in distribution],
      predicted_distribution=other_distribution,
  )
  assert self_cross_entropy.data < other_cross_entropy.data
