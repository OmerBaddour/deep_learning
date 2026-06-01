from src.deep_learning.value import Value
from src.deep_learning.value import Op
from src.deep_learning.value import draw
import pytest

@pytest.fixture
def simple_value_graph() -> Value:
  a = Value(2, label='a')
  b = Value(3, label='b')
  c = Value(label='c', children=[a, b], op=Op.PLUS)

  d = Value(5, label='d')
  e = Value(6, label='e')
  f = Value(label='f', children=[d, e], op=Op.MULTIPLY)

  L = Value(label='L', children=[c, f], op=Op.PLUS)
  return L

def test_basic():
  assert Value(1).data == 1.0

def test_forward(simple_value_graph: Value):
  assert simple_value_graph.forward() == 35.0

def test_draw(simple_value_graph: Value):
  draw(simple_value_graph)

def test_backward():
  pass
