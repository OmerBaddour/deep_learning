from src.deep_learning.value import Value
from src.deep_learning.op import MULTIPLY
from src.deep_learning.op import PLUS
from src.deep_learning.op import TANH
import pytest


@pytest.fixture
def simple_value_graph() -> Value:
  a = Value(2, label='a')
  b = Value(3, label='b')
  c = Value(label='c', children=[a, b], op=PLUS)

  d = Value(5, label='d')
  e = Value(6, label='e')
  f = Value(label='f', children=[d, e], op=MULTIPLY)

  L = Value(label='L', children=[c, f], op=PLUS)
  return L


def test_basic():
  assert Value(1).data == 1.0
  
  x = Value(1) + 2
  x.forward()
  assert isinstance(x, Value) and x.data == 3.0 and x.op == PLUS
  
  x = Value(1) * 2
  x.forward()
  assert isinstance(x, Value) and x.data == 2.0 and x.op == MULTIPLY


def test_forward(simple_value_graph: Value):
  L = simple_value_graph

  f = L.children[1]
  e = f.children[1]
  d = f.children[0]
  
  c = L.children[0]
  b = c.children[1]
  a = c.children[0]

  assert L.forward().data == 35.0

  assert a.data == 2.0
  assert b.data == 3.0
  assert c.data == a.data + b.data
  
  assert d.data == 5.0
  assert e.data == 6.0
  assert f.data == d.data * e.data

  assert L.data == c.data + f.data


def test_backward(simple_value_graph: Value):
  L = simple_value_graph
  L.forward()

  L.gradient = 1.0
  L.backward()

  f = L.children[1]
  e = f.children[1]
  d = f.children[0]
  
  c = L.children[0]
  b = c.children[1]
  a = c.children[0]

  assert L.gradient == 1.0
  
  assert c.gradient == 1.0
  assert f.gradient == 1.0
  
  assert a.gradient == 1.0
  assert b.gradient == 1.0

  assert d.gradient == e.data
  assert e.gradient == d.data


def test_backward_multi_reference():
  x = Value(3.0)
  y: Value = x * x
  y.forward()

  y.gradient = 1.0
  y.backward()

  assert x.gradient == x.data * 2


def test_tanh(simple_value_graph: Value):
  x = simple_value_graph
  L = Value(
      label='tanh',
      op=TANH,
      children=[x]
  )
  x_forward = x.forward()
  assert L.forward().data == TANH.forward([x_forward.data])
