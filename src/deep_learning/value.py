from __future__ import annotations
from enum import StrEnum
from src.deep_learning.util import is_numeric
from graphviz import Digraph

class Op(StrEnum):
  PLUS = '+'
  MULTIPLY = '*'

class Value:
  def __init__(
      self,
      data: float | int = 0.0,
      label: str = '',
      op: Op | None = None,
      children: list[Value] | None = None,
      gradient: float = 0.0,
  ):
    if is_numeric(data):
      self.data = float(data)
    else:
      raise ValueError()

    self.label = label
    self.op = op
    self.children = [] if children is None else children
    self.gradient = gradient

  def __add__(self, other: float | int) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented

    return Value(
        data=self.data + other.data,
        op=Op.PLUS,
        children=[self, other],
    )

  def __mult__(self, other: float | int) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented
    
    return Value(
        data=self.data * other.data,
        op=Op.MULTIPLY,
        children=[self, other],
    )
  
  def forward(self) -> float:
    if len(self.children) == 0:
      return self.data
    else:
      return eval(self.op.value.join(str(value.forward()) for value in self.children))

  def backward(self) -> None:
    pass


def draw(root: Value) -> Digraph:
  dot = Digraph(graph_attr={'rankdir': 'LR'})
  seen = set()
  def build(v: Value):
    if id(v) in seen:
      return
    seen.add(id(v))
    dot.node(str(id(v)), f'{v.label} | data {v.data:.4f} | gradient {v.gradient:.4f}', shape='record')
    if v.op:
      dot.node(str(id(v)) + v.op, v.op)
      dot.edge(str(id(v)) + v.op, str(id(v)))
    for child in v.children:
      build(child)
      dot.edge(str(id(child)), str(id(v)) + v.op)
  build(root)
  return dot
