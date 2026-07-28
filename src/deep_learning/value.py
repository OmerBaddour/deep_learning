from __future__ import annotations
from typing import Any
from deep_learning.op import Op
from deep_learning.op import Power
from deep_learning.op import DIVIDE
from deep_learning.op import EXPONENTIATE
from deep_learning.op import PLUS
from deep_learning.op import MINUS
from deep_learning.op import MULTIPLY


def _is_numeric(x: Any) -> bool:
  try:
    float(x)
    return True
  except:
    return False

'''
TODO: can I set `data: float | int | None = None,`?
if so I could have `Value.__init__()` do:
```
if data is None:
  self.data = data
elif _is_numeric(data):
  self.data = float(data)
else:
  raise ValueError()
```
and could have `Value.forward()` do:
```
if len(self.children) == 0:
  assert self.data is not None
```
'''

class Value:
  def __init__(
      self,
      data: float | int = 0.0,
      label: str = '',
      op: Op | None = None,
      children: list[Value] | None = None,
  ):
    if _is_numeric(data):
      self.data = float(data)
    else:
      raise ValueError()

    self.label = label
    self.op = op
    self.children = [] if children is None else children
    self.gradient = 0.0

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.__dict__})'

  def __add__(self, other: float | int | Value) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif _is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented

    return Value(
        op=PLUS,
        children=[self, other_value],
    )
  
  def __radd__(self, other: float | int) -> Value:
    return self + other
  
  def __sub__(self, other: float | int | Value) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif _is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented

    return Value(
        op=MINUS,
        children=[self, other_value],
    )
  
  def __rsub__(self, other: float | int) -> Value:
    return self - other

  def __mul__(self, other: float | int | Value) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif _is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented
    
    return Value(
        op=MULTIPLY,
        children=[self, other_value],
    )
  
  def __rmul__(self, other: float | int) -> Value:
    return self * other
  
  def __truediv__(self, other: float | int | Value) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif _is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented
    
    return Value(
        op=DIVIDE,
        children=[self, other_value]
    )
  
  def __rtruediv__(self, other: float | int) -> Value:
    return self / other
  
  def __neg__(self) -> Value:
    return self * -1

  def __pow__(self, other: float | int | Value) -> Value:
    if isinstance(other, Value):
      return Value(
          op=EXPONENTIATE,
          children=[self, other],
      )
    elif _is_numeric(other):
      return Value(
          op=Power(exponent=other),
          children=[self],
      )
    else:
      return NotImplemented
  
  def forward(self) -> Value:
    visited: set[Value] = set()

    def _traverse(node: Value, visited: set[Value]) -> None:
      if node in visited:
        return
      elif len(node.children) == 0:
        visited.add(node)
        return
      else:
        for child in node.children:
          _traverse(child, visited)
        node.data = node.op.forward([child.data for child in node.children])
        visited.add(node)
    
    _traverse(self, visited)
    return self

  def backward(self) -> None:
    # topologically sort graph
    topologically_sorted_graph: list[Value] = []
    visited: set[Value] = set()
    
    def _traverse(
        node: Value,
        topologically_sorted_graph: list[Value],
        visited: set[Value],
    ) -> None:
      # NOTE: assume acyclic for simplicity
      if node in visited:
        return
      elif len(node.children) == 0:
        visited.add(node)
        topologically_sorted_graph.append(node)
      else:
        for child in node.children:
          _traverse(child, topologically_sorted_graph, visited)
        visited.add(node)
        topologically_sorted_graph.append(node)
    _traverse(self, topologically_sorted_graph, visited)

    for node in reversed(topologically_sorted_graph):
      '''
      Limit definition of derivative: https://en.wikipedia.org/wiki/Derivative#As_a_limit
      Let y = f(a), then dy/da = lim h -> 0 of (f(a + h) - f(a)) / f(h)
      We can algebraically derive that
      - if f(a) = a + <stuff>, dy/da = 1.0
      - if f(a) = a * <number> + <stuff>, dy/da = <number>
      
      The chain rule tells us how to compute the derivative of z = f(y), where y = g(x): https://en.wikipedia.org/wiki/Chain_rule
      dz/dx = dz/dy * dy/dx
      We call dy/dx the "local derivative"

      Final note: in the multivariate case: https://en.wikipedia.org/wiki/Chain_rule#Example:_arithmetic_operations
      we do += to the gradient
      '''
      if len(node.children) > 0:
        children_local_derivatives = node.op.backward([child.data for child in node.children])
        zipped: list[tuple[Value, float]] = zip(node.children, children_local_derivatives, strict=True)
        for child, local_derivative in zipped:
          child.gradient += local_derivative * node.gradient
