from __future__ import annotations
from src.deep_learning.op import Op
from src.deep_learning.op import PLUS
from src.deep_learning.op import MULTIPLY
from src.deep_learning.op import TANH
from src.deep_learning.util import is_numeric
from graphviz import Digraph

class Value:
  def __init__(
      self,
      data: float | int = 0.0,
      label: str = '',
      op: Op | None = None,
      children: list[Value] | None = None,
  ):
    if is_numeric(data):
      self.data = float(data)
    else:
      raise ValueError()

    self.label = label
    self.op = op
    self.children = [] if children is None else children
    self.gradient = 0.0

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.__dict__})'

  def __add__(self, other: float | int) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented

    return Value(
        data=self.data + other_value.data,
        op=PLUS,
        children=[self, other_value],
    )

  def __mul__(self, other: float | int) -> Value:
    other_value = None
    if isinstance(other, Value):
      other_value = other
    elif is_numeric(other):
      other_value = Value(data=float(other))
    if other_value is None:
      return NotImplemented
    
    return Value(
        data=self.data * other_value.data,
        op=MULTIPLY,
        children=[self, other_value],
    )
  
  def forward(self) -> Value:
    if len(self.children) == 0:
      return self
    else:
      for child in self.children:
        child.forward()
      self.data = self.op.forward([child.data for child in self.children])
      return self

  def backward(self) -> None:
    # topologically sort graph
    topologically_sorted_graph: list[Value] = []
    
    def do_topological_sort(node: Value) -> None:
      # NOTE: assume acyclic for simplicity
      if len(node.children) == 0:
        if node not in topologically_sorted_graph:
          topologically_sorted_graph.append(node)
      else:
        for child in node.children:
          do_topological_sort(child)
        if node not in topologically_sorted_graph:
          topologically_sorted_graph.append(node)
    do_topological_sort(self)

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
        zipped: list[tuple[Value, float]] = zip(node.children, children_local_derivatives)
        for child, local_derivative in zipped:
          child.gradient += local_derivative * node.gradient

def draw(root: Value) -> Digraph:
  dot = Digraph(graph_attr={'rankdir': 'LR'})
  seen = set()
  def build(v: Value):
    if id(v) in seen:
      return
    seen.add(id(v))
    dot.node(str(id(v)), f'{v.label} | data {v.data:.4f} | gradient {v.gradient:.4f}', shape='record')
    if v.op:
      dot.node(str(id(v)) + v.op.to_string(), v.op.to_string())
      dot.edge(str(id(v)) + v.op.to_string(), str(id(v)))
    for child in v.children:
      build(child)
      dot.edge(str(id(child)), str(id(v)) + v.op.to_string())
  build(root)
  return dot
