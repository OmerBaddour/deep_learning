import random
from deep_learning.value import Value
import math
from deep_learning.op import DIVIDE
from deep_learning.op import EXPONENTIATE
from deep_learning.op import LOGARITHM
from deep_learning.op import PLUS
from graphviz import Digraph


def draw(root: Value) -> Digraph:
  '''
  Draw graph of Value nodes
  '''
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


def random_uniform(lower: float, upper: float) -> float:
  '''
  Random uniform float between specified bounds
  '''
  return random.uniform(lower, upper)


def xavier_uniform(num_inputs: float, num_outputs: float) -> float:
  '''
  Xavier uniform float given num_inputs and num_outputs
  '''
  bound = math.sqrt(6.0 / (num_inputs + num_outputs))
  return random_uniform(-bound, bound)


def zero_all_gradients(root: Value) -> None:
  '''
  Zero all gradients of all Values recursing from the root
  '''
  visited: set[Value] = set()
  
  def _traverse(root: Value, visited: set[Value]) -> None:
    if root in visited:
      return
    root.gradient = 0.0
    visited.add(root)
    for child in root.children:
      _traverse(child, visited)
  
  _traverse(root, visited)


def mean_squared_error(
    outputs: list[Value],
    predicted_outputs: list[Value],
) -> Value:
  '''
  Build Value graph for mean squared error
  '''
  assert len(outputs) == len(predicted_outputs)
  sum_squared_error_value = Value(0.0)
  for output, predicted_output in zip(outputs, predicted_outputs, strict=True):
    error_value = output - predicted_output
    squared_error_value = error_value ** 2
    sum_squared_error_value += squared_error_value
  mean_sum_squared_error_value = sum_squared_error_value / len(predicted_outputs)
  return mean_sum_squared_error_value


def softmax(input_values: list[Value]) -> list[Value]:
  '''
  Build Value graph for softmax
  '''
  softmax_numerators: list[Value] = []
  softmax_layer: list[Value] = []
  e_value = Value(math.e, label='e')
  for input in input_values:
    softmax_numerators.append(
        Value(
            op=EXPONENTIATE,
            children=[e_value, input],
        ),
    )
  softmax_denominator = Value(
      op=PLUS,
      children=softmax_numerators,
  )

  for softmax_numerator in softmax_numerators:
    softmax_layer.append(
        Value(
            op=DIVIDE,
            children=[softmax_numerator, softmax_denominator]
        )
    )
  return softmax_layer


def cross_entropy(
    distribution: list[Value],
    predicted_distribution: list[Value],
 ) -> Value:
  '''
  Build Value graph for cross entropy
  '''
  assert len(distribution) == len(predicted_distribution)

  terms: list[Value] = []
  epsilon = 1e-16  # adding prevents log(0) error without disturbing the distribution
  log_base = Value(2)
  for event, predicted_event in zip(distribution, predicted_distribution, strict=True):
    event_plus_epsilon = event + epsilon
    predicted_event_plus_epsilon = predicted_event + epsilon
    terms.append(event_plus_epsilon * Value(op=LOGARITHM, children=[predicted_event_plus_epsilon, log_base]))

  return -Value(
      op=PLUS,
      children=terms,
  )
