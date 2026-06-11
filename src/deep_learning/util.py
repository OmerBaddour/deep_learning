from src.deep_learning.value import Value
import math
from src.deep_learning.op import DIVIDE
from src.deep_learning.op import EXPONENTIATE
from src.deep_learning.op import LOGARITHM
from src.deep_learning.op import PLUS
from graphviz import Digraph


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


def sum_mean_squared_error(
    outputs: list[float],
    predicted_outputs: list[Value],
) -> Value:
  assert len(outputs) == len(predicted_outputs)
  sum_squared_error_value = Value(0.0)
  for output, predicted_output in zip(outputs, predicted_outputs):
    output_value = Value(output)
    predicted_output_value = predicted_output
    error_value = output_value - predicted_output_value
    squared_error_value = error_value ** 2
    sum_squared_error_value += squared_error_value
  mean_sum_squared_error_value = sum_squared_error_value / len(predicted_outputs)
  return mean_sum_squared_error_value


def softmax(input_values: list[Value]) -> list[Value]:
  softmax_numerators: list[Value] = []
  softmax_layer: list[Value] = []
  e_value = Value(math.e, label='e')
  for input in input_values:
    softmax_numerators.append(
        Value(
            label=EXPONENTIATE.to_string(),
            op=EXPONENTIATE,
            children=[e_value, input],
        ),
    )
  softmax_denominator = Value(
      label=PLUS.to_string(),
      op=PLUS,
      children=softmax_numerators,
  )

  for softmax_numerator in softmax_numerators:
    softmax_layer.append(
        Value(
            label=DIVIDE.to_string(),
            op=DIVIDE,
            children=[softmax_numerator, softmax_denominator]
        )
    )
  return softmax_layer


def cross_entropy(
    distribution: list[float],
    predicted_distribution: list[Value],
 ) -> Value:
  assert len(distribution) == len(predicted_distribution)

  terms: list[Value] = []
  epsilon = 1e-16  # adding prevents log(0) error without disturbing the distribution
  for event, predicted_event in zip([Value(event) for event in distribution], predicted_distribution):
    event.data += epsilon
    predicted_event.data += epsilon
    terms.append(event * LOGARITHM.forward([predicted_event.data, 2]))

  return -Value(
      label=PLUS.to_string(),
      op=PLUS,
      children=terms,
  ).forward()
