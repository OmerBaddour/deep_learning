from src.deep_learning.op import MULTIPLY
from src.deep_learning.op import PLUS
from src.deep_learning.op import TANH
from src.deep_learning.value import Value

class Neuron:
  def __init__(
      self,
      weights: list[float],
      bias: float,
  ):
    self.weights = weights
    self.bias = bias

    self.weight_values = [Value(weight) for weight in self.weights]
    self.bias_value = Value(self.bias)

    self._graph: Value = None
  
  def forward(
      self,
      inputs: list[float],
  ) -> float:
    assert len(inputs) == len(self.weights)
    input_values = [Value(input) for input in inputs]

    # construct graph out of Values, then call Value.forward()
    multiply_values: list[Value] = []
    for i, (input_value, weight_value) in enumerate(zip(input_values, self.weight_values), start=1):
      multiply_values.append(
          Value(
              label=f'w{i}x{i}',
              op=MULTIPLY,
              children=[input_value, weight_value],
          ),
      )

    sum_value = Value(
        label='sum',
        op=PLUS,
        children=multiply_values + [self.bias_value],
    )

    tanh_value = Value(
        label='tanh',
        op=TANH,
        children=[sum_value]
    )

    self._graph = tanh_value
    return self._graph.forward()

  def backward(self) -> None:
    assert self._graph is not None
    self._graph.backward()
