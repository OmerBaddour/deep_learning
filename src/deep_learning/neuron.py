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
    assert len(self.weights) > 0
    self.bias = bias

    self.weight_values: list[Value] = []
    for i, weight in enumerate(self.weights):
      self.weight_values.append(Value(weight, label=f'w{i}'))
    self.bias_value = Value(self.bias, label='b')

  def forward(
      self,
      inputs: list[float | Value],
  ) -> Value:
    assert len(inputs) == len(self.weights)
    input_values: list[Value] = []
    for i, input in enumerate(inputs):
      label = f'x{i}'
      if isinstance(input, float):
        input_values.append(Value(input, label=label))
      elif isinstance(input, Value):
        if input.label == '':
          input.label = label
        input_values.append(input)
      else:
        raise TypeError('Unsupported type')

    # construct graph out of Values, then call Value.forward()
    multiply_values: list[Value] = []
    for i, (weight_value, input_value) in enumerate(zip(self.weight_values, input_values)):
      multiply_values.append(
          Value(
              label=f'w{i}x{i}',
              op=MULTIPLY,
              children=[weight_value, input_value],
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

    return tanh_value.forward()
