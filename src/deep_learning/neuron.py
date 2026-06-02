from src.deep_learning.value import Op
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
              op=Op.MULTIPLY,
              children=[input_value, weight_value],
          ),
      )

    sum_value = Value(
        label='sum',
        op=Op.PLUS,
        children=multiply_values + [self.bias_value],
    )

    # TODO: add activation
    
    return sum_value.forward()
