from src.deep_learning.layer import Layer
from src.deep_learning.value import Value

class MultiLayerPerceptron:
  def __init__(
      self,
      layers: list[Layer],
  ):
    self.layers = layers

    assert len(self.layers) > 0
    for i in range(0, len(self.layers) - 1):
      assert len(self.layers[i].neurons) == len(self.layers[i+1].neurons[0].weights), f'{len(self.layers[i].neurons)=} {len(self.layers[i+1].neurons[0].weights)=}'

  def forward(
      self,
      inputs: list[float | Value],
  ) -> list[Value]:
    assert len(inputs) == len(self.layers[0].neurons[0].weights)

    current_input: list[float | Value] = inputs
    for i in range(0, len(self.layers) - 1):
      current_input = self.layers[i].forward(current_input)
    return self.layers[-1].forward(current_input)
