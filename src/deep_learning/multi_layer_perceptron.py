from src.deep_learning.layer import Layer

class MultiLayerPerceptron:
  def __init__(
      self,
      layers: list[Layer],
  ):
    self.layers = layers

    assert len(self.layers) > 0
    for i in range(0, len(self.layers) - 1):
      assert len(self.layers[i].neurons) == len(self.layers[i+1].neurons[0].weights), f'{len(self.layers[i].neurons)=} {len(self.layers[i+1].neurons[0].weights)=}'
    assert len(self.layers[-1].neurons) == 1

  def forward(
      self,
      inputs: list[float],
  ) -> float:
    for input in inputs:
      assert len(input) == len(self.layers[0].neurons[0].weights)
    
    current_input: list[float] = inputs
    current_output: list[float] = None
    for i in range(0, len(self.layers) - 1):
      current_output = self.layers[i].forward(current_input)
      current_input = [current_output for _ in range(len(self.layers[i+1].neurons))]
    self.layers[-1].forward(current_input)
    return current_output

  def backward(self) -> None:
    self.layers[-1].backward()
