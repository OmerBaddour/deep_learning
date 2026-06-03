from src.deep_learning.neuron import Neuron

class Layer:
  def __init__(
      self,
      neurons: list[Neuron],
  ):
    self.neurons = neurons
    
    assert len(neurons) > 0
    self.num_neuron_weights = len(neurons[0].weights)
    
    for neuron in self.neurons:
      assert len(neuron.weights) == self.num_neuron_weights
  
  def forward(
      self,
      inputs: list[list[float]],
  ) -> list[float]:
    for input in inputs:
      assert len(input) == self.num_neuron_weights
    
    outputs: list[float] = []
    for neuron, input in zip(self.neurons, inputs):
      outputs.append(neuron.forward(input))
    return outputs

  def backward(self) -> None:
    for neuron in self.neurons:
      neuron.backward()
