import math
import random
from src.deep_learning.neuron import Neuron
from src.deep_learning.value import Value

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
      inputs: list[float | Value],
  ) -> list[Value]:
    assert len(inputs) == self.num_neuron_weights

    outputs: list[Value] = []
    for neuron in self.neurons:
      outputs.append(neuron.forward(inputs))
    return outputs

def get_random_layer(
    num_neurons: int,
    num_weights_per_neuron: int,
) -> Layer:
  std = math.sqrt(1 / num_weights_per_neuron)  # Xavier, suited to tanh
  neurons: list[Neuron] = []
  for _ in range(num_neurons):
    neurons.append(
        Neuron(
            weights=[random.gauss(0, std) for _ in range(num_weights_per_neuron)],
            bias=0.0,
        )
    )
  return Layer(neurons)