import random
from src.deep_learning.neuron import Neuron
from src.deep_learning.op import Op
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
  
  def build_graph(
      self,
      inputs: list[float | Value],
  ) -> list[Value]:
    assert len(inputs) == self.num_neuron_weights

    outputs: list[Value] = []
    for neuron in self.neurons:
      outputs.append(neuron.build_graph(inputs))
    return outputs

def get_random_layer(
    num_neurons: int,
    num_weights_per_neuron: int,
    activation: Op | None = None,
) -> Layer:
  neurons: list[Neuron] = []
  for _ in range(num_neurons):
    neurons.append(
        Neuron(
            weights=[random.uniform(-1, 1) for _ in range(num_weights_per_neuron)],
            bias=0.0,
            activation=activation,
        )
    )
  return Layer(neurons)