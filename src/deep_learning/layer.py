from typing import Callable
import random
from src.deep_learning.neuron import Neuron
from src.deep_learning.op import Op
from src.deep_learning.value import Value
from src.deep_learning.util import random_uniform

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

def get_layer(
    num_neurons: int,
    num_weights_per_neuron: int,
    activation: Op | None = None,
    fn_weight_initializer: Callable[[], float] | None = None
) -> Layer:
  default_fn_weight_initializer = lambda: random_uniform(-1, 1)
  fn_weight_initializer = fn_weight_initializer or default_fn_weight_initializer

  neurons: list[Neuron] = []
  for _ in range(num_neurons):
    neurons.append(
        Neuron(
            weights=[fn_weight_initializer() for _ in range(num_weights_per_neuron)],
            bias=0.0,
            activation=activation,
        )
    )
  return Layer(neurons)