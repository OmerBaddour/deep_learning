from src.deep_learning.neuron import Neuron
import pytest

def test_basic():
  neuron = Neuron(
      weights=[1.0, 2.0, 3.0],
      bias=5.0,
  )
  forward_result = neuron.forward(
      inputs=[1.0, 2.0, 3.0],
  )
  assert forward_result == (1.0 ** 2 + 2.0 ** 2 + 3.0 ** 2 + 5.0)
