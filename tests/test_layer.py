from src.deep_learning.layer import Layer
from src.deep_learning.neuron import Neuron
from src.deep_learning.op import TANH
import pytest


@pytest.fixture
def simple_layer() -> Layer:
  return Layer(
      neurons=[
          Neuron(
              weights=[1.0, 2.0, 3.0],
              bias=5.0,
          ),
          Neuron(
              weights=[3.0, 1.0, 2.0],
              bias=6.0,
          ),
          Neuron(
              weights=[2.0, 3.0, 1.0],
              bias=-7.0,
          ),
      ],
  )

def test_zeros(simple_layer: Layer):
  # input zeros outputs tanh of biases
  assert simple_layer.forward(
      [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
      ],
  ) == [
      TANH.forward([5.0]),
      TANH.forward([6.0]),
      TANH.forward([-7.0]),
  ]

def test_basic(simple_layer: Layer):
  assert simple_layer.forward(
      [
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
      ],
  ) == [
      TANH.forward([14.0 + 5.0]),
      TANH.forward([11.0 + 6.0]),
      TANH.forward([11.0 + -7.0]),
  ]
