import math
import random

import pytest

from src.deep_learning.layer import Layer
from src.deep_learning.layer import get_random_layer
from src.deep_learning.neuron import Neuron
from src.deep_learning.op import TANH
from src.deep_learning.value import Value


@pytest.fixture
def simple_layer() -> Layer:
  # Weights/biases are integers (not realistic init values) so the expected
  # dot products are checkable by eye. Sensible init lives in get_random_layer.
  return Layer(
      neurons=[
          Neuron(weights=[1.0, 2.0, 3.0], bias=5.0),
          Neuron(weights=[3.0, 1.0, 2.0], bias=6.0),
          Neuron(weights=[2.0, 3.0, 1.0], bias=-7.0),
      ],
  )


def data(values: list[Value]) -> list[float]:
  '''forward() returns Values; tests care about the underlying floats.'''
  return [v.data for v in values]


# --- construction invariants ---

def test_empty_neurons_rejected():
  with pytest.raises(AssertionError):
    Layer(neurons=[])


def test_mismatched_weight_counts_rejected():
  with pytest.raises(AssertionError):
    Layer(
        neurons=[
            Neuron(weights=[1.0, 2.0], bias=0.0),
            Neuron(weights=[1.0, 2.0, 3.0], bias=0.0),
        ],
    )


def test_num_neuron_weights_inferred(simple_layer: Layer):
  assert simple_layer.num_neuron_weights == 3


# --- forward: shape contract ---

def test_forward_returns_one_value_per_neuron(simple_layer: Layer):
  outputs = simple_layer.forward([1.0, 2.0, 3.0])
  assert len(outputs) == 3
  assert all(isinstance(o, Value) for o in outputs)


def test_forward_rejects_wrong_input_length(simple_layer: Layer):
  with pytest.raises(AssertionError):
    simple_layer.forward([1.0, 2.0])  # layer expects 3 inputs


# --- forward: numerics ---

def test_zero_input_yields_tanh_of_bias(simple_layer: Layer):
  # w . 0 + b == b, so each output is tanh(bias)
  outputs = simple_layer.forward([0.0, 0.0, 0.0])
  assert data(outputs) == pytest.approx([
      TANH.forward([5.0]),
      TANH.forward([6.0]),
      TANH.forward([-7.0]),
  ])


def test_known_dot_products(simple_layer: Layer):
  outputs = simple_layer.forward([1.0, 2.0, 3.0])
  assert data(outputs) == pytest.approx([
      TANH.forward([1 + 4 + 9 + 5.0]),    # 19
      TANH.forward([3 + 2 + 6 + 6.0]),    # 17
      TANH.forward([2 + 6 + 3 - 7.0]),    # 4
  ])


def test_outputs_are_bounded_by_tanh_range(simple_layer: Layer):
  # Example of the saturated regime training wants to AVOID (gradients ~0),
  # here just to confirm tanh saturates rather than overflowing.
  outputs = simple_layer.forward([1e6, 1e6, 1e6])
  for o in outputs:
    assert -1.0 <= o.data <= 1.0
    assert o.data == pytest.approx(1.0)  # all weights positive -> +1


def test_accepts_value_inputs(simple_layer: Layer):
  # forward should treat raw floats and Value inputs identically
  floats = simple_layer.forward([1.0, 2.0, 3.0])
  values = simple_layer.forward([Value(1.0), Value(2.0), Value(3.0)])
  assert data(values) == pytest.approx(data(floats))


# --- get_random_layer ---

def test_random_layer_dimensions():
  layer = get_random_layer(num_neurons=4, num_weights_per_neuron=5)
  assert isinstance(layer, Layer)
  assert len(layer.neurons) == 4
  assert layer.num_neuron_weights == 5
  assert all(len(n.weights) == 5 for n in layer.neurons)


def test_random_layer_biases_zero():
  layer = get_random_layer(num_neurons=3, num_weights_per_neuron=4)
  assert all(n.bias == 0.0 for n in layer.neurons)


def test_random_layer_is_deterministic_under_seed():
  random.seed(0)
  a = get_random_layer(num_neurons=2, num_weights_per_neuron=3)
  random.seed(0)
  b = get_random_layer(num_neurons=2, num_weights_per_neuron=3)
  a_weights = [n.weights for n in a.neurons]
  b_weights = [n.weights for n in b.neurons]
  assert a_weights == b_weights


def test_random_layer_weight_std_matches_xavier():
  # This is the test that actually pins down "weights should be small": small
  # std keeps pre-activations near 0, where tanh has gradient (avoids saturation).
  # Xavier init: std == sqrt(1 / fan_in). Check empirically over many weights.
  fan_in = 50
  random.seed(123)
  layer = get_random_layer(num_neurons=200, num_weights_per_neuron=fan_in)
  weights = [w for n in layer.neurons for w in n.weights]

  mean = sum(weights) / len(weights)
  variance = sum((w - mean) ** 2 for w in weights) / len(weights)
  std = math.sqrt(variance)

  expected_std = math.sqrt(1 / fan_in)
  assert mean == pytest.approx(0.0, abs=0.02)
  assert std == pytest.approx(expected_std, rel=0.1)
