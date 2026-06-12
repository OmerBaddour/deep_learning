from src.deep_learning.neuron import Neuron
from src.deep_learning.op import TANH
import pytest


def test_forward_without_activation_is_weighted_sum():
  '''Forward pass: output is tanh(w . x + b).'''
  weights = [0.1, -0.2, 0.05]
  bias = 0.1
  inputs = [0.3, -0.4, 0.2]

  neuron = Neuron(weights=weights, bias=bias)
  value = neuron.build_graph(inputs=inputs)

  assert value.data == sum(w * x for w, x in zip(weights, inputs)) + bias


def test_forward_matches_tanh_of_preactivation():
  '''Forward pass: output is tanh(w . x + b).'''
  weights = [0.1, -0.2, 0.05]
  bias = 0.1
  inputs = [0.3, -0.4, 0.2]

  neuron = Neuron(weights=weights, bias=bias, activation=TANH)
  value = neuron.build_graph(inputs=inputs)

  preactivation = sum(w * x for w, x in zip(weights, inputs)) + bias
  assert value.data == pytest.approx(TANH.forward([preactivation]))


def test_backward_matches_finite_differences():
  '''
  Gold-standard autograd test: compare analytic gradients from backward()
  against numerically estimated gradients (central finite differences).

  Operating point is chosen near 0 so tanh is NOT saturated -- otherwise
  the true gradients really are ~0 and the test would prove nothing.
  '''
  weights = [0.1, -0.2, 0.05]
  bias = 0.1
  inputs = [0.3, -0.4, 0.2]
  h = 1e-6

  # Analytic gradients via backward().
  neuron = Neuron(weights=weights, bias=bias, activation=TANH)
  out = neuron.build_graph(inputs=inputs)
  out.forward()
  out.gradient = 1.0
  out.backward()
  analytic = [wv.gradient for wv in neuron.weight_values] + [neuron.bias_value.gradient]

  # Numeric gradients: rebuild the neuron with one parameter nudged by +/- h.
  def output_data(w: list[float], b: float) -> float:
    return Neuron(weights=w, bias=b, activation=TANH).build_graph(inputs=inputs).data

  numeric: list[float] = []
  for i in range(len(weights)):
    w_plus = weights.copy(); w_plus[i] += h
    w_minus = weights.copy(); w_minus[i] -= h
    numeric.append((output_data(w_plus, bias) - output_data(w_minus, bias)) / (2 * h))
  numeric.append((output_data(weights, bias + h) - output_data(weights, bias - h)) / (2 * h))

  for analytic_grad, numeric_grad in zip(analytic, numeric):
    assert analytic_grad == pytest.approx(numeric_grad, abs=1e-6)


def test_backward_seeds_root_gradient():
  '''The node backward() is called on keeps the gradient it was seeded with.'''
  neuron = Neuron(weights=[0.1, -0.2, 0.05], bias=0.1)
  out = neuron.build_graph(inputs=[0.3, -0.4, 0.2])
  out.gradient = 1.0
  out.backward()
  assert out.gradient == 1.0
