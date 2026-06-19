import random
from src.deep_learning.layer import get_layer
from src.deep_learning.multi_layer_perceptron import MultiLayerPerceptron
from src.deep_learning.op import TANH
from src.deep_learning.value import Value


def mse_loss(
    multi_layer_perceptron: MultiLayerPerceptron,
    inputs: list[list[float]],
    targets: list[list[float]],
) -> Value:
  '''Mean squared error over a dataset, as a Value graph (single-output MLP).'''
  sum_squared_difference = Value(0.0)
  for input_row, target_row in zip(inputs, targets, strict=True):
    predicted = multi_layer_perceptron.build_graph(input_row)[0]
    difference = Value(target_row[0]) + (predicted * -1)
    sum_squared_difference += difference ** 2
  loss = sum_squared_difference * (1 / len(inputs))
  loss.label = 'loss'
  return loss


def gradient_descent_step(
    multi_layer_perceptron: MultiLayerPerceptron,
    inputs: list[list[float]],
    targets: list[list[float]],
    alpha: float,
) -> float:
  '''One full-batch step: returns the loss BEFORE the parameter update.'''
  loss = mse_loss(multi_layer_perceptron, inputs, targets)
  loss.forward()

  # zero gradients
  for layer in multi_layer_perceptron.layers:
    for neuron in layer.neurons:
      for weight_value in neuron.weight_values:
        weight_value.gradient = 0.0
      neuron.bias_value.gradient = 0.0

  loss.gradient = 1.0
  loss.backward()

  # update parameters
  for layer in multi_layer_perceptron.layers:
    for neuron in layer.neurons:
      for weight_value in neuron.weight_values:
        weight_value.data -= weight_value.gradient * alpha
      neuron.bias_value.data -= neuron.bias_value.gradient * alpha

  return loss.data


def test_mse_loss_decreases_with_gradient_descent():
  '''On a fixed-seed MLP, MSE loss should drop after some gradient descent.'''
  random.seed(0)
  multi_layer_perceptron = MultiLayerPerceptron(
      layers=[
          get_layer(
              num_weights_per_neuron=3,
              num_neurons=4,
              activation=TANH,
          ),
          get_layer(
              num_weights_per_neuron=4,
              num_neurons=4,
              activation=TANH,
          ),
          get_layer(
              num_weights_per_neuron=4,
              num_neurons=1,
              activation=TANH,
          ),
      ],
  )

  inputs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
  ]
  targets = [[1.0], [-1.0], [-1.0], [1.0]]

  losses = [
      gradient_descent_step(multi_layer_perceptron, inputs, targets, alpha=0.01)
      for _ in range(100)
  ]

  # Loss at the end is meaningfully lower than at the start.
  assert losses[-1] < losses[0]
  # And it trends down rather than diverging.
  assert min(losses) == losses[-1]
