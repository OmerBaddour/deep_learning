from src.deep_learning.layer import Layer
from src.deep_learning.neuron import Neuron
from src.deep_learning.multi_layer_perceptron import MultiLayerPerceptron
import pytest

def test_basic():
  multi_layer_perceptron = MultiLayerPerceptron(
      layers=[
          Layer(
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
          ),
          Layer(
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
          ),
          Layer(
              neurons=[
                  Neuron(
                      weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      bias=5.0,
                  ),
              ],
          ),
      ],
  )
  multi_layer_perceptron.forward(
      [
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
          [1.0, 2.0, 3.0],
      ],
  )
  multi_layer_perceptron.backward()
