# %%
from __future__ import annotations
import numpy as np
from PIL import Image
from IPython.display import display

# %%

NUM_ROWS = 28
NUM_COLS = 28
NUM_LABELS = 10

class Data:
  def __init__(self, pixels: list[list[int]], label: int):
    self.pixels = pixels
    self.flattened_pixels: list[int] = []
    for row in self.pixels:
      assert len(row) == NUM_ROWS
      self.flattened_pixels.extend(row)
    self.flattened_pixels = [pixel / 255.0 for pixel in self.flattened_pixels]
    
    self.label = label
  
  @classmethod
  def from_csv_row(cls, csv: list[int]) -> Data:
    label = csv[0]
    pixels: list[list[int]] = []
    for row in range(NUM_ROWS):
      start_index = 1 + row * NUM_ROWS
      pixels.append(csv[start_index : start_index + NUM_COLS])

    return cls(pixels, label)
  
  def to_image(self) -> Image:
    pixel_array = np.array(self.pixels, dtype=np.uint8)
    return Image.fromarray(pixel_array, mode='L')
  
  def label_to_one_hot(self) -> list[int]:
    result = [0] * NUM_LABELS
    result[self.label] = 1
    return result

# %%
lines: list[str] = []
datas: list[Data] = []

with open('data/MNIST_CSV/mnist_train.csv') as f:
  lines = [f.readline()]

for line in lines:
  datas.append(Data.from_csv_row([int(num) for num in line.split(',')]))
  break

# %%
distinct_labels = set()
for data in datas:
  distinct_labels.add(data.label)

# %%
distinct_labels

# %% [markdown]
# ## Train

# %%
from deep_learning.layer import get_layer
from deep_learning.multi_layer_perceptron import MultiLayerPerceptron
from deep_learning.op import TANH
from deep_learning.value import Value

from deep_learning.util import xavier_uniform
from deep_learning.util import zero_all_gradients
from deep_learning.util import softmax
from deep_learning.util import cross_entropy
from datetime import datetime

# %%
# build graph

input_values = [Value(0.0, label=f'pixel_{i}') for i in range(NUM_ROWS*NUM_COLS)]
output_values = [Value(0.0, label=f'label_{i}') for i in range(NUM_LABELS)]

multi_layer_perceptron = MultiLayerPerceptron(
    layers=[
        get_layer(
            num_weights_per_neuron=NUM_ROWS*NUM_COLS,
            num_neurons=32,
            activation=TANH,
            fn_weight_initializer=lambda: xavier_uniform(NUM_ROWS*NUM_COLS, 32),
        ),
        get_layer(
            num_weights_per_neuron=32,
            num_neurons=NUM_LABELS,
        ),
    ],
)
predicted_output = multi_layer_perceptron.build_graph(input_values)
softmax_output = softmax(predicted_output)
loss = cross_entropy(
    distribution=output_values,
    predicted_distribution=softmax_output,
)

# %%
# topologically sort graph
topologically_sorted_graph: list[Value] = []
visited: set[Value] = set()

def _traverse(
    node: Value,
    topologically_sorted_graph: list[Value],
    visited: set[Value],
) -> None:
  # NOTE: assume acyclic for simplicity
  if node in visited:
    return
  elif len(node.children) == 0:
    visited.add(node)
    topologically_sorted_graph.append(node)
  else:
    for child in node.children:
      _traverse(child, topologically_sorted_graph, visited)
    visited.add(node)
    topologically_sorted_graph.append(node)
_traverse(loss, topologically_sorted_graph, visited)

# %%
# `Value.forward()` recursively populates `node.data`. since `node.data` is already set for leaf nodes, we omit them
forward_order = [node for node in topologically_sorted_graph if len(node.children) > 0]

# `Value.backward()` recursively populates `node.gradient`. since leaf nodes get their gradients populated by their parents, we omit them
# this decomposes to just reversed `forward``_order`
backward_order = list(reversed(forward_order))

# %%
# prepare graph
data = datas[0]
for input_value, pixel in zip(input_values, data.flattened_pixels, strict=True):
  input_value.data = float(pixel)
for output_value, label_one_hot in zip(output_values, data.label_to_one_hot(), strict=True):
  output_value.data = label_one_hot

# %%
import argparse, cProfile, io, pstats, time
from pstats import SortKey

def run() -> None:
  # instead of `loss.forward()` we do
  for node in forward_order:
    node.data = node.op.forward([child.data for child in node.children])

  # backwards
  # instead of `zero_all_gradients(loss)` we do
  for node in topologically_sorted_graph:
    node.gradient = 0

  loss.gradient = 1.0
  # instead of `loss.backward()` we do
  for node in backward_order:
    children_local_derivatives = node.op.backward([child.data for child in node.children])
    zipped: list[tuple[Value, float]] = zip(node.children, children_local_derivatives, strict=True)
    for child, local_derivative in zipped:
      child.gradient += local_derivative * node.gradient

def _time(iterations: int) -> float:
  start = time.perf_counter()
  for _ in range(iterations):
    run()
  return time.perf_counter() - start

def main() -> None:
  parser = argparse.ArgumentParser(description='profile or benchmark the single-traversal forward + backward pass')
  parser.add_argument('mode', nargs='?', choices=('profile', 'benchmark'), default='profile')
  parser.add_argument('-n', '--repeat', type=int, default=1, help='iterations to run')
  # `parse_known_args` so the script still works when run cell-by-cell in a notebook
  args, _ = parser.parse_known_args()

  if args.mode == 'profile':
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.repeat):
      run()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE).print_stats()
    print(s.getvalue())
  else:
    # differential measurement, as done by nanoBench (https://d-nb.info/1212853466/34):
    # time `n` iterations and `2 * n` iterations, then report `(t_2n - t_n) / n`.
    # both runs pay the same fixed overhead (timer calls, loop setup, cache/branch
    # predictor warm-up), so subtracting cancels it out and leaves just the marginal
    # cost of `n` more iterations. the naive `t_n / n` includes that overhead.
    run()  # warm up caches / lazily-initialized state before either timed run

    t_n = _time(args.repeat)
    t_2n = _time(2 * args.repeat)

    print(f'iterations:   {args.repeat}')
    print(f't(n):         {t_n * 1e3:.3f} ms')
    print(f't(2n):        {t_2n * 1e3:.3f} ms')
    print(f'naive mean:   {t_n / args.repeat * 1e6:.3f} us/iter')
    print(f'differential: {(t_2n - t_n) / args.repeat * 1e6:.3f} us/iter')

if __name__ == '__main__':
  main()
