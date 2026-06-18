from abc import ABC
from abc import abstractmethod
import math


class Op(ABC):

  @property
  @abstractmethod
  def arity(self) -> int | None:
    '''How many inputs this op takes; None means variadic (any number).'''
    raise NotImplementedError()

  @abstractmethod
  def to_string(self) -> str:
    raise NotImplementedError()
  
  def __repr__(self) -> str:
    return self.to_string()
  
  @abstractmethod
  def forward(self, inputs: list[float]) -> float:
    raise NotImplementedError()
    
  
  @abstractmethod
  def backward(self, inputs: list[float]) -> list[float]:
    '''
    Local derivative for each child given the data of its siblings
    '''
    raise NotImplementedError()
  

class Plus(Op):

  arity = None

  def to_string(self) -> str:
    return '+'
  
  def forward(self, inputs: list[float]) -> float:
    return sum(inputs)
  
  def backward(self, inputs: list[float]) -> list[float]:
    return [1.0 for _ in inputs]


class Minus(Op):

  arity = 2

  def to_string(self) -> str:
    return '-'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == self.arity
    return inputs[0] - inputs[1]
  
  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    return [1.0, -1.0]


class Multiply(Op):

  arity = None

  def to_string(self) -> str:
    return '*'
  
  def forward(self, inputs: list[float]) -> float:
    result = 1.0
    for input in inputs:
      result *= input
    return result

  def backward(self, inputs: list[float]) -> list[float]:
    '''
    Identical problem to https://leetcode.com/problems/product-of-array-except-self/
    '''
    product_before = [1.0] * len(inputs)
    product_after = [1.0] * len(inputs)
    for i in range(1, len(inputs)):
      product_before[i] = product_before[i-1] * inputs[i-1]
    for i in range(len(inputs)-2, -1, -1):
      product_after[i] = product_after[i+1] * inputs[i+1]
    return [before * after for (before, after) in zip(product_before, product_after, strict=True)]


class Power(Op):
  '''
  Raise a variable to a constant power
  '''

  arity = 1

  def __init__(self, exponent: float | int):
    self.exponent = exponent

  def to_string(self) -> str:
    return 'power'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == self.arity
    return inputs[0] ** self.exponent
  
  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    return [self.exponent * Power(self.exponent - 1).forward([inputs[0]])]


class Exponentiate(Op):
  '''
  Raise a variable to a variable power
  '''

  arity = 2

  def to_string(self) -> str:
    return 'exponentiate'

  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == self.arity
    return inputs[0] ** inputs[1]

  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    # forward(a, b) -> a ** b
    # backward(a, b) -> [d/da, d/db]
    # d/da a ** b = b * a ** (b-1)
    # d/db a ** b = log_e(a) * a ** b
    return [
        inputs[1] * self.forward([inputs[0], inputs[1] - 1]),
        math.log(inputs[0]) * self.forward(inputs),
    ]


class Logarithm(Op):

  arity = 2

  def to_string(self) -> str:
    return 'logarithm'
  
  def forward(self, inputs: list[float]) -> float:
    '''
    inputs[0] = x, inputs[1] = base
    '''
    assert len(inputs) == self.arity
    return math.log(
        inputs[0],  # x
        inputs[1],  # base
    )
  
  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    # forward(a, b) -> log_b(a)
    # backward(a, b) -> [d/da, d/db]
    # d/da log_b(a) = 1 / a * log_e(b)
    # d/db log_b(a) = - log_e(a) / b * (log_e(b) ** 2)
    return [
        1 / (inputs[0] * math.log(inputs[1])),
        -math.log(inputs[0]) / inputs[1] * (math.log(inputs[1]) ** 2),
    ]


class Divide(Op):

  arity = 2

  def to_string(self) -> str:
    return 'divide'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == self.arity
    return inputs[0] / inputs[1]
  
  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    # forward(a, b) -> a / b
    # backward(a, b) -> [d/da, d/db]
    # d/da a / b = 1 / b
    # d/db a / b = - a / b ** 2
    return [
        1.0 / inputs[1],
        -inputs[0] / inputs[1] ** 2,
    ]


class Tanh(Op):

  arity = 1

  def to_string(self) -> str:
    return 'tanh'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == self.arity
    # math.tanh saturates to +/-1 for large |x| instead of overflowing on e**(2x)
    return math.tanh(inputs[0])

  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == self.arity
    return [1 - self.forward(inputs) ** 2]


PLUS = Plus()
MINUS = Minus()
MULTIPLY = Multiply()
EXPONENTIATE = Exponentiate()
DIVIDE = Divide()
LOGARITHM = Logarithm()
TANH = Tanh()
