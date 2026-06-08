from abc import ABC
from abc import abstractmethod
import math


class Op(ABC):

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
  
  def to_string(self) -> str:
    return '+'
  
  def forward(self, inputs: list[float]) -> float:
    return sum(inputs)
  
  def backward(self, inputs: list[float]) -> list[float]:
    return [1.0 for _ in inputs]


class Multiply(Op):
  
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
    return [before * after for (before, after) in zip(product_before, product_after)]


class Exponentiate(Op):
  def to_string(self) -> str:
    return 'exponentiate'

  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == 2
    return inputs[0] ** inputs[1]

  def backward(self, inputs: list[float]) -> list[float]:
    assert len(inputs) == 2
    return [inputs[1] * self.forward([inputs[0], inputs[1] - 1])]


class Tanh(Op):
  
  def to_string(self) -> str:
    return 'tanh'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == 1
    # math.tanh saturates to +/-1 for large |x| instead of overflowing on e**(2x)
    return math.tanh(inputs[0])

  def backward(self, inputs: list[float]) -> list[float]:
    return [1 - self.forward(inputs) ** 2]


PLUS = Plus()
MULTIPLY = Multiply()
EXPONENTIATE = Exponentiate()
TANH = Tanh()
