from abc import ABC
from abc import abstractmethod
import math


class Op(ABC):

  @abstractmethod
  def to_string(self) -> str:
    raise NotImplementedError()
  
  @abstractmethod
  def forward(self, inputs: list[float]) -> float:
    raise NotImplementedError()
  
  @abstractmethod
  def backward(self, inputs: list[float]) -> float:
    '''
    Local derivative for each child given the data of its siblings
    '''
    raise NotImplementedError()
  

class Plus(Op):
  
  def to_string(self) -> str:
    return '+'
  
  def forward(self, inputs: list[float]) -> float:
    return sum(inputs)
  
  def backward(self, inputs: list[float]) -> float:
    return [1.0 for _ in inputs]


class Multiply(Op):
  
  def to_string(self) -> str:
    return '*'
  
  def forward(self, inputs: list[float]) -> float:
    result = 1.0
    for input in inputs:
      result *= input
    return result

  def backward(self, inputs: list[float]) -> float:
    product = self.forward(inputs)
    result = [product for _ in inputs]
    for i, child_data in enumerate(inputs):
      result[i] /= child_data
    return result


class Tanh(Op):
  
  def to_string(self) -> str:
    return 'tanh'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == 1
    input = inputs[0]
    return (math.e ** (2 * input) - 1) / (math.e ** (2 * input) + 1)

  def backward(self, inputs: list[float]) -> float:
    assert len(inputs) == 1
    input = inputs[0]
    return 1 - self.forward(input) ** 2


PLUS = Plus()
MULTIPLY = Multiply()
TANH = Tanh()
