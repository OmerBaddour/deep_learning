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
    raise NotImplementedError()
  

class Plus(Op):
  
  def to_string(self) -> str:
    return '+'
  
  def forward(self, inputs: list[float]) -> float:
    return sum(inputs)
  
  def backward(self) -> float:
    return 1.0


class Multiply(Op):
  
  def to_string(self) -> str:
    return '*'
  
  def forward(self, inputs: list[float]) -> float:
    result = 1.0
    for input in inputs:
      result *= input
    return result

  def backward(self) -> float:
    pass


class Tanh(Op):
  
  def to_string(self) -> str:
    return 'tanh'
  
  def forward(self, inputs: list[float]) -> float:
    assert len(inputs) == 1
    input = inputs[0]
    return (math.e ** input - math.e ** (-input)) / (math.e ** input + math.e ** (-input))

  def backward(self) -> float:
    pass


PLUS = Plus()
MULTIPLY = Multiply()
TANH = Tanh()
