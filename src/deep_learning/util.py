from typing import Any

def is_numeric(x: Any) -> bool:
  try:
    float(x)
    return True
  except:
    return False
