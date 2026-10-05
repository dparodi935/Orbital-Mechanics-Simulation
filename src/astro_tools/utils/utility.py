import os, yaml
from typing import Any

def open_yaml_file(filepath:str) -> Any:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
 