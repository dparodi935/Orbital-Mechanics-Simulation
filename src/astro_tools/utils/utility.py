import os, yaml
from typing import Any

def open_yaml_file(folder_path:str, name:str) -> Any:
    config_file_path = os.path.join(folder_path,f'{name}.yaml')
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)
 