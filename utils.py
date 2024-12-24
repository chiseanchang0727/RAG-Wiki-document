from typing import Type, TypeVar
from pydantic import BaseModel
import yaml


T = TypeVar('T', bound=BaseModel)

def load_config_from_yaml(file_path: str, model: Type[T]) -> T:

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return model(**yaml_data['Configs'])