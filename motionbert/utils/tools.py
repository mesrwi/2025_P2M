import os
import pickle
import yaml
import json
from easydict import EasyDict
from typing import Any, IO

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialize Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    # '!include' 태그가 붙어 있으면 -> construct_include 함수를 써서 파싱
    # ex) file_content: !include path/to/file.yaml
    yaml.add_constructor('!include', construct_include, Loader)
    
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = EasyDict(config)
    
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config