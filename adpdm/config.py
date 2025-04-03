from collections import OrderedDict
from typing import Dict, Union

import json
import yaml


__all__ = ["load_config", "Configuration"]


def load_config(config: str) -> OrderedDict:
    """
    Load a configuration file and return its content as an OrderedDict.

    Args:
        config (str): The path to the configuration file.

    Returns:
        OrderedDict: The content of the configuration file.

    Raises:
        ValueError: If the file type is not supported.
    """
    with open(config, "r") as f:
        if config.endswith(".json"):
            config = json.load(f)
        elif config.endswith(".yaml"):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError(f"Unsupported config file type: {config}")
    return OrderedDict(config)


class Configuration(OrderedDict):
    """
    A custom configuration class that inherits from OrderedDict.
    It loads configuration from a file or a dictionary and freezes itself to prevent modification.
    """
    def __init__(self, config: Union[str, Dict]) -> None:
        if isinstance(config, str):
            config = load_config(config)
        super(Configuration, self).__init__(config)

        for k, v in self.items():
            if isinstance(v, dict):
                v = Configuration(v)
            setattr(self, k, v)

        self.__frozen = True

    @classmethod
    def load(cls, config: str) -> "Configuration":
        """
        Load a configuration from a file and create a new Configuration instance.

        Args:
            config (str): The path to the configuration file.

        Returns:
            Configuration: A new Configuration instance with the loaded configuration.
        """
        config = load_config(config)
        cfg = Configuration.__new__(cls)
        cfg.__init__(config)
        return cfg

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super(Configuration, self).__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super(Configuration, self).__setitem__(name, value)

    def __repr__(self) -> str:
        return json.dumps(self, indent=4)
    
    @property
    def dict(self) -> OrderedDict:
        return OrderedDict(json.loads(self.__repr__()))
    
    def save(self, filename: str) -> None:
        """
        Set an item in the configuration.

        Args:
            name: The key of the item.
            value: The value of the item.

        Raises:
            Exception: If the configuration is frozen and modification is not allowed.
        """
        with open(filename, "w", encoding="utf-8") as f:
            if filename.endswith(".json"):
                json.dump(self, f, indent=4)
            elif filename.endswith(".yaml"):
                yaml.dump(self, f, indent=4)
            else:
                raise ValueError(f"Unsupported config file type: {filename.split('.')[-1]}")
