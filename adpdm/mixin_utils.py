import os
import json
import inspect
from typing import Union, Dict, OrderedDict
import importlib
import functools
from torch import device as Device
from torch.nn import Module
from safetensors import torch as safetorch, safe_open

from .config import load_config, Configuration


__all__ = [
    "register_to_config",
    "mixin",
    "configmixin",
    "modelmixin",
    "hf_modelmixin",
    "ConfigMixin",
    "ModelMixin",
    "PipelineMixin"
]


def _parse_config_dict(cls, config_dict: Dict) -> Dict:
    # Remove the private attributes from the config
    private_kwargs = OrderedDict({k: v for k, v in config_dict.items() if k.startswith("_")})

    # Check the class name
    if "_class_name" in private_kwargs:
        cls_name = cls.__name__ if inspect.isclass(cls) else cls.__class__.__name__
        if cls_name != private_kwargs["_class_name"]:
            raise ValueError(f"Class name mismatch: {cls_name} != {private_kwargs['_class_name']}")

    # Check the module name
    if "_module_name" in private_kwargs:
        if cls.__module__ != private_kwargs["_module_name"]:
            raise ValueError(f"Module name mismatch: {cls.__module__} != {private_kwargs['_module_name']}")
    
    # Remove the private attributes from the config
    config_dict = OrderedDict({k: v for k, v in config_dict.items() if not k.startswith("_")})
    full_config = OrderedDict()
    init_kwargs = dict()

    # Get the signature of the __init__ method
    # Parse the config dict
    sub_class_kwargs = dict()
    for k, v in inspect.signature(cls.__init__).parameters.items():
        if k == "self":
            continue
        # If the config has the same key as the parameter, use the config value
        if k in config_dict:
            v = config_dict[k]
            if isinstance(v, dict):
                if "_module_name" in v and "_class_name" in v:
                    init_kwargs[k] = None
                    sub_class_kwargs[k] = v
                elif "_module_name" not in v and "_class_name" not in v:
                    init_kwargs[k] = v
                else:
                    raise ValueError(
                        f"Missing argument: '_class_name' or '_module_name' in argument."
                    )
            else:
                init_kwargs[k] = v
            full_config[k] = v
        elif v.default is inspect._empty:
            raise ValueError(f"Missing argument: {k}")
        else:
            init_kwargs[k] = full_config[k] = v.default

    # Get the full kwargs
    private_kwargs.update(full_config)
    return init_kwargs, private_kwargs, sub_class_kwargs
    

def _is_json_serializable(data) -> bool:
    try:
        json.dumps(data)
        return True
    except TypeError:
        return False
    

def register_to_config(init):
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        config_init_kwargs = OrderedDict()
        init_kwargs = OrderedDict()
        for k, v in kwargs.items():
            if issubclass(v.__class__, ConfigMixin):
                init_kwargs[k] = v
                config_init_kwargs[k] = v.config.dict
                continue

            if not _is_json_serializable(v):
                raise TypeError(f"The instance <{k}> is not serializable.")

            if k.startswith("_"):
                config_init_kwargs[k] = v
            else:
                init_kwargs[k] = v


        if "_class_name" not in config_init_kwargs:
            config_init_kwargs = {"_class_name": self.__class__.__name__, **config_init_kwargs}
        if "_module_name" not in config_init_kwargs:
            config_init_kwargs = {"_module_name": self.__module__, **config_init_kwargs}

        if not isinstance(self, ConfigMixin):
            raise ValueError(f"{self.__class__.__name__} must be a subclass of ConfigMixin")
        
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init).parameters.items()
        parameters = {name: p.default for i, (name, p) in enumerate(signature) if i > 0}

        for arg, name in zip(args, parameters.keys()):
            if issubclass(arg.__class__, ConfigMixin):
                arg = arg.config.dict
            elif not _is_json_serializable(arg):
                raise TypeError(f"The instance <{name}> is not serializable.")
            new_kwargs[name] = arg 

        new_kwargs.update(
            {
                k: init_kwargs.get(k, default) 
                for k, default in parameters.items()
                if k not in new_kwargs and k not in config_init_kwargs
            }
        )
    
        # Combine the private config with the new kwargs
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        self.config = Configuration(new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init


def mixin(cls, new_cls: type):
    if not inspect.isclass(cls):
        cls = cls.__class__
    if not issubclass(cls, new_cls):
        new_cls = type(cls.__name__, (new_cls, cls), dict(cls.__dict__))
        if issubclass(new_cls, ConfigMixin):
            new_cls.__init__ = register_to_config(new_cls.__init__)
        return new_cls
    return cls


def configmixin(cls) -> "ConfigMixin":
    return mixin(cls, ConfigMixin)


def modelmixin(cls) -> "ModelMixin":
    return mixin(cls, ModelMixin)


# def hf_modelmixin(cls) -> HfModelMixin:
#     new_cls = type(cls.__name__, (HfModelMixin, cls), dict(cls.__dict__))
#     new_cls.__init__ = hf_register_to_config(
#         new_cls.__init__.__wrapped__ 
#         if hasattr(new_cls.__init__, "__wrapped__") 
#         else new_cls.__init__
#     )
#     return new_cls


def _dynamic_import_module_and_interit(class_name: str, module_name: str):
    # Import the module dynamically and inherit from it
    module = importlib.import_module(module_name)
    module = getattr(module, class_name, None)
    if module is None:
        raise ValueError(f"Class {class_name} not found in module {module_name}.")
    if issubclass(module, ConfigMixin):
        pass
    elif issubclass(module, Module):
        module = modelmixin(module)
    else:
        module = configmixin(module)
    return module


def _try_load_file(directory: str, suffix: str) -> str:
    maybe_the_file = []

    # Check if the file exists
    for file in os.listdir(directory):
        if file.endswith(suffix):
            maybe_the_file.append(file)
    
    num = len(maybe_the_file)
    if num == 0:
        raise FileNotFoundError(f"No file with suffix {suffix} found in directory {directory}.")
    elif num > 1:
        raise ValueError(f"Multiple files with suffix {suffix} found in directory {directory}.")
    else:
        return os.path.join(directory, maybe_the_file[0])
    

class ConfigMixin(object):

    """
    A mixin class that provides a unified interface for loading models from a configuration file.
    The configuration file is a json file that contains the information about the model.
    The configuration file should contain the following information:
        - _class_name: The name of the class.
        - _module_name: The name of the module.
        - The rest of the information is the configuration of the model.
    """

    config: Configuration

    @classmethod
    def from_config(cls, config_file_or_dict: Union[str, Dict]):
        """
        Load the instance from a configuration file.

        Args:
            config_file_or_dict (Union[str, Dict]): 
                The path to the configuration file or the configuration dictionary.
        
                
        Example:
            >>> from seu.mixin_utils import ConfigMixin

            1. Automatically load the corresponding instance directly by the `ConfigMixin`
            >>> instance = ConfigMixin.from_config(<config.json>)
            
            2. Specify the class name and module name manually, the specified class must inherit from `ConfigMixin`
            >>> instance = MyClass.from_config(<config.json>)
        """

        # Convert to a configuration dictionary uniformly
        if isinstance(config_file_or_dict, str):
            config_dict = load_config(config_file_or_dict)
        elif isinstance(config_file_or_dict, dict):
            config_dict = OrderedDict(config_file_or_dict)
        else:
            raise ValueError(f"Unsupported config file type: {config_file_or_dict}")

        if cls is ConfigMixin:
            if "_class_name" not in config_dict or "_module_name" not in config_dict:
                raise ValueError(
                    "The config file doesn't contain the information about the configuration of the model. "
                    "Please define the model manually."
                )
            # Dynamically import the module
            cls = _dynamic_import_module_and_interit(
                class_name=config_dict["_class_name"],
                module_name=config_dict["_module_name"]
            )

        # Parse the configuration dictionary
        init_kwargs, config_dict, sub_class_kwargs = _parse_config_dict(cls, config_dict)
        for sub_cls_name, sub_cls_config in sub_class_kwargs.items():
            # Dynamically import the submodule
            submodule = _dynamic_import_module_and_interit(
                class_name=sub_cls_config["_class_name"],
                module_name=sub_cls_config["_module_name"]
            )

            # Recursively parse the sub-configuration dictionary
            init_kwargs[sub_cls_name] = submodule.from_config(sub_cls_config)
        
        # Initialization
        self = cls.__new__(cls)
        self.config = Configuration(config_dict)
        self.__init__(**init_kwargs)
        return self

    def save_config(self, filename: str) -> None:
        """
        Save the configuration of the model to a json file.

        Args:
            filename (str): The path to the configuration file.
        """
        new_config = OrderedDict()

        # If there are no private attributes, add them to the configuration dictionary
        # _class_name and _module_name are required
        if not hasattr(self.config, "_class_name"):
            new_config["_class_name"] = self.__class__.__name__
        
        if not hasattr(self.config, "_module_name"):
            new_config["_module_name"] = self.__module__

        if len(new_config) == 0: 
            self.config.save(filename)
        else:
            new_config.update(self.config.dict)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=4)


class ModelMixin(Module, ConfigMixin):
    """
    A mixin class that provides a unified interface for loading models from a configuration file.
    The configuration file is a json file that contains the information about the model.
    The configuration file should contain the following information:
        - _class_name: The name of the class.
        - _module_name: The name of the module.
        - The rest of the information is the configuration of the model.

    A class that inherits from `torch.nn.Module` can be transformed into a subclass of `ModelMixin` by using the `modelmixin` method.
    """
    @classmethod
    def from_pretrained(cls, pretrained_path: str, *, strict: bool = True):
        """
        Load the instance from a configuration file.

        Args:
            pretrained_path (str): 
                The path to the configuration file or the model weight file. 
                1. If `pretrained_path` is a folder, it is necessary to ensure that the folder contains at least
                    a configuration file *.json or a model weight file *.safetensors. The configuration file 
                    is the superiority. If the configuration file represented by the attribution `cls.config_name`
                    does not exist, it will be then check the most likely configuration under the pretrained path.
                    Else try to be automatically loaded using the default file name *.safetensors, premise that the 
                    weight file contains the configuration information in the meta content.
                2. If the `pretrained_path` is a file, it is necessary to ensure that the file contains the
                    configuration information in the meta content.
            strict (bool): 
                Whether to strictly match the names of the parameters in the model weight file.

        Example:
            >>> from seu.mixin_utils import ModelMixin

            1. Automatically load the corresponding instance directly by the `ModelMixin`.
            >>> instance = ModelMixin.from_pretrained(<model_path>)

            2. Specify the class name and module name manually, the specified class must inherit from `ModelMixin`
            >>> instance = MyClass.from_pretrained(<model_path>)
        
        """
        # Check the path
        assert os.path.exists(pretrained_path), \
            f"Model path {pretrained_path} does not exist."
        
        # When the current path is a folder, it is necessary to ensure that the folder contains 
        # a configuration file *.json or a model weight file *.safetensors
        if os.path.isdir(pretrained_path):
            # Try to load the config file
            # If config file does not exist, try to load the model file
            try:
                # First, check if there is a configuration file
                config_file = getattr(cls, "config_name", "config.json")
                config_file = os.path.join(pretrained_path, config_file)

                assert os.path.exists(config_file), \
                    f"Config file {config_file} does not exist."
                exist = True
            
            except:
                try:
                    config_file = _try_load_file(pretrained_path, ".json")
                    exist = True
                except:
                    exist = False
            
            if exist:
                # If the model is loaded using the ModelMixin base class
                # It is necessary to load the _class_name and _module_name in the configuration file 
                # to determine the class name and module name of the model
                # Then load the model
                if cls is ModelMixin:
                    config_dict = load_config(config_file)
                    module_name = config_dict.get("_module_name", None)
                    cls_name = config_dict.get("_class_name", None)
                    # If the configuration file does not contain _class_name and _module_name, 
                    # an error is reported
                    if cls_name is None or module_name is None:
                        raise ValueError(
                            "The config file doesn't contain the information about the configuration of the model. "
                            "Please define the model manually."
                        )
                    # If the module is a subclass of ModelMixin, create an instance 
                    # directly through the configuration file
                    module = _dynamic_import_module_and_interit(
                        class_name=cls_name,
                        module_name=module_name
                    )
                    instance: ModelMixin = module.from_config(config_file)
                    
                # If the module is a subclass of ModelMixin, create an instance directly through the configuration file
                else:
                    instance: ModelMixin = cls.from_config(config_file)
            # If there is no configuration file, check if there is 
            # configuration information in the model weight file
            else:
                # Try to load the model directly using the *.safetensors model
                # Since the class name and module name of the model are unknown, 
                # we can only try to load the model using the default file name
                try:
                    try:
                        ckpt_name = os.path.join(
                            pretrained_path, getattr(cls, "_ckpt_name", "pytorch_model.safetensors")
                        )
                        assert os.path.exists(ckpt_name), \
                            f"Model weight file {ckpt_name} does not exist."
                    except:
                        ckpt_name = _try_load_file(pretrained_path, ".safetensors")
                        return cls.from_pretrained(ckpt_name)
                except:
                    raise AttributeError(
                        "The path doesn't contain the information about the configuration of the model. "
                        "Please define the model manually."
                    )
                    
            # Load the model weight file
            weight_file = os.path.join(
                pretrained_path, getattr(instance, "_ckpt_name", "pytorch_model.safetensors")
            )
        
        # If the path is a file and is loaded in class name form
        # Since there is no configuration file, the corresponding object must 
        # be instantiated based on the configuration information in the model file
        else:
            with safe_open(pretrained_path, framework="pt") as f:
                metadata = f.metadata()
            if (metadata is None) or ("config" not in metadata):
                raise AttributeError(
                    "The file doesn't contain the information about the configuration of the model. "
                    "Please define the model manually."
                )
            config = json.loads(metadata["config"])
            module_name = config.get("_module_name", None)
            cls_name = config.get("_class_name", None)
            # If the configuration file does not contain _class_name and _module_name, 
            # an error is reported
            if cls_name is None or module_name is None:
                raise ValueError(
                    "The config file doesn't contain the information about the configuration of the model. "
                    "Please define the model manually."
                )
            
            if cls is ModelMixin:
                module = _dynamic_import_module_and_interit(
                    class_name=cls_name,
                    module_name=module_name 
                )
            else:
                if cls.__name__ != cls_name:
                    raise TypeError(
                        f"[{cls.__name__}]: The class name: <{cls_name}> in file is not matching."
                    )
                if cls.__module__ != module_name:
                    raise TypeError(
                        f"[{cls.__name__}]: The module name: <{module_name}> in file is not matching."
                    )
                module = modelmixin(cls)
                
            instance = module.from_config(config)
            weight_file = pretrained_path

        if not os.path.exists(weight_file):
            raise ValueError(f"Model weight file {weight_file} does not exist.")
        
        missing, unexcepted = safetorch.load_model(instance, weight_file, strict)
        if len(missing) > 0:
            raise Warning(f"Get the missing attribution: {missing}")
        if len(unexcepted):
            raise Warning(f"Get the unexcepted attribution: {unexcepted}")
        instance.eval()
        return instance

    def save_pretrained(self, pretrained_path: str, *, metaconfig: bool = True) -> None:
        """
        Save the model weight file.

        Args:
            pretrained_path (str):
                The path to the model weight file.
            metaconfig (bool):
                Whether to save the configuration information in the meta content.
        """
        if not pretrained_path.endswith(".safetensors"):
            os.makedirs(pretrained_path, exist_ok=False)
            config_path = os.path.join(pretrained_path, getattr(self, "config_name", "config.json"))
            self.config.save(config_path)
            pretrained_path = os.path.join(
                pretrained_path, getattr(self, "_ckpt_name", "pytorch_model.safetensors")
            )

        metadata = {"config": str(self.config)} if metaconfig else None
        safetorch.save_model(self, pretrained_path, metadata=metadata)


class PipelineMixin(ConfigMixin):
    """
    A mixin class that provides a unified interface for loading models from a configuration file.
    The configuration file is a json file that contains the information about the model.
    The configuration file should contain the following information:
        - _class_name: The name of the class.
        - _module_name: The name of the module.
        - The configuration of the sub-components.
        - The rest of the information is the configuration of the pipeline.
    """
    _registered_modules = []

    device: Device = Device("cpu")

    def to(self, device: Union[str, Device]):
        if not isinstance(device, (str, Device)):
            raise TypeError(
                f"Device must be a str or torch.device, but got {type(device)}." 
            )
        device = device if isinstance(device, Device) else Device(device)
        if device == self.device:
            return self
        
        self.device = device
        for sub_component in self._registered_modules:
            component = getattr(self, sub_component)
            if isinstance(component, (PipelineMixin, ModelMixin)):
                component.to(device)
        return self
            
    def cuda(self):
        return self.to("cuda")
    
    def enable_cpu_offload(self) -> None:
        raise NotImplementedError

    def register_modules(self, **kwargs) -> None:
        """
        Register the sub-components of the pipeline.

        Args:
            kwargs (Dict):
                The sub-components of the pipeline.
                The key is the name of the sub-component.
                The value is the instance of the sub-component.
        """
        for k, v in kwargs.items():
            setattr(self, k, v) 
            self._registered_modules.append(k)

    def save_pretrained(self, pretrained_path: str) -> None:

        """
        Save the pipeline. The structure of the saving folder is as follows:

        ```
        pretrained_path
        ├── config.json
        ├── sub_component_1
        │   ├── config.json
        │   └── pytorch_model.safetensors
        ├── sub_component_2
        │   ├── config.json
        └── ...
        ```

        Sub-components are the registered modules in the initialzation method of the pipeline.

        Args:
            pretrained_path (str):
                The path to the pipeline.
        """

        # Create the folder
        os.makedirs(pretrained_path, exist_ok=False)

        # Save the configuration file
        config_file = os.path.join(
            pretrained_path, getattr(self, "config_name", "config.json")
        )
        self.config.save(config_file)

        # Save the model weight file
        for sub_component in self._registered_modules:
            component = getattr(self, sub_component)

            if isinstance(component, ModelMixin):
                component.save_pretrained(
                    os.path.join(pretrained_path, sub_component)
                )
            elif isinstance(component, Module):
                component = modelmixin(component)
                component.save_pretrained(
                    os.path.join(pretrained_path, sub_component)
                )
            elif isinstance(component, ConfigMixin):
                config_name = getattr(component, "config_name", "config.json")
                component.save_config(os.path.join(pretrained_path, sub_component, config_name))
            else:
                component = configmixin(component)
                component.save_config(os.path.join(pretrained_path, sub_component, "config.json"))


    @classmethod
    def from_pretrained(cls, pretrained_path: str, *, strict: bool = True):
        """
        Load the pipeline. 
        
        Args:
            pretrained_path (str):
                The path to the pipeline.
            strict (bool):
                Whether to strictly match the names of the parameters in the model weight file.

        Example:
            >>> from seu.mixin_utils import PipelineMixin

            1. Automatically load the corresponding instance directly by the `PipelineMixin`.
            >>> instance = PipelineMixin.from_pretrained(<model_path>)

            2. Specify the class name and module name manually, the specified class must inherit from `PipelineMixin`
            >>> instance = MyClass.from_pretrained(<model_path>)
        """

        # Check the path
        assert os.path.isdir(pretrained_path), \
            f"Model path {pretrained_path} must be an existing directory."

        try:
            config_file = os.path.join(pretrained_path, getattr(cls, "config_name", "config.json"))
            assert os.path.exists(config_file), \
                f"Model path <{pretrained_path}> does not contain the configuration file {config_file}."
        except:
            config_file = _try_load_file(pretrained_path, ".json")

        if cls is PipelineMixin:
            # Load the configuration file mannually
            config_dict = load_config(config_file)
            module_name = config_dict.get("_module_name", None)
            cls_name = config_dict.get("_class_name", None)
            # If the configuration file does not contain _class_name and _module_name,
            # an error is reported
            if cls_name is None or module_name is None:
                raise ValueError(
                    "The config file doesn't contain the information about the configuration of the model. "
                    "Please define the model manually."
                )
            # Dynamically import the module
            cls = _dynamic_import_module_and_interit(
                class_name=cls_name,
                module_name=module_name
            )
            
            # new instance
            instance = cls.from_config(config_file)

        else:
            # Load the configuration file
            instance = cls.from_config(config_file)

        # Get pretrained sub-components
        for sub_component in instance._registered_modules:
            sub_component_path = os.path.join(pretrained_path, sub_component)
            if not os.path.exists(sub_component_path):
                raise ValueError(
                    f"Model path {pretrained_path} does not contain the sub-component {sub_component}."
                )
            component = getattr(instance, sub_component)

            if isinstance(component, ModelMixin):
                try:
                    weight_file = os.path.join(
                        pretrained_path, sub_component, getattr(component, "_ckpt_name", "pytorch_model.safetensors")
                    )

                    if not os.path.exists(weight_file):
                        raise ValueError(f"Model weight file {weight_file} does not exist.")
                except:
                    weight_file = _try_load_file(sub_component_path, ".safetensors")
                
                missing, unexcepted = safetorch.load_model(component, weight_file, strict)
                if len(missing) > 0:
                    raise Warning(f"Get the missing attribution: {missing}")
                if len(unexcepted):
                    raise Warning(f"Get the unexcepted attribution: {unexcepted}")
                component.eval()
        return instance

