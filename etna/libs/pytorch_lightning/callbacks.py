
from etna.core.utils import init_collector
from copy import deepcopy

from pytorch_lightning.callbacks import __all__ as pl_callbacks

generated_classes = []

for class_name in pl_callbacks:
    
    class_ = deepcopy(getattr(__import__('pytorch_lightning.callbacks', fromlist=[class_name]), class_name))
    # exec(f"{class_name} = type('{class_name}', class_.__bases__, dict(class_.__dict__))")
    # exec(f"{class_name} = new_class('{class_name}', (class_,))")
    # exec(f"class {class_name}(class_): pass")
    
    new_class = type('{class_name}', class_.__bases__, dict(class_.__dict__))
    
    globals()[class_name] = new_class
    if hasattr(class_, '__init__'):
        setattr(new_class, '__init__', init_collector(globals()[class_name].__init__))
    generated_classes.append(new_class)

__all__ = generated_classes