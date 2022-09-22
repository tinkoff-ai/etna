
from etna.core.utils import init_collector
from copy import deepcopy

from pytorch_lightning.callbacks import __all__ as pl_callbacks

generated_classes = []

for class_name in pl_callbacks:
    
    class_ = deepcopy(getattr(__import__('pytorch_lightning.callbacks', fromlist=[class_name]), class_name))
    new_class = type(f'{class_name}', (class_,), {**dict(class_.__dict__), "__module__": __name__})
    
    globals()[class_name] = new_class
    if hasattr(class_, '__init__'):
        setattr(new_class, '__init__', init_collector(globals()[class_name].__init__))
    generated_classes.append(new_class)

__all__ = generated_classes