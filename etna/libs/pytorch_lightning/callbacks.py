
from etna.core.utils import init_collector
from copy import deepcopy

from pytorch_lightning.callbacks import __all__

for class_name in __all__:
    
    class_ = deepcopy(getattr(__import__('pytorch_lightning.callbacks', fromlist=[class_name]), class_name))
    # exec(f"{class_name} = type('{class_name}', class_.__bases__, dict(class_.__dict__))")
    # exec(f"{class_name} = new_class('{class_name}', (class_,))")
    exec(f"class {class_name}(class_): pass")
    if hasattr(class_, '__init__'):
        setattr(globals()[class_name], '__init__', init_collector(globals()[class_name].__init__))