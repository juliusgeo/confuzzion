import main
from main import transform_func, my_extension_sleep_random
import requests


def recursively_patch(object):
    if hasattr(object, "__call__"):
        for k, v in object.__dict__.items():
            if callable(v):
                setattr(object, k, transform_func(recursively_patch(v), my_extension_sleep_random))
    return object

for k, v in requests.__dict__.items():
    setattr(requests, k, recursively_patch(v))




