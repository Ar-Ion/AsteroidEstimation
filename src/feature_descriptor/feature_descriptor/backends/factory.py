import importlib

def instance(params, mode, size):
    module = importlib.import_module("feature_descriptor.backends")
    backend_class = getattr(module, params["type"])
    backend = backend_class()
    return backend