import importlib

def instance(params, client, server, size):
    module = importlib.import_module("feature_descriptor.backends")
    backend_class = getattr(module, params["type"]+"Backend")
    backend = backend_class(client, server, size, params)
    return backend